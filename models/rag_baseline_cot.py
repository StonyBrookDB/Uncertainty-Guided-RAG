# region imports, models, constants
import os
import torch
import pandas as pd
import time as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
model.eval()

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", trust_remote_code=True, device="cuda")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")

client = MilvusClient(uri="http://localhost:19530")
client.load_collection("MedRAG_combined_collection")
COLLECTION = "MedRAG_combined_collection"

option_tokens = {
    "A": tokenizer.encode("A", add_special_tokens=False)[0],
    "B": tokenizer.encode("B", add_special_tokens=False)[0],
    "C": tokenizer.encode("C", add_special_tokens=False)[0],
    "D": tokenizer.encode("D", add_special_tokens=False)[0],
    "E": tokenizer.encode("E", add_special_tokens=False)[0],
}

errors = 0
USE_RERANKER = True  # flip to True to rerank retrieved chunks before keeping the top 5

# --- CoT fallback settings ---
MARGIN_THRESHOLD = 0.15   # if (top1_prob - top2_prob) < this, re-run with CoT reasoning
COT_MAX_NEW_TOKENS = 300  # budget for the free-form reasoning generation
# endregion


def _run_model(messages):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    torch.cuda.synchronize()
    start = t.time()
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    torch.cuda.synchronize()
    end = t.time()

    last_token_logits = outputs.logits[0, -1, :]
    return inputs, last_token_logits, end - start


def _check_confident(inputs, last_token_logits):
    global errors
    next_token_id = torch.argmax(last_token_logits).item()
    if next_token_id not in option_tokens.values():
        next_token_text = tokenizer.decode(next_token_id)
        print(f"Warning: model's top token is not an option: {next_token_text}")
        errors += 1
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=100)
        print(tokenizer.decode(generated[0], skip_special_tokens=True))
        return False
    return True


def _option_probs(last_token_logits, letters):
    """Softmax over just the option-letter logits, in `letters` order."""
    option_logits = [last_token_logits[option_tokens[letter]].item() for letter in letters]
    probs = torch.softmax(torch.tensor(option_logits), dim=0).tolist()
    return probs


def _margin(probs):
    """Difference between the top-1 and top-2 probabilities."""
    sorted_probs = sorted(probs, reverse=True)
    return sorted_probs[0] - sorted_probs[1]


def _run_cot(context, question_block, letters, system_hint):
    """
    Two-pass CoT fallback:
      1) Ask the model to reason step-by-step over the context/question (free generation).
      2) Feed that reasoning back in and force a single constrained-letter answer,
         reusing the same next-token-logit approach as the main pass so the
         returned probs stay comparable to the first pass.

    Returns (last_token_logits, reasoning_text, elapsed_seconds).
    """
    start = t.time()

    user_content = f"""Context:
{context}
Question: {question_block}
"""

    reasoning_messages = [
        {"role": "system", "content": f"Use the provided context and step-by-step clinical reasoning to work through this question. {system_hint} Think through the relevant facts, then reason toward an answer."},
        {"role": "user", "content": user_content}
    ]

    cot_inputs = tokenizer.apply_chat_template(
        reasoning_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated = model.generate(
            **cot_inputs,
            max_new_tokens=COT_MAX_NEW_TOKENS,
            do_sample=False,
        )
    reasoning_text = tokenizer.decode(
        generated[0][cot_inputs.shape[-1]:], skip_special_tokens=True
    )

    letters_str = "/".join(letters)
    final_messages = reasoning_messages + [
        {"role": "assistant", "content": reasoning_text},
        {"role": "user", "content": f"Based on the reasoning above, output your final answer as a single letter ({letters_str}). No explanation.\nAnswer: "}
    ]

    final_inputs, last_token_logits, _ = _run_model(final_messages)
    _check_confident(final_inputs, last_token_logits)

    return last_token_logits, reasoning_text, t.time() - start


def get_context(prompt, use_reranker=USE_RERANKER):
    start = t.time()
    query_embedding = embedding_model.encode(prompt, normalize_embeddings=True).tolist()
    search_limit = 15 if use_reranker else 5

    search = client.search(
        collection_name=COLLECTION,
        data=[query_embedding],
        limit=search_limit,
        output_fields=["id", "source", "content"],
        search_params={"metric_type": "COSINE", "params": {}}
    )

    results = []
    for r in search[0]:
        results.append({
            "score": r["distance"],
            "id": r["id"],
            "source": r["entity"].get("source"),
            "content": r["entity"].get("content")
        })

    if use_reranker:
        pairs = [(prompt, r["content"]) for r in results]
        rerank_scores = reranker.predict(pairs, show_progress_bar=False, batch_size=search_limit)
        for r, score in zip(results, rerank_scores):
            r["rerank_score"] = score
        results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:5]

    return "\n\n".join([f"{r['content']}" for r in results]), results, t.time() - start


# returns [[probA, probB, probC, probD], time, confident flag, search_results, used_cot flag]
def eval_medmcqa(q):
    letters = ["A", "B", "C", "D"]
    question_block = f"""{q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}"""

    context, search_results, elapsed_retrieval = get_context(f"Question: {question_block}", use_reranker=USE_RERANKER)

    messages = [
        {"role": "system", "content": "Use provided context and reasoning to guide your answer. Output your answer as a single letter (A, B, C, D). No explanation"},
        {"role": "user", "content": f"""Context: 
{context}
Question: {question_block}
Answer: """}
    ]

    inputs, last_token_logits, elapsed_model = _run_model(messages)
    confident = _check_confident(inputs, last_token_logits)

    probs = _option_probs(last_token_logits, letters)
    total_time = elapsed_retrieval + elapsed_model
    used_cot = False

    if _margin(probs) < MARGIN_THRESHOLD:
        used_cot = True
        last_token_logits, _reasoning_text, elapsed_cot = _run_cot(
            context, question_block, letters,
            system_hint="Options are A, B, C, D."
        )
        probs = _option_probs(last_token_logits, letters)
        total_time += elapsed_cot

    return [probs, total_time, confident, search_results, used_cot]


# returns [[probA, probB, probC, probD, probE], time, confident flag, search_results, used_cot flag]
def eval_medqa(q):
    letters = ["A", "B", "C", "D", "E"]
    question_block = f"""{q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}
    E) {q["e"]}"""

    context, search_results, elapsed_retrieval = get_context(f"Question: {question_block}", use_reranker=USE_RERANKER)

    messages = [
        {"role": "system", "content": "Use provided context and reasoning to guide your answer. Output your answer as a single letter (A, B, C, D, E). No explanation"},
        {"role": "user", "content": f"""Context: 
{context}
Question: {question_block}
Answer: """}
    ]

    inputs, last_token_logits, elapsed_model = _run_model(messages)
    confident = _check_confident(inputs, last_token_logits)

    probs = _option_probs(last_token_logits, letters)
    total_time = elapsed_retrieval + elapsed_model
    used_cot = False

    if _margin(probs) < MARGIN_THRESHOLD:
        used_cot = True
        last_token_logits, _reasoning_text, elapsed_cot = _run_cot(
            context, question_block, letters,
            system_hint="Options are A, B, C, D, E."
        )
        probs = _option_probs(last_token_logits, letters)
        total_time += elapsed_cot

    return [probs, total_time, confident, search_results, used_cot]


def write(res, OUT_FILE):
    df = pd.DataFrame(res)
    file_exists = os.path.exists(OUT_FILE)
    df.to_csv(OUT_FILE, mode="a", header=not file_exists, index=False)


# Warmup
print("Warming up model")
warmup_messages = [
    {"role": "system", "content": "Output your answer as a single letter (A, B, C, D). No explanation"},
    {"role": "user", "content": """Question: What is the capital of France?
    A) London
    B) Paris
    C) Berlin
    D) Madrid
Answer: """}
]
for _ in range(5):
    _run_model(warmup_messages)
torch.cuda.synchronize()

# MEDQA
print("Evaluating MedQA test")
results = []
df = pd.read_csv("question-set/medqa_1273.csv")
for i, q in enumerate(df.to_dict("records")):
    if i <= 100: continue
    probs, time, confident, search_results, used_cot = eval_medqa(q)
    results.append({
        "id": q["id"],
        "source": q["source"],
        "probA": probs[0],
        "probB": probs[1],
        "probC": probs[2],
        "probD": probs[3],
        "probE": probs[4],
        "answer": q["answer"],
        "time": time,
        "error": not confident,
        "used_cot": used_cot,
        "sources": [r["source"] for r in search_results],
        "ids": [r["id"] for r in search_results],
        "similarity": [r["score"] for r in search_results]
    })
    if i % 100 == 0 and i != 0:
        write(results, "medqa_results.csv")
        results.clear()

if results:
    write(results, "medqa_results.csv")
    results.clear()

print(f"Total errors: {errors}")