"""
Benchmark reference: https://openreview.net/pdf?id=ri3Si3GBOm

22 errors, not enough tokens for CoT
"""

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
model = torch.compile(model, mode="reduce-overhead")
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

MAX_NEW_TOKENS = 512

errors = 0
USE_RERANKER = True
# endregion


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


def _run_model_cot(messages, max_new_tokens=MAX_NEW_TOKENS):
    """Runs full generation (CoT) and returns the generated token ids/text,
    per-step logits (scores), and elapsed time."""
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")
    prompt_len = inputs["input_ids"].shape[-1]

    torch.cuda.synchronize()
    start = t.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    end = t.time()

    generated_ids = outputs.sequences[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    scores = outputs.scores  # tuple of length num_generated_tokens, each [1, vocab]

    return generated_ids, generated_text, scores, end - start


def _extract_answer_letter_and_logits(generated_ids, scores):
    global errors

    for i in range(len(generated_ids) - 1, -1, -1):
        tok_text = tokenizer.decode([generated_ids[i]], skip_special_tokens=True).strip()
        if tok_text in option_tokens:
            return tok_text, scores[i][0]

    errors += 1
    return None, None


def _probs_from_logits(logits, valid_letters):
    option_logits = [logits[option_tokens[letter]].item() for letter in valid_letters]
    return torch.softmax(torch.tensor(option_logits), dim=0).tolist()


# returns [probs, time, confident flag, search_results]
def eval_medmcqa(q):
    letters = ["A", "B", "C", "D"]
    question_block = f"""{q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}"""

    context, search_results, elapsed_retrieval = get_context(f"Question: {question_block}", use_reranker=USE_RERANKER)

    messages = [
        {
            "role": "system",
            "content": (
                "You are answering a multiple-choice medical question using the provided "
                "context. Think through the question step by step, briefly explaining your "
                "reasoning using the context and your own medical knowledge. After your "
                "reasoning, on a new final line, give your answer in EXACTLY this format: "
                "'Answer: <letter>' where <letter> is one of A, B, C, or D. Do not include "
                "anything after that line."
            ),
        },
        {
            "role": "user",
            "content": f"""Context:
{context}
Question: {question_block}

Think step by step, then finish with 'Answer: <letter>'.""",
        },
    ]

    generated_ids, generated_text, scores, elapsed_model = _run_model_cot(messages)
    answer_letter, answer_logits = _extract_answer_letter_and_logits(generated_ids, scores)

    confident = answer_letter is not None
    if confident:
        probs = _probs_from_logits(answer_logits, letters)
    else:
        print(f"Warning: could not parse an answer letter from generated text:\n{generated_text}")
        probs = [1.0 / len(letters)] * len(letters)

    return [probs, elapsed_retrieval + elapsed_model, confident, search_results]


# returns [probs, time, confident flag, search_results]
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
        {
            "role": "system",
            "content": (
                "You are answering a multiple-choice medical question using the provided "
                "context. Think through the question step by step, briefly explaining your "
                "reasoning using the context and your own medical knowledge. After your "
                "reasoning, on a new final line, give your answer in EXACTLY this format: "
                "'Answer: <letter>' where <letter> is one of A, B, C, D, or E. Do not include "
                "anything after that line."
            ),
        },
        {
            "role": "user",
            "content": f"""Context:
{context}
Question: {question_block}

Think step by step, then finish with 'Answer: <letter>'.""",
        },
    ]

    generated_ids, generated_text, scores, elapsed_model = _run_model_cot(messages)
    answer_letter, answer_logits = _extract_answer_letter_and_logits(generated_ids, scores)

    confident = answer_letter is not None
    if confident:
        probs = _probs_from_logits(answer_logits, letters)
    else:
        print(f"Warning: could not parse an answer letter from generated text:\n{generated_text}")
        probs = [1.0 / len(letters)] * len(letters)

    return [probs, elapsed_retrieval + elapsed_model, confident, search_results]


def write(res, OUT_FILE):
    df = pd.DataFrame(res)
    file_exists = os.path.exists(OUT_FILE)
    df.to_csv(OUT_FILE, mode="a", header=not file_exists, index=False)


# Warmup
print("Warming up model")
warmup_messages = [
    {
        "role": "system",
        "content": "You are answering a multiple-choice question. Think step by step, then finish with 'Answer: <letter>'.",
    },
    {
        "role": "user",
        "content": """Question: What is the capital of France?
    A) London
    B) Paris
    C) Berlin
    D) Madrid

Think step by step, then finish with 'Answer: <letter>'.""",
    },
]
for _ in range(5):
    _run_model_cot(warmup_messages, max_new_tokens=32)
torch.cuda.synchronize()

# MEDMCQA
print("Evaluating MedMCQA test")
results = []
df = pd.read_csv("question-set/medmcqa_1000.csv")
for i, q in enumerate(df.to_dict("records")):
    probs, time_taken, confident, search_results = eval_medmcqa(q)
    results.append({
        "id": q["id"],
        "source": q["source"],
        "probA": probs[0],
        "probB": probs[1],
        "probC": probs[2],
        "probD": probs[3],
        "answer": q["answer"],
        "time": time_taken,
        "error": not confident,
        "sources": [r["source"] for r in search_results],
        "ids": [r["id"] for r in search_results],
        "similarity": [r["score"] for r in search_results]
    })
    if i % 100 == 0 and i != 0:
        write(results, "medmcqa_results.csv")
        results.clear()

if results:
    write(results, "medmcqa_results.csv")
    results.clear()

# MEDQA
print("Evaluating MedQA test")
results = []
df = pd.read_csv("question-set/medqa_1273.csv")
for i, q in enumerate(df.to_dict("records")):
    probs, time_taken, confident, search_results = eval_medqa(q)
    results.append({
        "id": q["id"],
        "source": q["source"],
        "probA": probs[0],
        "probB": probs[1],
        "probC": probs[2],
        "probD": probs[3],
        "probE": probs[4],
        "answer": q["answer"],
        "time": time_taken,
        "error": not confident,
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