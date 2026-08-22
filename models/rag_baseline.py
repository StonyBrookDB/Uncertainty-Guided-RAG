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


# returns [[probA, probB, probC, probD], time, confident flag, search_results]
def eval_medmcqa(q):
    context, search_results, elapsed_retrieval = get_context(f"""Question: {q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}""", use_reranker=USE_RERANKER)

    messages = [
        {"role": "system", "content": "Use provided context and reasoning to guide your answer. Output your answer as a single letter (A, B, C, D). No explanation"},
        {"role": "user", "content": f"""Context: 
{context}
Question: {q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}
Answer: """}
    ]

    inputs, last_token_logits, elapsed_model = _run_model(messages)
    confident = _check_confident(inputs, last_token_logits)

    option_logits = [
        last_token_logits[option_tokens["A"]].item(),
        last_token_logits[option_tokens["B"]].item(),
        last_token_logits[option_tokens["C"]].item(),
        last_token_logits[option_tokens["D"]].item(),
    ]

    probs = torch.softmax(torch.tensor(option_logits), dim=0).tolist()
    return [probs, elapsed_retrieval + elapsed_model, confident, search_results]


# returns [[probA, probB, probC, probD, probE], time, confident flag, search_results]
def eval_medqa(q):
    start = t.time()
    context, search_results, elapsed_retrieval = get_context(f"""Question: {q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}
    E) {q["e"]}""", use_reranker=USE_RERANKER)
    end = t.time()

    messages = [
        {"role": "system", "content": "Use provided context and reasoning to guide your answer. Output your answer as a single letter (A, B, C, D, E). No explanation"},
        {"role": "user", "content": f"""Context: 
{context}
Question: {q["question"]}
    A) {q["a"]}
    B) {q["b"]}
    C) {q["c"]}
    D) {q["d"]}
    E) {q["e"]}
Answer: """}
    ]

    inputs, last_token_logits, elapsed_model = _run_model(messages)
    confident = _check_confident(inputs, last_token_logits)

    option_logits = [
        last_token_logits[option_tokens["A"]].item(),
        last_token_logits[option_tokens["B"]].item(),
        last_token_logits[option_tokens["C"]].item(),
        last_token_logits[option_tokens["D"]].item(),
        last_token_logits[option_tokens["E"]].item(),
    ]

    probs = torch.softmax(torch.tensor(option_logits), dim=0).tolist()
    return [probs, elapsed_retrieval + elapsed_model, confident, search_results]

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

# # MEDMCQA Test
# print("Evaluating MedMCQA test")
# results = []
# df = pd.read_csv("question-set/medmcqa_20.csv")
# for i, q in enumerate(df.to_dict("records")):
#     probs, time, confident, search_results = eval_medmcqa(q)
#     results.append({
#         "id": q["id"],
#         "source": q["source"],
#         "probA": probs[0],
#         "probB": probs[1],
#         "probC": probs[2],
#         "probD": probs[3],
#         "answer": q["answer"],
#         "time": time,
#         "error": not confident,
#         "sources": [r["source"] for r in search_results],
#         "ids": [r["id"] for r in search_results],
#         "similarity": [r["score"] for r in search_results]
#     })
#     print(results[-1])
#     if i % 100 == 0 and i != 0:
#         write(results, "medmcqa_results.csv")
#         results.clear()
#
# if results:
#     write(results, "medmcqa_results.csv")
#     results.clear()

# MEDMCQA
print("Evaluating MedMCQA test")
results = []
df = pd.read_csv("question-set/medmcqa_1000.csv")
for i, q in enumerate(df.to_dict("records")):
    probs, time, confident, search_results = eval_medmcqa(q)
    results.append({
        "id": q["id"],
        "source": q["source"],
        "probA": probs[0],
        "probB": probs[1],
        "probC": probs[2],
        "probD": probs[3],
        "answer": q["answer"],
        "time": time,
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
    if i <= 100: continue
    probs, time, confident, search_results = eval_medqa(q)
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