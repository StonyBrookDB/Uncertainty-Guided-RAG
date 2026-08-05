"""
2.2: 
1.1: 2
1.2: 0
RR: 0
NPB: 0

prompt.embedding
v1 - instructions
v2 - no instructions
v3 - structured
NPB - no pubmed
RR - reranker, takes top 10 from each and choses top 5
"""

# region Imports
from pyexpat.errors import messages

from pymilvus import MilvusClient
import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import time as t
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
# endregion

# region Milvus Connect
client = MilvusClient(uri="http://localhost:19530")
client.load_collection("MedRAG_textbook_collection")
client.load_collection("MedRAG_statpearls_collection")
client.load_collection("MedRAG_pubmed_collection")
# endregion

# region Models
model_name = "meta-llama/Llama-3.1-8B-Instruct"
reranker = reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
model.eval()
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", trust_remote_code=True, device="cuda")
# endregion

# region Constants
option_tokens = {"A": tokenizer.encode("A", add_special_tokens=False)[0],
                 "B": tokenizer.encode("B", add_special_tokens=False)[0],
                 "C": tokenizer.encode("C", add_special_tokens=False)[0],
                 "D": tokenizer.encode("D", add_special_tokens=False)[0]}
COLLECTIONS = ["MedRAG_textbook_collection", "MedRAG_statpearls_collection", "MedRAG_pubmed_collection"]
INSTRUCTIONS = """You are a helpful medical expert, and your task is to answer a multi-choice medical question. 
The question is provided below, along with four answer options labeled A, B, C, and D. 
Your goal is to select the most appropriate answer based on your medical knowledge and reasoning, as well as any additional context provided. 
Output your answer as a single letter first. Then on a new line, briefly explain why the context led you to believe that answer is correct. Include which paragraph number was most helpful."""
error = 0
# endregion

# Reranker
# def get_context(prompt):
    # query_embedding = embedding_model.encode(prompt, normalize_embeddings=True).tolist()
    # search = []
    # for c in COLLECTIONS:
    #     query = client.search(
    #         collection_name=c,
    #         data=[query_embedding],
    #         limit=10,
    #         output_fields=["id", "source", "content"],
    #         search_params={
    #             "metric_type": "COSINE",
    #             "params": {}
    #         }
    #     )
    #     search.extend(query[0])

    # results = []
    # for r in search:
    #     results.append({
    #         "score": r["distance"],
    #         "id": r["id"],
    #         "source": r["entity"].get("source"),
    #         "content": r["entity"].get("content")
    #     })

    # pairs = [(prompt, r["content"]) for r in results]
    # rerank_scores = reranker.predict(pairs, show_progress_bar=False, batch_size=len(COLLECTIONS) * 10)
    # for r, score in zip(results, rerank_scores):
    #     r["rerank_score"] = score

    # results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:5]
    # return "\n\n".join([f"{r['content']}" for r in results]), results

# No reranker
def get_context(prompt):
    query_embedding = embedding_model.encode(prompt, normalize_embeddings=True).tolist()
    search = []
    for c in COLLECTIONS:
        query = client.search(
            collection_name=c,
            data=[query_embedding],
            limit=5,
            output_fields=["id", "source", "content"],
            search_params={
                "metric_type": "COSINE",
                "params": {}
            }
        )
        search.extend(query[0])

    results = []
    for r in search:
        results.append({
            "score": r["distance"],
            "id": r["id"],
            "source": r["entity"].get("source"),
            "content": r["entity"].get("content")
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]
    return "\n\n".join([f"{r['content']}" for r in results]), results

# Keys: ["id", "question", "opa", "opb", "opc", "opd", "cop", "choice_type", "exp", "subject_name", "topic_name"]
def eval_medmcqa(q): # [probA, probB, probC, probD, time, search_results]
    start = t.time()
    context, search_results = get_context(f"""Question: {q["question"]}
A) {q["opa"]}
B) {q["opb"]}
C) {q["opc"]}
D) {q["opd"]}""")

    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": f"""Context: 
{context}
Question: {q["question"]}
    A) {q["opa"]}
    B) {q["opb"]}
    C) {q["opc"]}
    D) {q["opd"]}
Answer only with the letter of the correct option. Answer: """}
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt = True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    end = t.time()
        
    last_token_logits = outputs.logits[0, -1, :]

    option_logits = torch.tensor([last_token_logits[option_tokens["A"]],
                                  last_token_logits[option_tokens["B"]],
                                  last_token_logits[option_tokens["C"]],
                                  last_token_logits[option_tokens["D"]]])

    # Safety Check: 
    next_token_id = torch.argmax(last_token_logits).item()
    next_token_text = tokenizer.decode(next_token_id)
    if next_token_id not in option_tokens.values():
        print(f"Error, next token is not an option: {next_token_text}")
        global error
        error += 1

        # Print model response
        with torch.no_grad():
            generated = model.generate(**inputs,max_new_tokens=100)
        print(tokenizer.decode(generated[0], skip_special_tokens=True))

    return torch.softmax(option_logits, dim=0).tolist() + [end - start] + [search_results]

# Keys: ["centerpiece", "options", "correct_options", "correct_options_idx", "correct_options_literal", "subject", "id"]
def eval_mmlu(q): # [probA, probB, probC, probD, time, search_results]
    options = ast.literal_eval(q["options"])
    start = t.time()
    context, search_results = get_context(f"""Question: {q["centerpiece"]}
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}""")
                                        
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": f"""Context: 
{context}
Question: {q["centerpiece"]}
    A) {options[0]}
    B) {options[1]}
    C) {options[2]}
    D) {options[3]}
Answer only with the letter of the correct option. Answer: """}
    ]
    
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt = True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    end = t.time()
        
    last_token_logits = outputs.logits[0, -1, :]

    option_logits = torch.tensor([last_token_logits[option_tokens["A"]],
                                  last_token_logits[option_tokens["B"]],
                                  last_token_logits[option_tokens["C"]],
                                  last_token_logits[option_tokens["D"]]])

    # Safety Check: 
    next_token_id = torch.argmax(last_token_logits).item()
    next_token_text = tokenizer.decode(next_token_id)
    if next_token_id not in option_tokens.values():
        print(f"Error, next token is not an option: {next_token_text}")
        global error
        error += 1
                                  
    return torch.softmax(option_logits, dim=0).tolist() + [end - start] + [search_results]

# Warmup
print("Warming up model")
warmup_prompt = """Question: What is the capital of France?
    A) London
    B) Paris
    C) Berlin
    D) Madrid
Provide the answer as a single letter (A, B, C, or D).
"""

warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    for _ in range(5):
        _ = model(**warmup_inputs, return_dict=True)
torch.cuda.synchronize()

# MEDMCQA
print("Evaluating medmcqa splits")
for i in range(1, 6):
    results = []
    df = pd.read_csv(f"data-splits/medmcqa_{i}.csv")
    for n, q in enumerate(df.to_dict("records")):
        if (n + 1) % 100 == 0:
            print(f"Evaluated {n + 1} questions")
        prob = eval_medmcqa(q)
        results.append({
            "id" : q["id"],
            "subject" : q["subject_name"],
            "probA" : prob[0],
            "probB" : prob[1],
            "probC" : prob[2],
            "probD" : prob[3],
            "result" : q["cop"] == max(range(4), key=lambda x: prob[x]),
            "answer" : ["A", "B", "C", "D"][q["cop"]],
            "time" : prob[4],
            "split" : f"medmcqa_{i}",
            "sources" : [r["source"] for r in prob[5]],
            "ids" : [r["id"] for r in prob[5]],
            "similarity" : [r["score"] for r in prob[5]]
        })
    df_split = pd.DataFrame(results, columns=["id", "subject", "probA", "probB", "probC", "probD", "result", "answer", "time", "split", "sources", "ids", "similarity"])
    df_split.to_csv(f"medmcqa_results_{i}.csv", index=False)

# MMLU
print("Evaluating mmlu splits")
for i in range(1, 6):
    results = []
    df = pd.read_csv(f"data-splits/mmlu_{i}.csv")
    for n, q in enumerate(df.to_dict("records")):
        if (n + 1) % 100 == 0:
            print(f"Evaluated {n + 1} questions")
        prob = eval_mmlu(q)
        results.append({
            "id" : q["id"],
            "subject" : q["subject"],
            "probA" : prob[0],
            "probB" : prob[1],
            "probC" : prob[2],
            "probD" : prob[3],
            "result" : int(q["correct_options_idx"][1]) == max(range(4), key=lambda x: prob[x]),
            "answer" : ["A", "B", "C", "D"][int(ast.literal_eval(q["correct_options_idx"])[0])],
            "time" : prob[4],
            "split" : f"mmlu_{i}",
            "sources" : [r["source"] for r in prob[5]],
            "ids" : [r["id"] for r in prob[5]],
            "similarity" : [r["score"] for r in prob[5]]
        })
    df_split = pd.DataFrame(results, columns=["id", "subject", "probA", "probB", "probC", "probD", "result", "answer", "time", "split", "sources", "ids", "similarity"])
    df_split.to_csv(f"mmlu_results_{i}.csv", index=False)
print(f"Total errors: {error}")