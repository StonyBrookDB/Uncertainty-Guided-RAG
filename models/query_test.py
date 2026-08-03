# region Imports
from pymilvus import MilvusClient
import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import time as t
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
# endregion

# region Models
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
model.eval()
# endregion

# region Constants
option_tokens = {"A": tokenizer.encode(" A", add_special_tokens=False)[0],
                 "B": tokenizer.encode(" B", add_special_tokens=False)[0],
                 "C": tokenizer.encode(" C", add_special_tokens=False)[0],
                 "D": tokenizer.encode(" D", add_special_tokens=False)[0]}
COLLECTIONS = ["MedRAG_textbook_collection", "MedRAG_statpearls_collection", "MedRAG_pubmed_collection"]
one_shot =  """Medical Question: What is the most effective initial pharmacological therapy for stable angina?
Options: A) Nitroglycerin B) Beta-blockers C) Calcium channel blockers D) Aspirin
Generate a specific search query that would retrieve medical evidence strongly supporting the choice: "Beta-blockers"
Search Query: Clinical trial evidence and guideline recommendations for beta-blockers as first-line therapy in stable angina"""
# endregion

def get_context(prompt):
    query_embedding = embedding_model.encode(prompt, normalize_embeddings=True).tolist()
    search = []
    for c in COLLECTIONS:
        query = client.search(
            collection_name=c,
            data=[query_embedding],
            limit=10,
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

    pairs = [(prompt, r["content"]) for r in results]
    rerank_scores = reranker.predict(pairs, show_progress_bar=False, batch_size=len(COLLECTIONS) * 10)
    for r, score in zip(results, rerank_scores):
        r["rerank_score"] = score

    results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:5]
    return "\n\n".join([f"{r['content']}" for r in results]), results

def get_query(question, options, confident_options):
    queries = []
    for opt in confident_options:\
    
        # messages = [{
        #     "role": "system", "content": "You are a medical expert and a search engine query generator. You will be given a medical question and multiple choice options. Your task is to generate a specific search query that would retrieve medical evidence strongly supporting the choice provided. The search query should be focused on the medical evidence related to the choice. Do not include any other information or context in your response."
        #     "role": "user", "content": f""
        # }]

        
        prompt = f"""Generate search queries for the following medical questions and options. Queries should be not be multiple choice.
{one_shot}

Medical Question: {question}
Options: A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}
Generate a specific search query that would retrieve medical evidence strongly supporting the choice: "{opt}"
Search Query:"""
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3, do_sample=True)
        query = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Search Query:")[-1].strip()
        queries.append(query)
    return queries

for i in range(1, 6):
    results = []
    df = pd.read_csv(f"data-splits/medmcqa_{i}.csv")
    for n, q in enumerate(df.to_dict("records")):
        print(f"""Question: {q["question"]}
    A) {q["opa"]}
    B) {q["opb"]}
    C) {q["opc"]}
    D) {q["opd"]}""")
        for query in get_query(q["question"], [q["opa"], q["opb"], q["opc"], q["opd"]], [q["opa"], q["opb"], q["opc"], q["opd"]]):
            print(query)
        print("------------------------------")
            