"""
llama-3.1-8B-Instruct
Average time per question: 8.590002164840698 seconds
Max time for a question: 12.166870355606079 seconds
Min time for a question: 6.126097202301025 seconds
llama-3.2-3B-Instruct
Average time per question: 6.413669996261596 seconds
Max time for a question: 14.53083848953247 seconds
Min time for a question: 4.051152229309082 seconds
Qwen3-4B-Instruct
Average time per question: 6.17600314617157 seconds
Max time for a question: 8.69042181968689 seconds
Min time for a question: 4.083135604858398 seconds
Phi4-mini-instruct
Average time per question: 4.213488435745239 seconds
Max time for a question: 6.049081087112427 seconds
Min time for a question: 2.957401990890503 seconds
gemma3-4b-it
Average time per question: 4.577749080657959 seconds
Max time for a question: 6.853046655654907 seconds
Min time for a question: 2.401185989379883 seconds
"""


# region Imports
from pymilvus import MilvusClient
import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import time as t
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from transformers import BitsAndBytesConfig
# endregion

# region Milvus Connect
# client = MilvusClient(uri="http://localhost:19530")
# client.load_collection("MedRAG_textbook_collection")
# client.load_collection("MedRAG_statpearls_collection")
# client.load_collection("MedRAG_pubmed_collection")
# endregion

# region Models
bnb_config = bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model_name = "meta-llama/Llama-3.1-8b-instruct"
reranker = reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", quantization_config=bnb_config, torch_dtype=torch.float16)
model.eval()
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", trust_remote_code=True, device="cuda")
# endregion

# region Constants
option_tokens = {"A": tokenizer.encode(" A", add_special_tokens=False)[0],
                 "B": tokenizer.encode(" B", add_special_tokens=False)[0],
                 "C": tokenizer.encode(" C", add_special_tokens=False)[0],
                 "D": tokenizer.encode(" D", add_special_tokens=False)[0]}
COLLECTIONS = ["MedRAG_textbook_collection", "MedRAG_statpearls_collection", "MedRAG_pubmed_collection"]
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
    for opt in confident_options:
    
        messages = [
            {"role": "system", "content": "You are a medical expert and a search engine query generator. You will be given a medical question and multiple choice options. Your task is to generate a specific and descriptive search query that would retrieve medical evidence strongly supporting the choice provided. The search query should be focused on the medical evidence related to the choice. Do not include any other information or context in your response. Do not use negation words like \"excluding\" or \"not\"."},
            {"role" : "user", "content": f"""Example:
Medical Question: What is the most effective initial pharmacological therapy for stable angina?
Options: A) Nitroglycerin B) Beta-blockers C) Calcium channel blockers D) Aspirin
Generate one specific search query that would retrieve medical evidence strongly supporting the choice: \"Beta-blockers\"
Search Query: Clinical evidence on why beta-blockers are recommended as the initial treatment for stable angina, including guideline recommendations on beta-blockers, randomized trials, and meta-analyses comparing beta-blockers with other antianginal medications.
Now generate:
Medical Question: {question}
Options: A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}
Generate a specific search query that would retrieve medical evidence strongly supporting the choice: "{opt}"
Search Query: """}
        ]

        inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512)
        query = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Search Query:")[-1].strip()
        queries.append(query)
    return queries

results = []
times = []
df = pd.read_csv(f"data-splits/medmcqa_1.csv")
for n, q in enumerate(df.to_dict("records")):
    if n != 0 and n % 10 == 0:
        print(f"Processed {n} questions")
    if n == 50: break
    print(f"""Question: {q["question"]}
A) {q["opa"]}
B) {q["opb"]}
C) {q["opc"]}
D) {q["opd"]}""")
    start = t.time()
    queries = get_query(q["question"], [q["opa"], q["opb"], q["opc"], q["opd"]], [q["opa"], q["opb"], q["opc"], q["opd"]])
    end = t.time()
    times.append(end - start)
    for query in queries:
        print(query)

print(f"Average time per question: {sum(times)/len(times)} seconds")
print(f"Max time for a question: {max(times)} seconds")
print(f"Min time for a question: {min(times)} seconds")