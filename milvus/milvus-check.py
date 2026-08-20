from pymilvus import MilvusClient
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import time as t

client = MilvusClient(uri="http://localhost:19530")
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", trust_remote_code=True, device="cuda")

print(client.list_collections())
for c in client.list_collections():
    print(f"Collection: {c}")
    print(client.get_collection_stats(c))
    print(client.get_load_state(c))
    print(client.describe_collection(c))
    print(client.list_indexes(c))
          
# print("Attempting to load textbook collection")
# client.load_collection("MedRAG_textbook_collection")
# print("Collection loaded")
# client.release_collection("MedRAG_textbook_collection")
# client.release_collection("MedRAG_statpearls_collection")
# client.release_collection("MedRAG_pubmed_collection")
# client.load_collection("MedRAG_textbook_collection")
# client.load_collection("MedRAG_statpearls_collection")
# client.load_collection("MedRAG_pubmed_collection")

# def get_context(prompt):

#     query_embedding = embedding_model.encode(prompt, normalize_embeddings=True).tolist()
#     search = []
#     for c in client.list_collections():
#         query = client.search(
#             collection_name=c,
#             data=[query_embedding],
#             limit=5,
#             output_fields=["id", "source", "content"],
#             search_params={
#                 "metric_type": "COSINE",
#                 "params": {}
#             }
#         )
#         search.extend(query[0])

#     results = []
#     for r in search:
#         results.append({
#             "score": r["distance"],
#             "id": r["id"],
#             "source": r["entity"].get("source"),
#             "content": r["entity"].get("content")
#         })

#     results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]
#     return "\n\n".join([f"{r['content']}" for r in results]), results

# start = t.time()
# context = get_context("How many chambers does the human heart have?")
# end = t.time()
# print(context)
# print("Query took", end - start, "seconds")