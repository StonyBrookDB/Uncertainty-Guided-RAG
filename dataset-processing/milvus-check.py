from pymilvus import MilvusClient
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

client = MilvusClient(uri="http://localhost:19530")
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", trust_remote_code=True, device="cuda")

COLLECTIONS = ["MedRAG_textbook_collection", "MedRAG_statpearls_collection", "MedRAG_pubmed_collection"]
for c in COLLECTIONS:
    stats = client.get_collection_stats(collection_name=c)
    print(f"Collection stats for {c}: {stats}")

# def get_context(prompt):
#     query_embedding = embedding_model.encode(
#         prompt,
#         normalize_embeddings=True
#     ).tolist()

#     search_results = client.search(
#         collection_name="MedRAG_statpearls_collection",
#         data=[query_embedding],
#         limit=5,
#         output_fields=["id", "source", "content"],
#         search_params={
#             "metric_type": "COSINE",
#             "params": {}
#         }
#     )

#     results = []

#     for r in search_results[0]:
#         results.append({
#             "score": r["distance"],
#             "id": r["id"],
#             "source": r["entity"].get("source"),
#             "content": r["entity"].get("content")
#         })

#     return ["\n\n".join([r["content"] for r in results]),results]

# print(get_context("Name the four chambers of the human heart")[0])
