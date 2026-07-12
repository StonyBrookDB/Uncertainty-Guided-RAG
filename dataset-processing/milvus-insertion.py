from pymilvus import MilvusClient, DataType
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

"""
Wikipedia (WIKI): Sample keys: ['id', 'title', 'content', 'contents', 'wiki_id']
Textbook (TEXT): Sample keys: ['id', 'title', 'content', 'contents']
PubMD (PBMD): ['id', 'title', 'content', 'contents', 'PMID']
"""

client = MilvusClient(uri="http://localhost:19530")

# Set up Milvus collection
schema = client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("source", DataType.VARCHAR, max_length=4)
schema.add_field("content", DataType.VARCHAR, max_length=8192)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=768)


# client.create_collection(
#     collection_name="MedRAG_textbook_collection",
#     schema=schema,
# )
client.create_collection(
    collection_name="MedRAG_statpearls_collection",
    schema=schema,
)
# client.create_collection(
#     collection_name="MedRAG_pubmed_collection",
#     schema=schema,
# )

# Transformation and data processing
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", trust_remote_code=True, device="cuda")

def insert_batch(chunks, source, collection_name):
    embeddings = embedding_model.encode(chunks, normalize_embeddings=True).tolist()
    data = [{"source": source, "content": chunks[i], "vector": embeddings[i]} for i in range(len(chunks))]
    client.insert(
        collection_name=f"MedRAG_{collection_name}_collection",
        data=data
    )
    print(f"Inserted {len(data)} chunks into MedRAG_{collection_name}_collection from {source}")

# data = load_dataset("MedRAG/textbooks", split="train", streaming=True)
# print("Loaded MedRAG/textbooks")
# chunks = []
# for i, entry in enumerate(data):
#     if len(entry["content"].encode("utf-8")) <= 8192:
#         chunks.append(entry["content"])
#     if i % 5000 == 0 and i != 0:
#         insert_batch(chunks, "TEXT", "textbook")
#         chunks = []
# if chunks:
#     insert_batch(chunks, "TEXT", "textbook")

# folder = Path("dataset-processing/statpearls/chunk")
# records = []
# for file in folder.glob("*.jsonl"):
#     with file.open("r", encoding="utf-8") as f:
#         for line in f:
#             records.append(json.loads(line))

# chunks = []
# for i, record in enumerate(records):
#     if len(record["content"].encode("utf-8")) <= 8192:
#         chunks.append(record["content"])
#     if i % 5000 == 0 and i != 0:
#         insert_batch(chunks, "STAT", "statpearls")
#         chunks = []
#     if i % 100000 == 0 and i != 0:
#         print(f"Inserted {i} entries from STAT")

data = load_dataset("MedRAG/pubmed", split="train", streaming=True)
print("Loaded MedRAG/pubmed")
chunks = []
for i, entry in enumerate(data):
    if i < 135001: continue
    if i == 5e6: break
    if len(entry["content"].encode("utf-8")) <= 8192:
        chunks.append(entry["content"])
    if i % 5000 == 0 and i != 0:
        insert_batch(chunks, "PBMD", "pubmed")
        chunks = []
    if i % 100000 == 0 and i != 0:
        print(f"Inserted {i} entries from PBMD")
if chunks:
    insert_batch(chunks, "PBMD", "pubmed")

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector", 
    index_type="AUTOINDEX", # AUTOINDEX selects best index algorithm, can change later 
    metric_type="COSINE"
)

# client.create_index(
#     collection_name="MedRAG_textbook_collection",
#     index_params=index_params
# )

client.create_index(
    collection_name="MedRAG_statpearls_collection",
    index_params=index_params
)

client.create_index(
    collection_name="MedRAG_pubmed_collection",
    index_params=index_params
)