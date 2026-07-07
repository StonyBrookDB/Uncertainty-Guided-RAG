from pymilvus import MilvusClient, DataType
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

"""
Wikipedia (WIKI): Sample keys: ['id', 'title', 'content', 'contents', 'wiki_id']
Textbook (TEXT): Sample keys: ['id', 'title', 'content', 'contents']
PubMD (PBMD): ['id', 'title', 'content', 'contents', 'PMID']
"""

client = MilvusClient(uri="http://localhost:19530")

# Set up Milvus collection
schema = client.create_schema(
    partition_key_field = "source"
    )
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("source", DataType.VARCHAR, max_length=4)
schema.add_field("content", DataType.VARCHAR, max_length=1900) # Max size captures >96% of chunks for PBMD dataset
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=768)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector", 
    index_type="AUTOINDEX", # AUTOINDEX selects best index algorithm, can change later 
    metric_type="COSINE"
)

client.create_collection(
    collection_name="MedRAG_collection",
    schema=schema,
    index_params=index_params
)
print("Created collection MedRAG_collection")

# Transformation and data processing
embedding_model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True, device="auto")

def insert_batch(chunks, source):
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
    data = [{"source": source, "content": chunks[i], "vector": embeddings[i]} for i in range(len(chunks))]

    client.insert(
        collection_name="MedRAG_collection",
        data=data
    )
    print(f"Inserted {len(chunks)} chunks into MedRAG_collection from {source}")
    
data = load_dataset("MedRAG/textbooks", split="train", streaming=True)
print("Loaded MedRAG/textbooks")
chunks = []
for i, entry in enumerate(data):
    if len(entry["content"]) < 1900:
        chunks.append(entry["content"])
    if i % 3000 == 0 and i != 0:
        insert_batch(chunks, "TEXT")
        chunks = []
if chunks:
    insert_batch(chunks, "TEXT")


data = load_dataset("MedRAG/wikipedia", split="train", streaming=True)
print("Loaded MedRAG/wikipedia")
chunks = []
for i, entry in enumerate(data):
    if len(entry["content"]) < 1900:
        chunks.append(entry["content"])
    if i % 5000 == 0 and i != 0:
        insert_batch(chunks, "WIKI")
        chunks = []
if chunks:
    insert_batch(chunks, "WIKI")

data = load_dataset("MedRAG/pubmed", split="train", streaming=True)
print("Loaded MedRAG/pubmed")
chunks = []
for i, entry in enumerate(data):
    if len(entry["content"]) < 1900:
        chunks.append(entry["content"])
    if i % 5000 == 0 and i != 0:
        insert_batch(chunks, "PBMD")
        chunks = []
if chunks:
    insert_batch(chunks, "PBMD")