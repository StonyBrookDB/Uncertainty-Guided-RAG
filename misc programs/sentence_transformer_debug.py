from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Transformation and data processing
# model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
# model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True, device="cpu")
# print(model.encode(["Hello world"], normalize_embeddings=True).tolist())

# def insert_batch(chunks, source):
#     embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
#     print(embeddings)
    # data = [{"source": source, "content": chunks[i], "vector": embeddings[i]} for i in range(len(chunks))]

    # client.insert(
    #     collection_name="MedRAG_collection",
    #     data=data
    # )
    # print(f"Inserted {len(chunks)} chunks into MedRAG_collection from {source}")
    
# data = load_dataset("MedRAG/textbooks", split="train", streaming=True)
# print("Loaded MedRAG/textbooks")
# chunks = []
# for i, entry in enumerate(data):
    # insert_batch([entry["content"]], "TEXT")
#     if len(entry["content"]) < 1900:
#         chunks.append(entry["content"])
    # if i % 5000 == 0 and i != 0:
    #     insert_batch(chunks, "TEXT")
    #     chunks = []
# if chunks:
#     insert_batch(chunks, "TEXT")


# data = load_dataset("MedRAG/wikipedia", split="train", streaming=True)
# print("Loaded MedRAG/wikipedia")
# chunks = []
# for i, entry in enumerate(data):
#     if len(entry["content"]) < 1900:
#         chunks.append(entry["content"])
#     if i % 5000 == 0 and i != 0:
#         insert_batch(chunks, "WIKI")
#         chunks = []
# if chunks:
#     insert_batch(chunks, "WIKI")

# data = load_dataset("MedRAG/pubmed", split="train", streaming=True)
# print("Loaded MedRAG/pubmed")
# chunks = []
# for i, entry in enumerate(data):
#     if len(entry["content"]) < 1900:
#         chunks.append(entry["content"])
#     if i % 5000 == 0 and i != 0:
#         insert_batch(chunks, "PBMD")
#         chunks = []
# if chunks:
#     insert_batch(chunks, "PBMD")