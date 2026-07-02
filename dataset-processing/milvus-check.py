from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

collections = client.list_collections()
print(f"Collections: {collections}")

stats = client.get_collection_stats(collection_name="MedRAG_collection")
print(f"Collection stats: {stats}")

print(f"Number of entities in MedRAG_collection: {stats['row_count']}")