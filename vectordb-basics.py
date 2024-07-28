from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
from pymilvus import connections
 
# Connect to Milvus
connections.connect(host='localhost', port='19530')
 
# Define schema for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="Test collection")
 
# Create a collection
collection = Collection(name="text_collection", schema=schema)
 
# Load a pre-trained model from sentence-transformers
model = SentenceTransformer('all-MiniLM-L6-v2')
 
# Define some example texts
texts = ["Hello world", "Hi there", "Greetings, friend", "Hello there", "Goodbye"]
 
# Convert texts to embeddings
embeddings = model.encode(texts)
 
# Insert data into Milvus
mr = collection.insert([embeddings])
 
# Wait for insert to complete
collection.load()
 
# Perform a search for similar texts
query_embedding = model.encode(["Hello, how are you?"])
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=3)
 
# Print out the most similar texts
print("Query: 'Hello, how are you?'")
for result in results[0]:
    print(f"Found: {texts[result.id]}, distance: {result.distance}")
 
# Disconnect from server
connections.disconnect()
