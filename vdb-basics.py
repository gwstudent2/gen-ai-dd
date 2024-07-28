from pymilvus import Milvus, DataType, CollectionSchema, FieldSchema, Collection, connections
import numpy as np
 
# Initialize Milvus client
connections.connect(host='localhost', port='19530')
 
# Define a collection schema: 
# Each vector has 8 dimensions, and the collection will be called 'example_collection'
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="Test Collection")
 
# Create a collection
collection = Collection(name="example_collection", schema=schema)
 
# Generate some random vectors for inserting into the database
num_vectors = 10  # We'll create 10 random vectors
np.random.seed(1)  # Seed for reproducibility
vectors = np.random.random((num_vectors, 8)).astype('float32').tolist()  # 8-dimensional vectors
 
# Insert vectors into the collection
ids = collection.insert([vectors])
 
# Create an index for faster search
# Here we use FLAT index, which is simple but may not be the fastest for large datasets
collection.create_index(field_name="example_field", index_params={"index_type": "FLAT", "metric_type": "L2"})
 
# Load the collection into memory for searching
collection.load()
 
# Querying the database: Search for vectors similar to a given query vector
query_vector = np.random.random((1, 8)).astype('float32').tolist()  # Generating a random query vector
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(data=query_vector, anns_field="example_field", param=search_params, limit=3)
 
# Output the search results
print("Search results:")
print(results)
 
# Clean up: Drop the collection when done
collection.drop()
