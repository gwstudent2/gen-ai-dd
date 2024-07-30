import chromadb
from chromadb.utils import embedding_functions

# set some config variables for ChromaDB
CHROMA_DATA_PATH = "vdb_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

# create a new ChromaDB client
vdb_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# specify the embedding functions we'll use
vdb_embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name=EMBED_MODEL
)

# create a new collection or reuse existing one
vdb_collection = vdb_client.get_or_create_collection(
     name=COLLECTION_NAME,
     embedding_function=vdb_embedding_func,
     metadata={"hnsw:space": "cosine"},
)

# our virtual "documents" to draw on for context
datadocs = [
     "The latest iPhone model comes with impressive features and a powerful camera.",
     "Exploring the beautiful beaches and vibrant culture of Bali is a dream for many travelers.",
     "Einstein's theory of relativity revolutionized our understanding of space and time.",
     "Traditional Italian pizza is famous for its thin crust, fresh ingredients, and wood-fired ovens.",
     "The American Revolution had a profound impact on the birth of the United States as a nation.",
     "Regular exercise and a balanced diet are essential for maintaining good physical health.",
     "Leonardo da Vinci's Mona Lisa is considered one of the most iconic paintings in art history.",
     "Climate change poses a significant threat to the planet's ecosystems and biodiversity.",
     "Startup companies often face challenges in securing funding and scaling their operations.",
     "Beethoven's Symphony No. 9 is celebrated for its powerful choral finale, 'Ode to Joy.'",
]

# add some metadata about the categories of the documents
# provides additional info about the document
# can also query on this
categories = [
     "technology",
     "travel",
     "science",
     "food",
     "history",
     "fitness",
     "art",
     "climate change",
     "business",
     "music",
]

vdb_collection.add(
    documents=datadocs,
    ids=[f"id{i}" for i in range(len(datadocs))],
    metadatas=[{"category": c} for c in categories]
)


while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue
    query_results = vdb_collection.query(
        query_texts=[query],
        n_results=1,
    )
    
    print(query_results["documents"])
    print(query_results["distances"])
    print(query_results["metadatas"])