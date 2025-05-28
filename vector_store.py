# vector_store.py

import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List

# Initialize the embedding model (CPU friendly)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_documents(docs: List[str]):
    """Convert a list of strings into dense vectors."""
    return embedding_model.encode(docs, show_progress_bar=True)

def create_faiss_index(embeddings):
    """Create a FAISS index from embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_faiss_index(index, path="vector_store/index.faiss", metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

    if metadata:
        with open(path + ".pkl", "wb") as f:
            pickle.dump(metadata, f)

def load_faiss_index(path="vector_store/index.faiss"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")
    index = faiss.read_index(path)

    metadata_path = path + ".pkl"
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = None

    return index, metadata

def search_index(index, query: str, documents: List[str], top_k=5):
    """Perform similarity search on the index using a query."""
    query_vec = embed_documents([query])
    distances, indices = index.search(query_vec, top_k)
    return [(documents[i], distances[0][j]) for j, i in enumerate(indices[0])]
