# Handles similarity search and retrieval

import numpy as np
from utils.embedder import load_model

def cosine_similarity(a, b):
    """Calculate cosine similarity."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def get_relevant_chunks(query, vector_store, top_k=5):
    """Retrieve most relevant chunks."""
    model = load_model()

    query_vec = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = [
        cosine_similarity(query_vec, emb)
        for emb in vector_store["embeddings"]
    ]

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [
        vector_store["chunks"][i]
        for i in top_indices
    ]