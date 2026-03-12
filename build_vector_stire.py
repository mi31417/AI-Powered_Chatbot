# build_vector_store.py
"""
Run this to build the vector store from faq.json:
    python build_vector_store.py
Creates:
 - faq.index      (faiss index)
 - faq_store.json (answers/questions/links metadata)
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_JSON = os.path.join(BASE_DIR, "faq.json")
INDEX_PATH = os.path.join(BASE_DIR, "faq.index")
STORE_PATH = os.path.join(BASE_DIR, "faq_store.json")

def build_index():
    print("Loading FAQ JSON:", FAQ_JSON)
    with open(FAQ_JSON, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    questions = [item["question"] for item in faqs]
    print(f"Loaded {len(questions)} FAQ items.")

    print("Loading embedding model (MiniLM)...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Computing embeddings (this may take a few seconds)...")
    embeddings = embedder.encode(questions, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    # Build FAISS index (L2)
    index = faiss.IndexFlatL2(dim)
    index.add(np.asarray(embeddings))

    print("Saving FAISS index to", INDEX_PATH)
    faiss.write_index(index, INDEX_PATH)

    # Save metadata (we keep questions, answers, topics, links)
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(faqs, f, indent=2)

    print("Saved store to", STORE_PATH)
    print("Vector store build complete.")

if __name__ == "__main__":
    build_index()

