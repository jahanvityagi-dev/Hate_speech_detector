import faiss
import os
import pickle
import numpy as np
import json

INDEX_PATH = "app/vector_store/faiss_index.bin"
MAPPING_PATH = "app/vector_store/id_mapping.json"
EMBEDDING_DIM = 384  # or 768, depending on your model

class FaissService:
    def __init__(self, index_path="vector_store/index.faiss", metadata_path="vector_store/metadata.pkl", dim=384):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dim = dim
        

    def save(self, index, text_chunks, chunk_sources):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump((text_chunks, chunk_sources), f)
        print("[FAISS] Saved index and metadata")



    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            print("[FAISS] Index or metadata file not found. Returning empty index.")
            index = faiss.IndexFlatIP(self.dim)
            return index, [], []
        

        index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            text_chunks, chunk_sources = pickle.load(f)
        print(f"[FAISS] Loaded index with {len(text_chunks)} entries.")
        return index, text_chunks, chunk_sources
    
    def _load_index(self):
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        if os.path.exists(MAPPING_PATH):
            with open(MAPPING_PATH, 'r') as f:
                self.id_mapping = json.load(f)

    def add_embedding(self, embedding: np.ndarray, metadata: dict):
        embedding = embedding.astype("float32").reshape(1, -1)
        self.index.add(embedding)
        self.id_mapping.append(metadata)
        self._save()

    def search(self, query_embedding: np.ndarray, k: int = 5):
        query_embedding = query_embedding.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return [
            self.id_mapping[i] for i in indices[0] if i < len(self.id_mapping)
        ]

    def _save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(MAPPING_PATH, 'w') as f:
            json.dump(self.id_mapping, f)
