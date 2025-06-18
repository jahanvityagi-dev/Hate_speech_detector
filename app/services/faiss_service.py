import faiss
import os
import pickle
import numpy as np
import json

INDEX_PATH = "app/vector_store/faiss_index.bin"
MAPPING_PATH = "app/vector_store/id_mapping.json"
EMBEDDING_DIM = 384 

class FaissService:
    def __init__(self, index_path="app/vector_store/faiss_index.bin", metadata_path="app/vector_store/id_mapping.json", dim=384):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dim = dim

    def save(self, index, metadata):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        print("[FAISS] Saved index and metadata")

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            print("[FAISS] Index or metadata file not found. Returning empty index.")
            index = faiss.IndexFlatIP(self.dim)
            return index, []
        index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"[FAISS] Loaded index with {len(metadata)} entries.")
        return index, metadata
    
    def _load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.id_mapping = json.load(f)
        else:
            self.id_mapping = []

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
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.id_mapping, f)            
            