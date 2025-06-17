import faiss
import os
import json
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from app.services.faiss_service import FaissService
import numpy as np
import logging
from datetime import datetime
#from app.services.moderation_pipeline import load_policy_documents

faiss_service = FaissService()
#
INDEX_PATH = "app/vector_store/faiss_index.bin"
MAPPING_PATH = "app/vector_store/id_mapping.json"
EMBEDDING_DIM = 384

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_policy_documents(policy_dir="data/policy_docs/"):
    """
    Read all .txt policy files and return dict of filename -> list of text chunks (paragraphs).
    """
    documents = {}
    for fname in os.listdir(policy_dir):
        if fname.endswith(".txt"):
            full_path = os.path.join(policy_dir, fname)
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
                paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
                documents[fname] = paragraphs
    return documents


class EmbeddingService:
    """
    Provides methods for generating normalized embeddings and managing FAISS index.
    Uses cosine similarity via inner product on unit-normalized vectors.
    """

    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        index_exists = os.path.exists("app/vector_store/faiss_index.bin")
        mapping_exists = os.path.exists("app/vector_store/id_mapping.json")

        if index_exists and mapping_exists:
            self.index, self.metadata = faiss_service.load()
        else:
            print("[EmbeddingService] Missing FAISS index or metadata. Rebuilding from policy documents...")
            self.index = faiss.IndexFlatIP(384)
            self.metadata = []

            documents = load_policy_documents()
            for source_file, chunks in documents.items():
                self.add_to_index(chunks, source_file)

            print("[EmbeddingService] FAISS index rebuilt and saved.")


    def embed(self, texts):
        """
        Generate normalized embeddings for a list of texts.
        """
        raw_embeddings = self.model.encode(texts, convert_to_numpy=True)
        normalized_embeddings = normalize(raw_embeddings, axis=1)
        return normalized_embeddings

    
    def add_to_index(self, chunks, source_file):
        embeddings = self.embed(chunks)
        self.index.add(embeddings)
        start_id = self.index.ntotal
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "id": start_id + i,
                "text": chunk,
                "source_file": source_file,
                "created_at": datetime.utcnow().isoformat()
            })
        # Save updated FAISS index and metadata
        faiss_service.save(self.index, self.metadata)

    
    def search(self, query_text, top_k=3):
        query_embedding = self.embed([query_text])
        D, I = self.index.search(query_embedding.astype("float32"), top_k * 3)  # fetch more initially

        seen_texts = set()
        
        results = []

        for idx, score in zip(I[0], D[0]):
            entry = self.metadata[idx]
            text = entry["text"]
            source = entry["source_file"]

            # Avoid exact duplicates
            if text in seen_texts:
                continue

            

            results.append({
                "text": text,
                "source_file": source,
                "score": round(float(score), 4),
                "created_at": entry["created_at"]
            })
            seen_texts.add(text)
            #seen_sources.add(source)

            if len(results) >= top_k:
                break

        return results

def regenerate_faiss_index_if_missing(force: bool = False):
    if force:
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(MAPPING_PATH):
            os.remove(MAPPING_PATH)
        logger.info("[rebuild_index] Forced: Removed existing FAISS index and mapping.")

    if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
        logger.info("[rebuild_index] FAISS index or mapping missing. Regenerating...")

        documents = load_policy_documents()
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        metadata = []

        current_id = 0
        for source_file, chunks in documents.items():
            embeddings = normalize(model.encode(chunks, convert_to_numpy=True), axis=1)
            index.add(embeddings)

            for chunk in chunks:
                metadata.append({
                    "id": current_id,
                    "text": chunk,
                    "source_file": source_file,
                    "created_at": datetime.utcnow().isoformat()
                })
                current_id += 1

        FaissService().save(index, metadata)
        logger.info("[rebuild_index] Regeneration complete.")
    else:
        logger.info("[rebuild_index] Index and metadata already exist. Skipping regeneration.")
