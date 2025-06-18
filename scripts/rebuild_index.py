# scripts/rebuild_index.py
from app.services.embedding_service import EmbeddingService

if __name__ == "__main__":
    print("Rebuilding FAISS index...")
    EmbeddingService().rebuild_faiss_index()
    print("✅ FAISS index rebuilt successfully!")
