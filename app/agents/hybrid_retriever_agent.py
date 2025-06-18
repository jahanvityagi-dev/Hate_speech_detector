import os, logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.services.embedding_service import EmbeddingService
from app import config
from app.agents.error_handler_agent import ErrorHandlerAgent

class HybridRetrieverAgent:
    def __init__(self):
        self.error_handler = ErrorHandlerAgent()
        # Load embedding model for vectorstore
        embedding_model = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        self.embedding_service = EmbeddingService()
        if not embedding_model.startswith("sentence-transformers/"):
            embedding_model = f"sentence-transformers/{embedding_model}"
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        # Load FAISS vector store from disk
        try:
            self.vectorstore = FAISS.load_local(
                config.VECTOR_STORE_DIR, 
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"[HybridRetrieverAgent] Failed to load FAISS index from `{config.VECTOR_STORE_DIR}`: {e}"
            )
            # Raise or handle as needed (stop initialization if index missing)
            raise
        # Log successful load
        doc_count = getattr(self.vectorstore.index, "ntotal", 0)
        logging.getLogger(__name__).info(f"[HybridRetrieverAgent] Loaded FAISS index with {doc_count} documents.")

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve relevant policy text snippets for the given input query."""
        try:
            # Use the centralized EmbeddingService search
            snippets = self.embedding_service.search(query, top_k=3)
            return snippets
        except Exception as e:
            return self.error_handler.handle_error("HybridRetrieverAgent", str(e))

