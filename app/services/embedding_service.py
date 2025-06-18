import os, json, logging
from datetime import datetime
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths and settings from config
POLICY_DOCS_DIR = config.DATA_DIR
VECTOR_STORE_PATH = Path(config.VECTOR_STORE_DIR)
CHUNK_SIZE = config.CHUNK_SIZE

def load_policy_documents(policy_dir: str = POLICY_DOCS_DIR) -> dict:
    """Read all .txt policy files and return dict of filename -> list of text chunks (paragraphs)."""
    documents = {}
    for fname in os.listdir(policy_dir):
        if fname.endswith(".txt"):
            full_path = os.path.join(policy_dir, fname)
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
            # Split by blank lines into paragraphs
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            documents[fname] = paragraphs
    return documents

class EmbeddingService:
    """
    Service for generating normalized embeddings and managing the FAISS vectorstore.
    It can rebuild the vector index from policy documents and perform searches.
    """
    def __init__(self, embedding_model: str = None):
        # Determine embedding model name from .env or default
        model_name = embedding_model or config.EMBED_MODEL
        # Ensure model name has correct HuggingFace path prefix
        if not model_name.startswith("sentence-transformers/"):
            model_name = f"sentence-transformers/{model_name}"
        # Load embedding model
        self.model = SentenceTransformer(model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate unit-normalized embeddings for a list of texts."""
        raw_embeddings = self.model.encode(texts, convert_to_numpy=True)
        normalized_embeddings = normalize(raw_embeddings, axis=1)
        return normalized_embeddings

    def rebuild_faiss_index(self) -> None:
        """Build FAISS index from policy documents and save to disk."""
        docs_by_file = load_policy_documents()
        langchain_docs = []
        for source_file, paragraphs in docs_by_file.items():
            for para in paragraphs:
                # Further split paragraphs into chunks of CHUNK_SIZE words if needed
                words = para.split()
                if len(words) > CHUNK_SIZE:
                    # Break long paragraph into fixed-size word chunks
                    for i in range(0, len(words), CHUNK_SIZE):
                        chunk = " ".join(words[i:i+CHUNK_SIZE])
                        langchain_docs.append(Document(
                            page_content=chunk,
                            metadata={
                                "source_file": source_file,
                                "created_at": datetime.utcnow().isoformat()
                            }
                        ))
                else:
                    langchain_docs.append(Document(
                        page_content=para,
                        metadata={
                            "source_file": source_file,
                            "created_at": datetime.utcnow().isoformat()
                        }
                    ))
        logger.info(f"[EmbeddingService] Loaded {len(langchain_docs)} text chunks from policy documents.")

        # Build FAISS index from documents using LangChain
        vectorstore = FAISS.from_documents(langchain_docs, embedding=self.embeddings)
        VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(VECTOR_STORE_PATH))
        logger.info(f"[EmbeddingService] FAISS index rebuilt and saved to {VECTOR_STORE_PATH}")

    def search(self, query_text: str, top_k: int = 3) -> list[dict]:
        """Retrieve top-k similar policy snippets for the query text."""
        # Ensure index files exist before loading
        index_file = VECTOR_STORE_PATH / "index.faiss"
        store_file = VECTOR_STORE_PATH / "index.pkl"
        if not index_file.exists() or not store_file.exists():
            logger.error(f"[EmbeddingService] FAISS index not found in `{VECTOR_STORE_PATH}`. Please rebuild the index.")
            return []  # or raise an exception as needed

        # Load FAISS vectorstore from disk (with dangerous deserialization enabled for pickled data)
        vectorstore = FAISS.load_local(str(VECTOR_STORE_PATH), embeddings=self.embeddings, 
                                       allow_dangerous_deserialization=True)
        # Similarity search in vectorstore
        results = vectorstore.similarity_search_with_score(query_text, k=top_k)

        # Format results as list of snippet dicts
        snippets = [
            {
                "text": doc.page_content,
                "source_file": doc.metadata.get("source_file"),
                "created_at": doc.metadata.get("created_at"),
                "score": float(score)
            }
            for doc, score in results
        ]

        return snippets
