import os
import faiss
import numpy as np
from pathlib import Path
from app.services.embedding_service import EmbeddingService
from app.agents.error_handler_agent import ErrorHandlerAgent
from pathlib import Path
from app.services.embedding_service import EmbeddingService
POLICY_DIR = Path("data/policy_docs/")
EMBEDDING_DIM = 384  # for MiniLM
TOP_K = 3

class HybridRetrieverAgent:
    """
    Loads and indexes policy documents, retrieves top relevant chunks
    using cosine similarity on normalized sentence embeddings.
    """
    error_handler = ErrorHandlerAgent()
    def __init__(self, policy_dir="data/policy_docs"):
        self.embedding_service = EmbeddingService()
        self.policy_dir = Path(policy_dir)
        self._load_policy_documents()
        
    def _load_policy_documents(self):
        try:
            for file in self.policy_dir.glob("*.txt"):
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()

                raw_chunks = [para.strip() for para in content.split("\n\n") if len(para.strip()) > 30]
                combined_chunks = []
                skip_next = False

                for i in range(len(raw_chunks)):
                    if skip_next:
                        skip_next = False
                        continue

                    if raw_chunks[i].lower().startswith("title:"):
                        
                        if i + 1 < len(raw_chunks):
                            combined = f"{raw_chunks[i]}\n\n{raw_chunks[i+1]}"
                            combined_chunks.append(combined)
                            skip_next = True
                        else:
                            combined_chunks.append(raw_chunks[i])
                    else:
                        combined_chunks.append(raw_chunks[i])

                print(f"Loaded {len(combined_chunks)} combined chunks from {file.name}")
                self.embedding_service.add_to_index(combined_chunks, file.name)

            print(f"Built embeddings with shape: {self.embedding_service.index.ntotal, 384}")

        except Exception as e:
            self.error_handler.handle_error("HybridRetrieverAgent::_load_policy_documents", str(e))

    @error_handler.handle_errors(agent_name="HybridRetrieverAgent", method="retrieve")
    def retrieve(self, input_text, top_k=3):
        return self.embedding_service.search(input_text, top_k)
        