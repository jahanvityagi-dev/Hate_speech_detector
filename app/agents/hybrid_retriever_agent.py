import os
import faiss
import numpy as np
from pathlib import Path
from app.services.embedding_service import EmbeddingService

POLICY_DIR = Path("data/policy_docs/")
EMBEDDING_DIM = 384  # for MiniLM
TOP_K = 3

# class HybridRetrieverAgent:
#     def __init__(self):
#         self.embedding_service = EmbeddingService()
#         self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
#         self.text_chunks = []
#         self.chunk_sources = []
#         self._load_policy_documents()

#     def _load_policy_documents(self):
#         for file in POLICY_DIR.glob("*.txt"):
#             with open(file, "r", encoding="utf-8") as f:
#                 content = f.read()

#             chunks = [para.strip() for para in content.split("\n\n") if len(para.strip()) > 30]
#             print(f"Loaded {len(chunks)} chunks from {file.name}")

#             self.text_chunks.extend(chunks)
#             self.chunk_sources.extend([file.name] * len(chunks))

#         if not self.text_chunks:
#             raise ValueError("No valid text chunks found in policy_docs/.")

#         embeddings = self.embedding_service.embed(self.text_chunks)
#         print(f"Built embeddings with shape: {embeddings.shape}")
#         self.index.add(np.array(embeddings).astype("float32"))

#     def retrieve(self, input_text: str, k: int = TOP_K):
#         """
#         Embeds the input_text and retrieves top-k policy chunks.
#         Returns: List of dicts: {text, source_file, score}
#         """
#         query_embedding = self.embedding_service.embed([input_text])
#         D, I = self.index.search(query_embedding.astype("float32"), k)

#         results = []
#         for rank, idx in enumerate(I[0]):
#             results.append({
#                 "text": self.text_chunks[idx],
#                 "source_file": self.chunk_sources[idx],
#                 "score": float(D[0][rank])
#             })

#         return results
from pathlib import Path
from app.services.embedding_service import EmbeddingService

class HybridRetrieverAgent:
    """
    Loads and indexes policy documents, retrieves top relevant chunks
    using cosine similarity on normalized sentence embeddings.
    """

    def __init__(self, policy_dir="data/policy_docs"):
        self.embedding_service = EmbeddingService()
        self.policy_dir = Path(policy_dir)
        self._load_policy_documents()

    def _load_policy_documents(self):
        """
        Read and index all policy text files.
        """
        for file in self.policy_dir.glob("*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()

            # Chunk on paragraphs
            chunks = [para.strip() for para in content.split("\n\n") if len(para.strip()) > 30]
            print(f"Loaded {len(chunks)} chunks from {file.name}")
            self.embedding_service.add_to_index(chunks, file.name)

        print(f"Built embeddings with shape: {self.embedding_service.index.ntotal, 384}")

    def retrieve(self, input_text, top_k=3):
        return self.embedding_service.search(input_text, top_k)
