import faiss
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from app.services.faiss_service import FaissService
import numpy as np

faiss_service = FaissService()

class EmbeddingService:
    """
    Provides methods for generating normalized embeddings and managing FAISS index.
    Uses cosine similarity via inner product on unit-normalized vectors.
    """

    # def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
    #     self.model = SentenceTransformer(embedding_model)
    #     self.index = faiss.IndexFlatIP(384)  # Use inner product for cosine similarity
    #     self.text_chunks = []
    #     self.chunk_sources = []
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index, self.text_chunks, self.chunk_sources = faiss_service.load()


    def embed(self, texts):
        """
        Generate normalized embeddings for a list of texts.
        """
        raw_embeddings = self.model.encode(texts, convert_to_numpy=True)
        normalized_embeddings = normalize(raw_embeddings, axis=1)
        return normalized_embeddings

    # def add_to_index(self, chunks, source_file):
    #     """
    #     Add text chunks (and source) to FAISS index.
    #     """
    #     embeddings = self.embed(chunks)
    #     self.index.add(embeddings)
    #     self.text_chunks.extend(chunks)
    #     self.chunk_sources.extend([source_file] * len(chunks))
    def add_to_index(self, chunks, source_file):
        embeddings = self.embed(chunks)
        self.index.add(embeddings)
        self.text_chunks.extend(chunks)
        self.chunk_sources.extend([source_file] * len(chunks))
        
        # Save updated FAISS index and metadata
        faiss_service.save(self.index, self.text_chunks, self.chunk_sources)


    # def search(self, query_text, top_k=3):
    #     """
    #     Retrieve top-k policy chunks for given input text.
    #     """
    #     query_embedding = self.embed([query_text])
    #     D, I = self.index.search(query_embedding.astype("float32"), top_k)

    #     results = []
    #     for idx, score in zip(I[0], D[0]):
    #         results.append({
    #             "text": self.text_chunks[idx],
    #             "source_file": self.chunk_sources[idx],
    #             "score": round(float(score), 4)  # Higher = better
    #         })
    #     return results
    def search(self, query_text, top_k=3):
        query_embedding = self.embed([query_text])
        D, I = self.index.search(query_embedding.astype("float32"), top_k * 3)  # fetch more initially

        seen_texts = set()
        seen_sources = set()
        results = []

        for idx, score in zip(I[0], D[0]):
            text = self.text_chunks[idx]
            source = self.chunk_sources[idx]

            # Avoid exact duplicates
            if text in seen_texts:
                continue

            # Optionally diversify by source file (document)
            # if source in seen_sources:
            #     continue

            results.append({
                "text": text,
                "source_file": source,
                "score": round(float(score), 4)
            })
            seen_texts.add(text)
            seen_sources.add(source)

            if len(results) >= top_k:
                break

        return results

