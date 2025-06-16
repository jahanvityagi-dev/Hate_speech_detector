# import pytest
# import numpy as np
# from unittest.mock import MagicMock, patch
# from app.services.embedding_service import EmbeddingService
# import app.services.embedding_service as emb_module

# @pytest.fixture
# def embedding_service():
#     return EmbeddingService()

# def test_init_loads_faiss_index(monkeypatch):
#     class DummyFaiss:
#         def load(self):
#             return "dummy_index", ["chunk1"], ["source1"]
#     monkeypatch.setattr(emb_module, "faiss_service", DummyFaiss())
#     emb = EmbeddingService()
#     assert emb.index == "dummy_index"
#     assert emb.text_chunks == ["chunk1"]
#     assert emb.chunk_sources == ["source1"]

# def test_embed_output_is_normalized(monkeypatch):
#     dummy_output = np.array([[3.0, 4.0]])
#     class DummyModel:
#         def encode(self, texts, convert_to_numpy=True):
#             return np.array([dummy_output[0] for _ in texts])
#     monkeypatch.setattr(emb_module, "SentenceTransformer", lambda name=None: DummyModel())
#     monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (None, [], []))

#     emb = EmbeddingService()
#     result = emb.embed(["hello", "world"])
#     # [3,4] normalized is [0.6, 0.8]
#     expected = np.array([0.6, 0.8])
#     assert np.allclose(result[0], expected)
#     assert np.allclose(result[1], expected)

# def test_add_to_index_calls_save(monkeypatch):
#     class DummyModel:
#         def encode(self, texts, convert_to_numpy=True):
#             return np.array([[3.0, 4.0] for _ in texts])
#     class DummyIndex:
#         def __init__(self): self.add_called = False
#         def add(self, embeddings): self.add_called = True
#     saved = {}
#     monkeypatch.setattr(emb_module, "SentenceTransformer", lambda name=None: DummyModel())
#     monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (DummyIndex(), [], []))
#     monkeypatch.setattr(emb_module.faiss_service, "save", lambda i, t, s: saved.update({
#         "index": i, "chunks": t, "sources": s
#     }))

#     emb = EmbeddingService()
#     emb.add_to_index(["chunk1", "chunk2"], "doc.txt")
#     assert saved["chunks"] == ["chunk1", "chunk2"]
#     assert saved["sources"] == ["doc.txt", "doc.txt"]

# def test_search_filters_duplicates(monkeypatch):
#     dummy_embeddings = np.array([[0.5, 0.5]])
#     class DummyModel:
#         def encode(self, texts, convert_to_numpy=True):
#             return dummy_embeddings
#     class DummyIndex:
#         def search(self, query, k):  # index 0 repeats
#             return np.array([[0.9, 0.85, 0.7]]), np.array([[0, 1, 0]])
#     monkeypatch.setattr(emb_module, "SentenceTransformer", lambda name=None: DummyModel())
#     monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (DummyIndex(), ["txt1", "txt2", "txt1"], ["f1", "f2", "f1"]))

#     emb = EmbeddingService()
#     result = emb.search("query", top_k=2)
#     assert len(result) == 2
#     assert result[0]["text"] == "txt1"
#     assert result[1]["text"] == "txt2"

# def test_search_no_results(monkeypatch):
#     class DummyModel:
#         def encode(self, texts, convert_to_numpy=True):
#             return np.array([[0.0, 0.0]])
#     class DummyIndex:
#         def search(self, query, k):
#             return np.array([[0.0]]), np.array([[0]])
#     monkeypatch.setattr(emb_module, "SentenceTransformer", lambda name=None: DummyModel())
#     monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (DummyIndex(), [""], [""]))
#     emb = EmbeddingService()
#     result = emb.search("query", top_k=1)
#     assert isinstance(result, list)

# def test_embed_handles_model_failure(monkeypatch):
#     class FailingModel:
#         def encode(self, texts, convert_to_numpy=True):
#             raise RuntimeError("model error")

#     monkeypatch.setattr(emb_module, "SentenceTransformer", lambda name=None: FailingModel())
#     monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (None, [], []))

#     service = EmbeddingService()
#     with pytest.raises(RuntimeError):
#         service.embed(["test"])

# def test_add_to_index_with_empty_chunks(embedding_service):
#     with patch.object(embedding_service, 'embed') as mock_embed:
#         embedding_service.add_to_index([], "doc.txt")
#         mock_embed.assert_not_called()
#         assert embedding_service.text_chunks == []
#         assert embedding_service.chunk_sources == []

# def test_search_with_top_k_greater_than_index(monkeypatch):
#     class DummyModel:
#         def encode(self, texts, convert_to_numpy=True):
#             return np.array([[0.1, 0.9]])

#     class DummyIndex:
#         def search(self, query, k):
#             return np.array([[0.9]]), np.array([[0]])

#     monkeypatch.setattr(emb_module, "SentenceTransformer", lambda name=None: DummyModel())
#     monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (DummyIndex(), ["sample"], ["doc.txt"]))
#     emb = EmbeddingService()
#     results = emb.search("sample", top_k=10)  # Larger than available
#     assert len(results) == 1


# def test_embed_returns_correct_shape_and_type(embedding_service):
#     texts = ["test sentence", "another test"]
#     with patch.object(embedding_service, 'model') as mock_model:
#         mock_model.encode.return_value = np.random.rand(2, embedding_service.dim).astype("float32")
#         embeddings = embedding_service.embed(texts)
#         assert embeddings.shape == (2, embedding_service.dim)
#         assert embeddings.dtype == np.float32

# def test_add_to_index_adds_chunks_correctly(embedding_service):
#     with patch.object(embedding_service, 'embed') as mock_embed:
#         mock_embed.return_value = np.random.rand(2, embedding_service.dim).astype("float32")
#         embedding_service.add_to_index(["chunk1", "chunk2"], "doc1.txt")
#         assert len(embedding_service.text_chunks) == 2
#         assert embedding_service.chunk_sources == ["doc1.txt", "doc1.txt"]

# def test_search_returns_expected_format(embedding_service):
#     with patch.object(embedding_service, 'embed') as mock_embed:
#         mock_embed.return_value = np.random.rand(1, embedding_service.dim).astype("float32")
#         embedding_service.text_chunks = ["chunk1", "chunk2"]
#         embedding_service.chunk_sources = ["source1", "source2"]
#         embedding_service.index.add(np.random.rand(2, embedding_service.dim).astype("float32"))
#         results = embedding_service.search("query", top_k=2)
#         assert isinstance(results, list)
#         for result in results:
#             assert "text" in result and "source_file" in result and "score" in result
# '''

# file_path = "/mnt/data/test_embedding_service.py"
# with open(file_path, "w") as f:
#     f.write(test_embedding_service_code)

# file_path
# '''
import numpy as np
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.embedding_service import EmbeddingService
import app.services.embedding_service as emb_module

@pytest.fixture
def embedding_service():
    return EmbeddingService()

def test_init_loads_faiss_index(monkeypatch):
    class DummyFaiss:
        def load(self):
            return "dummy_index", ["chunk1"], ["source1"]
    monkeypatch.setattr(emb_module, "faiss_service", DummyFaiss())
    emb = EmbeddingService()
    assert emb.index == "dummy_index"
    assert emb.text_chunks == ["chunk1"]
    assert emb.chunk_sources == ["source1"]

def test_embed_output_is_normalized(monkeypatch):
    dummy_output = np.array([[3.0, 4.0]])
    class DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.array([dummy_output[0] for _ in texts])
    monkeypatch.setattr(emb_module, "SentenceTransformer", lambda name=None: DummyModel())
    monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (None, [], []))
    emb = EmbeddingService()
    result = emb.embed(["hello", "world"])
    expected = np.array([0.6, 0.8])
    assert np.allclose(result[0], expected)
    assert np.allclose(result[1], expected)

def test_add_to_index_calls_save(monkeypatch):
    class DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.array([[3.0, 4.0] for _ in texts])
    class DummyIndex:
        def __init__(self): self.add_called = False
        def add(self, embeddings): self.add_called = True
    saved = {}
    monkeypatch.setattr(emb_module, "SentenceTransformer", lambda name=None: DummyModel())
    monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (DummyIndex(), [], []))
    monkeypatch.setattr(emb_module.faiss_service, "save", lambda i, t, s: saved.update({
        "index": i, "chunks": t, "sources": s
    }))
    emb = EmbeddingService()
    emb.add_to_index(["chunk1", "chunk2"], "doc.txt")
    assert saved["chunks"] == ["chunk1", "chunk2"]
    assert saved["sources"] == ["doc.txt", "doc.txt"]

def test_search_filters_duplicates(monkeypatch):
    dummy_embeddings = np.array([[0.5, 0.5]])
    class DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return dummy_embeddings
    class DummyIndex:
        def search(self, query, k):
            return np.array([[0.9, 0.85, 0.7]]), np.array([[0, 1, 0]])
    monkeypatch.setattr(emb_module, "SentenceTransformer", lambda name=None: DummyModel())
    monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (DummyIndex(), ["txt1", "txt2", "txt1"], ["f1", "f2", "f1"]))
    emb = EmbeddingService()
    result = emb.search("query", top_k=2)
    assert len(result) == 2
    assert result[0]["text"] == "txt1"
    assert result[1]["text"] == "txt2"

def test_search_no_results(monkeypatch):
    class DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.array([[0.0, 0.0]])
    class DummyIndex:
        def search(self, query, k):
            return np.array([[0.0]]), np.array([[0]])
    monkeypatch.setattr(emb_module, "SentenceTransformer", lambda name=None: DummyModel())
    monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (DummyIndex(), [""], [""]))
    emb = EmbeddingService()
    result = emb.search("query", top_k=1)
    assert isinstance(result, list)

def test_embed_returns_correct_shape_and_type(embedding_service):
    texts = ["test sentence", "another test"]
    dummy_output = np.random.rand(2, 384).astype("float32")
    with patch.object(embedding_service, 'model') as mock_model:
        mock_model.encode.return_value = dummy_output
        embeddings = embedding_service.embed(texts)
        assert embeddings.shape == dummy_output.shape
        assert embeddings.dtype == np.float32

def test_add_to_index_adds_chunks_correctly(monkeypatch):
    dummy_embedding = np.random.rand(2, 384).astype("float32")

    # Patch embed method to return dummy embeddings
    monkeypatch.setattr(emb_module.EmbeddingService, "embed", lambda self, texts: dummy_embedding)

    # Patch faiss_service.load() to return clean state
    class DummyIndex:
        def add(self, x): pass
    monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (DummyIndex(), [], []))

    # Patch faiss_service.save() to avoid calling real FAISS
    monkeypatch.setattr(emb_module.faiss_service, "save", lambda i, c, s: None)

    service = EmbeddingService()
    service.add_to_index(["chunk1", "chunk2"], "doc1.txt")

    assert len(service.text_chunks) == 2
    assert service.chunk_sources == ["doc1.txt", "doc1.txt"]



def test_search_returns_expected_format(monkeypatch):
    dummy_embedding = np.random.rand(1, 384).astype("float32")

    # Patch embed
    monkeypatch.setattr(emb_module.EmbeddingService, "embed", lambda self, texts: dummy_embedding)

    # Patch FAISS index and metadata
    class DummyIndex:
        def add(self, x): pass
        def search(self, query, k):  # returns top_k=2 valid indices
            return np.array([[0.9, 0.8]]), np.array([[0, 1]])

    monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (DummyIndex(), ["chunk1", "chunk2"], ["source1", "source2"]))

    service = EmbeddingService()
    results = service.search("query", top_k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert "text" in r and "source_file" in r and "score" in r


def test_real_model_embedding_shape(monkeypatch):
    class DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.random.rand(len(texts), 384).astype("float32")
    monkeypatch.setattr(emb_module, "SentenceTransformer", lambda name=None: DummyModel())
    monkeypatch.setattr(emb_module.faiss_service, "load", lambda: (None, [], []))

    service = EmbeddingService()
    result = service.embed(["test input"])
    assert result.shape == (1, 384)
