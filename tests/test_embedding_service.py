import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from app.services.embedding_service import EmbeddingService


@pytest.fixture
def dummy_embedding():
    return np.array([[0.5] * 384], dtype="float32")


@pytest.fixture
def dummy_metadata():
    return [{
        "id": 0,
        "text": "Dummy policy text",
        "source_file": "dummy.txt",
        "created_at": "2025-06-18T12:00:00"
    }]


@pytest.fixture
def embedding_service(dummy_embedding, dummy_metadata):
    with patch("app.services.embedding_service.SentenceTransformer") as mock_model, \
         patch("app.services.embedding_service.faiss_service") as mock_faiss_service, \
         patch("os.path.exists", return_value=True):
        
        mock_model.return_value.encode.return_value = dummy_embedding
        dummy_index = MagicMock()
        dummy_index.ntotal = 1
        dummy_index.search.return_value = (np.array([[0.99]]), np.array([[0]]))
        
        mock_faiss_service.load.return_value = (dummy_index, dummy_metadata)
        
        return EmbeddingService()


def test_embed_returns_normalized(embedding_service):
    output = embedding_service.embed(["Test text"])
    assert np.allclose(np.linalg.norm(output, axis=1), 1.0, atol=1e-5)


def test_search_returns_expected_fields(embedding_service):
    results = embedding_service.search("test query", top_k=1)
    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]
    assert "text" in result
    assert "source_file" in result
    assert "score" in result
    assert "created_at" in result


def test_add_to_index_adds_chunks_correctly(dummy_embedding):
    with patch("app.services.embedding_service.SentenceTransformer") as mock_model, \
         patch("app.services.embedding_service.faiss_service") as mock_faiss_service, \
         patch("os.path.exists", return_value=True):  # 🚀 Force files to "exist"
        
        mock_model.return_value.encode.return_value = dummy_embedding
        dummy_index = MagicMock()
        dummy_index.ntotal = 0  # simulate empty index
        dummy_index.add = MagicMock()

        mock_faiss_service.load.return_value = (dummy_index, [])

        service = EmbeddingService()

        chunks = ["Test policy paragraph 1", "Test policy paragraph 2"]
        service.add_to_index(chunks, "dummy_source.txt")

        # 🚀 Assert .add() was called
        dummy_index.add.assert_called_once()


def test_regenerate_faiss_index(monkeypatch):
    with patch("app.services.embedding_service.SentenceTransformer") as mock_model, \
         patch("app.services.embedding_service.faiss_service.save", lambda i, m: None), \
         patch("os.path.exists", lambda path: False):
        
        dummy_embedding = np.array([[0.5] * 384], dtype="float32")
        mock_model.return_value.encode.return_value = dummy_embedding

        # Should not raise any exceptions
        from app.services.embedding_service import regenerate_faiss_index_if_missing
        regenerate_faiss_index_if_missing(force=True)
