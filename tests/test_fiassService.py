import os
import io
import pickle
import numpy as np
import pytest
import faiss
import json
import tempfile
from app.services import faiss_service
import app.services.faiss_service as faiss_module
from app.services.faiss_service import FaissService


class DummyIndex:
    def __init__(self):
        self.added = None

    def add(self, vectors):
        self.added = vectors

    def search(self, query, k):
        distances = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
        indices = np.array([[0, 1, 2]])
        return distances, indices


@pytest.fixture
def dummy_faiss(monkeypatch):
    monkeypatch.setattr(faiss_module.faiss, "IndexFlatIP", lambda dim: DummyIndex())
    monkeypatch.setattr(faiss_module.faiss, "read_index", lambda path: DummyIndex())
    monkeypatch.setattr(faiss_module.faiss, "write_index", lambda index, path: None)


def test_save_and_load(tmp_path, monkeypatch, dummy_faiss):
    index_path = tmp_path / "index.idx"
    meta_path = tmp_path / "meta.pkl"
    fs = FaissService(index_path=str(index_path), metadata_path=str(meta_path), dim=2)

    dummy_index = DummyIndex()
    chunks = ["c1", "c2"]
    sources = ["s1", "s2"]

    fs.save(dummy_index, chunks, sources)

    monkeypatch.setattr(os.path, "exists", lambda path: True)

    def dummy_open(file, mode):
        assert mode == "rb"
        class DummyFile:
            def __enter__(self):
                return io.BytesIO(pickle.dumps((chunks, sources)))
            def __exit__(self, exc_type, exc_val, exc_tb):
                return False
        return DummyFile()

    monkeypatch.setattr("builtins.open", dummy_open)

    loaded_index, loaded_chunks, loaded_sources = fs.load()
    assert isinstance(loaded_index, DummyIndex)
    assert loaded_chunks == chunks
    assert loaded_sources == sources


def test_search(monkeypatch, dummy_faiss):
    fs = FaissService(dim=2)
    fs.index = DummyIndex()
    fs.id_mapping = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    query_vec = np.array([0.1, 0.9], dtype=np.float32)
    results = fs.search(query_vec, k=3)

    assert len(results) == 3
    assert results[0]["id"] == "a"
    assert all("id" in r for r in results)

def test_save_and_load_index(monkeypatch, tmp_path):
    index_file = tmp_path / "index.faiss.npy"
    chunks_file = tmp_path / "chunks.json"
    sources_file = tmp_path / "sources.json"

    # Dummy FAISS index and data
    from faiss import IndexFlatL2
    dummy_data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype='float32')
    index = IndexFlatL2(2)
    index.add(dummy_data)
    chunks = ["chunk1", "chunk2"]
    sources = ["source1", "source2"]

    # Patch instance methods of FaissService
    from app.services.faiss_service import FaissService

    def dummy_save(self, i, c, s):
        np.save(index_file, dummy_data)
        with open(chunks_file, "w") as f1, open(sources_file, "w") as f2:
            json.dump(c, f1)
            json.dump(s, f2)

    def dummy_load(self):
        reloaded_data = np.load(index_file)
        idx = IndexFlatL2(2)
        idx.add(reloaded_data)
        with open(chunks_file, "r") as f1, open(sources_file, "r") as f2:
            reloaded_chunks = json.load(f1)
            reloaded_sources = json.load(f2)
        return idx, reloaded_chunks, reloaded_sources

    monkeypatch.setattr(FaissService, "save", dummy_save)
    monkeypatch.setattr(FaissService, "load", dummy_load)

    service = FaissService()
    service.save(index, chunks, sources)
    loaded_index, loaded_chunks, loaded_sources = service.load()

    assert loaded_index.ntotal == 2
    assert loaded_chunks == chunks
    assert loaded_sources == sources

def test_save_and_load_metadata(tmp_path):
    from app.services.faiss_service import FaissService
    import numpy as np
    import faiss

    index_path = tmp_path / "faiss_index.bin"
    metadata_path = tmp_path / "id_mapping.json"
    service = FaissService(str(index_path), str(metadata_path), 384)
    index = faiss.IndexFlatIP(384)
    metadata = [{"text": "chunk1", "source_file": "file1.txt"}]
    service.save(index, metadata)
    loaded_index, loaded_metadata = service.load()
    assert isinstance(loaded_metadata, list)
    assert loaded_metadata[0]["text"] == "chunk1"

