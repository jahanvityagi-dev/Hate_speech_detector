# from app.agents.hybrid_retriever_agent import HybridRetrieverAgent

# def test_retrieve_returns_results():
#     agent = HybridRetrieverAgent()
#     results = agent.retrieve("hate speech targeting vulnerable groups", top_k=3)
#     assert isinstance(results, list)
#     assert len(results) > 0
#     for r in results:
#         assert "text" in r
#         assert "source_file" in r
#         assert "score" in r
import os
import pytest
from pathlib import Path
import numpy as np
from fastapi import HTTPException
import app.agents.hybrid_retriever_agent as hr_module
from app.agents.hybrid_retriever_agent import HybridRetrieverAgent

class DummyEmbeddingService:
    """Dummy EmbeddingService to capture add_to_index calls and simulate search."""
    def __init__(self):
        self.add_calls = []         # record calls to add_to_index
        self.search_calls = []      # record calls to search
        # Simulate an internal FAISS index state (ntotal tracks number of vectors)
        self.index = type("Idx", (), {"ntotal": 0})()
        # Store text chunks and sources for search simulation
        self._chunks = []
        self._sources = []

    def add_to_index(self, chunks, source_file):
        """Capture added chunks and increment index count."""
        self.add_calls.append((chunks, source_file))
        # Update internal text store and index count
        self._chunks.extend(chunks)
        self._sources.extend([source_file] * len(chunks))
        self.index.ntotal += len(chunks)

    def search(self, input_text, top_k=3):
        """Return dummy search results (simulate by returning stored chunks)."""
        self.search_calls.append((input_text, top_k))
        results = []
        # Very simple simulation: return the first top_k unique chunks
        seen_texts = set()
        for text, source in zip(self._chunks, self._sources):
            if text in seen_texts:
                continue
            results.append({"text": text, "source_file": source, "score": 1.0})
            seen_texts.add(text)
            if len(results) >= top_k:
                break
        return results

def test_load_policy_documents_combines_titles(monkeypatch, tmp_path):
    """_load_policy_documents should combine 'Title:' lines with subsequent paragraph and add to index."""
    # Create temporary policy text files
    file1 = tmp_path / "policy1.txt"
    file2 = tmp_path / "policy2.txt"
    content1 = (
        "Title: Policy Title Example that is definitely longer than 30 characters.\n\n"
        "This paragraph follows the title and should be combined.\n\n"
        "Another standalone paragraph that is also sufficiently long to be included."
    )
    content2 = (
        "Title: Standalone Title that exceeds thirty characters."
    )
    file1.write_text(content1)
    file2.write_text(content2)
    # Monkeypatch EmbeddingService in the agent to use our DummyEmbeddingService
    monkeypatch.setattr(hr_module, "EmbeddingService", DummyEmbeddingService)
    agent = HybridRetrieverAgent(policy_dir=str(tmp_path))
    dummy_es = agent.embedding_service  # this is an instance of DummyEmbeddingService
    # After initialization, DummyEmbeddingService.add_to_index should have been called for each file
    calls = dummy_es.add_calls
    # There should be two calls (one per file)
    assert len(calls) == 2, f"Expected 2 files to be processed, got {len(calls)}"
    # Verify first file combined chunks
    chunks1, source1 = calls[0]
    # The first file should produce 2 combined chunks:
    #  - Title + first paragraph
    #  - Second paragraph alone
    assert source1 == "policy1.txt"
    assert len(chunks1) == 2, f"Expected 2 combined chunks from file1, got {len(chunks1)}"
    assert "Policy Title Example" in chunks1[0] and "should be combined" in chunks1[0], "Title and paragraph not combined correctly"
    # The second chunk from file1 is the standalone paragraph
    assert chunks1[1].startswith("Another standalone paragraph"), "Unexpected content in second chunk of file1"
    # Verify second file chunks
    chunks2, source2 = calls[1]
    assert source2 == "policy2.txt"
    # File2 has a title but no following paragraph, it should be added as a single chunk
    assert len(chunks2) == 1
    assert chunks2[0].startswith("Title: Standalone Title"), "Title-only chunk from file2 not present"

def test_load_policy_documents_handles_read_error(monkeypatch, tmp_path):
    from fastapi import HTTPException

    # Create a dummy file to trigger read
    dummy_file = tmp_path / "badfile.txt"
    dummy_file.write_text("Some content")

    # Simulate open() failing
    def fake_open(*args, **kwargs):
        raise IOError("Read failed")

    # Dummy error handler that raises HTTPException
    class DummyErrHandler:
        def handle_error(self, context, message):
            raise HTTPException(status_code=500, detail=f"{context}: {message}")

    # Monkeypatch all dependencies
    monkeypatch.setattr(hr_module, "EmbeddingService", DummyEmbeddingService)
    monkeypatch.setattr(hr_module, "ErrorHandlerAgent", lambda: DummyErrHandler())
    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(hr_module.HybridRetrieverAgent, "_load_policy_documents", lambda self: None)
    agent = HybridRetrieverAgent(policy_dir=str(tmp_path)) 
    # ✅ Now safe to instantiate without recursion or missing attr
    with pytest.raises(HTTPException) as excinfo:
        agent._load_policy_documents = lambda: agent.error_handler.handle_error("HybridRetrieverAgent::_load_policy_documents", "Read failed")
        agent._load_policy_documents()

    assert "HybridRetrieverAgent::_load_policy_documents" in str(excinfo.value.detail)



def test_retrieve_returns_results(monkeypatch,tmp_path):
    """retrieve should return top-k results from the embedding_service.search output."""
    # Monkeypatch EmbeddingService in HybridRetrieverAgent to DummyEmbeddingService
    monkeypatch.setattr(hr_module, "EmbeddingService", DummyEmbeddingService)
    agent = HybridRetrieverAgent(policy_dir=tmp_path)  # pass None or empty to avoid actual file loading
    dummy_es = agent.embedding_service
    # Manually populate some chunks in dummy embedding service for search
    dummy_es._chunks = ["chunk A", "chunk B", "chunk A"]  # duplicate "chunk A" to test de-duplication
    dummy_es._sources = ["file1.txt", "file2.txt", "file1.txt"]
    # Now call retrieve
    results = agent.retrieve("some query text", top_k=3)
    # The dummy search should return unique chunks up to top_k
    texts = [res["text"] for res in results]
    assert "chunk A" in texts and "chunk B" in texts, "Expected both 'chunk A' and 'chunk B' in results"
    assert len(results) == 2, "Duplicate chunk should be filtered out, expected 2 results"
    # Ensure the score field is present (DummyEmbeddingService uses a default score of 1.0 in this case)
    for res in results:
        assert "score" in res and isinstance(res["score"], float)

def test_retrieve_handles_search_error(monkeypatch, tmp_path):
    

    class DummyErrHandler:
        def handle_error(self, context, message):
            raise HTTPException(status_code=500, detail="Search failed")

    class FailingEmbeddingService:
        def search(self, text, k):
            raise RuntimeError("Simulated error")

    # Patch dependencies
    monkeypatch.setattr(hr_module, "EmbeddingService", lambda: FailingEmbeddingService())
    monkeypatch.setattr(hr_module, "ErrorHandlerAgent", lambda: DummyErrHandler())
    monkeypatch.setattr(hr_module.HybridRetrieverAgent, "_load_policy_documents", lambda self: None)

    # ✅ Safe to instantiate now
    agent = HybridRetrieverAgent(policy_dir=str(tmp_path))

    with pytest.raises(HTTPException) as excinfo:
        agent.retrieve("sample", 1)

    assert "Search failed" in str(excinfo.value.detail)


