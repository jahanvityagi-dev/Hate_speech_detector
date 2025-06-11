from app.agents.hybrid_retriever_agent import HybridRetrieverAgent

def test_retrieve_returns_results():
    agent = HybridRetrieverAgent()
    results = agent.retrieve("hate speech targeting vulnerable groups", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "text" in r
        assert "source_file" in r
        assert "score" in r
