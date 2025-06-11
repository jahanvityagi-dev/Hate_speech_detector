import pytest
from app.services.moderation_pipeline import ModerationPipeline
from app.agents.hate_speech_agent import HateSpeechDetectionAgent
from app.agents.hybrid_retriever_agent import HybridRetrieverAgent
from app.agents.policy_reasoning_agent import PolicyReasoningAgent
from app.agents.action_recommender_agent import ActionRecommenderAgent

class DummyHateSpeechAgent:
    def classify(self, text):
        return {"label": "Hate", "explanation": "Mock explanation."}

class DummyRetriever:
    def retrieve(self, text):
        return [
            {"text": "Policy text 1", "source_file": "policy1.txt", "score": 0.9},
            {"text": "Policy text 2", "source_file": "policy2.txt", "score": 0.85},
        ]

class DummyPolicyReasoner:
    def generate_explanation(self, input_text, label, policy_snippets):
        return "Mock reasoning based on policies."

class DummyActionRecommender:
    def recommend(self, label: str) -> dict:
        return {"action": "Ban", "reason": "Mock action reason."}

def test_full_pipeline(monkeypatch):
    pipeline = ModerationPipeline()
    pipeline.classifier = DummyHateSpeechAgent()
    pipeline.retriever = DummyRetriever()
    pipeline.reasoner = DummyPolicyReasoner()
    pipeline.recommender = DummyActionRecommender()

    input_text = "People from XYZ are bad."
    result = pipeline.run(input_text)

    assert result["input"] == input_text
    assert result["label"] == "Hate"
    assert result["classification_reason"] == "Mock explanation."
    assert len(result["policy_snippets"]) == 2
    assert result["explanation"] == "Mock reasoning based on policies."
    assert result["action"] == "Ban"
    #assert result["action"] == "Flag"
    assert result["action_reason"] == "Mock action reason."
    #assert result["action_reason"] == "Mock action for label: Hate"
