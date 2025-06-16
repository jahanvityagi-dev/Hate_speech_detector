# import pytest
# from app.services.moderation_pipeline import ModerationPipeline
# from app.agents.hate_speech_agent import HateSpeechDetectionAgent
# from app.agents.hybrid_retriever_agent import HybridRetrieverAgent
# from app.agents.policy_reasoning_agent import PolicyReasoningAgent
# from app.agents.action_recommender_agent import ActionRecommenderAgent

# class DummyHateSpeechAgent:
#     def classify(self, text):
#         return {"label": "Hate", "explanation": "Mock explanation."}

# class DummyRetriever:
#     def retrieve(self, text):
#         return [
#             {"text": "Policy text 1", "source_file": "policy1.txt", "score": 0.9},
#             {"text": "Policy text 2", "source_file": "policy2.txt", "score": 0.85},
#         ]

# class DummyPolicyReasoner:
#     def generate_explanation(self, input_text, label, policy_snippets):
#         return "Mock reasoning based on policies."

# class DummyActionRecommender:
#     def recommend(self, label: str) -> dict:
#         return {"action": "Ban", "reason": "Mock action reason."}

# def test_full_pipeline(monkeypatch):
#     pipeline = ModerationPipeline()
#     pipeline.classifier = DummyHateSpeechAgent()
#     pipeline.retriever = DummyRetriever()
#     pipeline.reasoner = DummyPolicyReasoner()
#     pipeline.recommender = DummyActionRecommender()

#     input_text = "People from XYZ are bad."
#     result = pipeline.run(input_text)

#     assert result["input"] == input_text
#     assert result["label"] == "Hate"
#     assert result["classification_reason"] == "Mock explanation."
#     assert len(result["policy_snippets"]) == 2
#     assert result["explanation"] == "Mock reasoning based on policies."
#     assert result["action"] == "Ban"
#     #assert result["action"] == "Flag"
#     assert result["action_reason"] == "Mock action reason."
#     #assert result["action_reason"] == "Mock action for label: Hate"
import pytest
from app.services.moderation_pipeline import ModerationPipeline

class DummyClassifier:
    def __init__(self):
        self.called = False
    def classify(self, text):
        self.called = True
        return {"label": "Toxic", "explanation": "Dummy classification reason"}

class DummyRetriever:
    def __init__(self):
        self.called = False
    def retrieve(self, text):
        self.called = True
        return [
            {"text": "Relevant policy snippet", "source_file": "policy.txt", "score": 0.95}
        ]

class DummyReasoner:
    def __init__(self):
        self.called = False
    def generate_explanation(self, input_text, label, policy_snippets):
        self.called = True
        return f"Because the content is '{label}' as per {policy_snippets[0]['text']}"

class DummyRecommender:
    def __init__(self):
        self.called = False
    def recommend(self, label):
        self.called = True
        return {"action": "Warn", "reason": "Dummy action reason for label "+label}

# ✅ Subclass to inject the dummy components
class DummyModerationPipeline(ModerationPipeline):
    def __init__(self):
        self.classifier = DummyClassifier()
        self.retriever = DummyRetriever()
        self.reasoner = DummyReasoner()
        self.recommender = DummyRecommender()

def test_run_pipeline():
    pipeline = DummyModerationPipeline()
    input_text = "Dummy input text"
    result = pipeline.run(input_text)

    assert pipeline.classifier.called
    assert pipeline.retriever.called
    assert pipeline.reasoner.called
    assert pipeline.recommender.called

    assert result["input"] == input_text
    assert result["label"] == "Toxic"
    assert result["classification_reason"] == "Dummy classification reason"
    assert isinstance(result["policy_snippets"], list)
    assert result["policy_snippets"][0]["text"] == "Relevant policy snippet"
    assert "Because the content is 'Toxic'" in result["explanation"]
    assert result["action"] == "Warn"
    assert result["action_reason"].startswith("Dummy action reason")
