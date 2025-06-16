# from app.agents.action_recommender_agent import ActionRecommenderAgent

# def test_recommend_action_hate():
#     agent = ActionRecommenderAgent()
#     result = agent.recommend_action("Hate")
#     assert result["action"] == "Ban"
#     assert "hate" in result["reason"].lower()

# def test_recommend_action_offensive():
#     agent = ActionRecommenderAgent()
#     result = agent.recommend_action("Offensive")
#     assert result["action"] == "Warn"
#     assert "offensive" in result["reason"].lower()

# def test_recommend_action_neutral():
#     agent = ActionRecommenderAgent()
#     result = agent.recommend_action("Neutral")
#     assert result["action"] == "Allow"
#     assert "no action" in result["reason"].lower()

# def test_recommend_action_unknown():
#     agent = ActionRecommenderAgent()
#     result = agent.recommend_action("SomethingElse")
#     assert result["action"] == "Review"
#     assert "manual review" in result["reason"].lower()
import pytest
from fastapi import HTTPException
from app.agents.action_recommender_agent import ActionRecommenderAgent

class DummyErrorHandler:
    """Dummy error handler to capture errors without raising HTTPException."""
    def __init__(self):
        self.called = False
        self.context = None
        self.message = None

    def handle_error(self, context, message):
        """Store the error context and message, return dummy error dict."""
        self.called = True
        self.context = context
        self.message = message
        return {"error": f"{context}: {message}"}

def test_recommend_known_labels():
    """It should return correct action and reason for each known label in the action map."""
    agent = ActionRecommenderAgent()
    # Test a variety of known labels and ensure outputs match expected actions/reasons
    known_labels = {
        "Hate": {"action": "Ban", "reason": "Content promotes hate or dehumanization, which violates platform policy."},
        "Toxic": {"action": "Warn", "reason": "Content is toxic in tone but not directly hateful."},
        "Offensive": {"action": "Flag", "reason": "Content is offensive and should be reviewed by a moderator."},
        "Neutral": {"action": "Allow", "reason": "No harmful content detected."},
        "Ambiguous": {"action": "Manual Review", "reason": "Content is unclear; requires human evaluation."}
    }
    for label, expected in known_labels.items():
        result = agent.recommend(label)
        assert result == expected, f"Expected {expected} for label '{label}', got {result}"

def test_recommend_unknown_label():
    """Unknown labels should default to 'Manual Review' action with appropriate reason."""
    agent = ActionRecommenderAgent()
    result = agent.recommend("UnknownLabel")
    # The default for unknown labels is Manual Review with a specific reason
    expected = {
        "action": "Manual Review",
        "reason": "Unknown label; escalate to a moderator."
    }
    assert result == expected, f"Unexpected recommendation for unknown label: {result}"

def test_recommend_exception_handling(monkeypatch):
    """If an exception occurs during lookup, it should delegate to error handler and return its output."""
    agent = ActionRecommenderAgent()
    # Monkeypatch the agent's action_map.get to force an exception
    def dummy_get(key, default=None):
        raise RuntimeError("Failed to get")
    agent.action_map = type("DummyMap", (), {"get": staticmethod(dummy_get)})()
    # Replace the agent's error_handler with a dummy to capture the call
    dummy_handler = DummyErrorHandler()
    agent.error_handler = dummy_handler

    result = agent.recommend("Hate")  # This will trigger dummy_get and raise
    # The dummy error handler should have been called and its return should be propagated
    assert dummy_handler.called, "Error handler was not called on exception"
    assert dummy_handler.context == "ActionRecommenderAgent::recommend"
    assert "Failed to get" in dummy_handler.message
    assert result == {"error": "ActionRecommenderAgent::recommend: Failed to get"}
