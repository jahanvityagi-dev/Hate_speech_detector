# from app.agents.policy_reasoning_agent import PolicyReasoningAgent

# def test_generate_explanation_format():
#     agent = PolicyReasoningAgent()
#     explanation = agent.generate_explanation(
#         input_text="XYZ people are criminals.",
#         label="Hate",
#         policy_snippets=[{
#             "text": "Platform prohibits hate based on nationality.",
#             "source_file": "meta_policy.txt",
#             "score": 0.9
#         }]
#     )
#     assert isinstance(explanation, str)
#     assert "hate" in explanation.lower()
import json
import pytest
from fastapi import HTTPException
import app.agents.policy_reasoning_agent as pr_module
from app.agents.policy_reasoning_agent import PolicyReasoningAgent

# Use DummyOpenAIClient from conftest via dummy_openai_client_factory fixture
def test_generate_explanation_success(monkeypatch, dummy_openai_client_factory):
    """generate_explanation should return the response content (stripped) from the OpenAI API."""
    # Dummy content returned by OpenAI (trailing whitespace to test strip)
    fake_explanation = "This is a policy reasoning explanation.  "
    dummy_client = dummy_openai_client_factory(fake_explanation)
    monkeypatch.setattr(pr_module, "client", dummy_client)
    agent = PolicyReasoningAgent(model="dummy-model")
    # Provide an input, label, and some policy snippets
    input_text = "Some user content."
    label = "Toxic"
    snippets = [
        {"text": "Policy text snippet 1", "source_file": "policy1.txt"},
        {"text": "Policy text snippet 2", "source_file": "policy2.txt"}
    ]
    result = agent.generate_explanation(input_text, label, snippets)
    # The result should match the dummy explanation (stripped of trailing spaces)
    assert result == "This is a policy reasoning explanation.", "Explanation output mismatch"
    # Verify the prompt was formed correctly by intercepting the call arguments
    captured = {}
    def dummy_create(model=None, messages=None, temperature=None):
        captured["model"] = model
        captured["messages"] = messages
        captured["temperature"] = temperature
        # Return the same dummy response as above
        return dummy_client._response
    monkeypatch.setattr(dummy_client, "create", dummy_create)
    agent.generate_explanation(input_text, label, snippets)
    prompt = captured["messages"][0]["content"]
    # The prompt should contain the input text, the label, and both policy snippets
    assert "Text:\n\"\"\"Some user content.\"\"\"" in prompt
    assert f"Label: {label}" in prompt
    assert "- Policy text snippet 1 (Source: policy1.txt)" in prompt
    assert "- Policy text snippet 2 (Source: policy2.txt)" in prompt

def test_generate_explanation_api_error(monkeypatch, dummy_error_handler):
    """If the OpenAI API call fails, generate_explanation should use the error handler and return its output."""
    # Monkeypatch the client.create to raise an exception
    class FailingClient:
        def __init__(self):
            self.chat = self
            self.completions = self
        def create(self, **kwargs):
            raise RuntimeError("OpenAI API error")
    monkeypatch.setattr(pr_module, "client", FailingClient())
    agent = PolicyReasoningAgent(model="dummy-model")
    agent.error_handler = dummy_error_handler
    result = agent.generate_explanation("text", "Neutral", [])
    # The dummy error handler should have been called and returned an error dict
    assert dummy_error_handler.calls, "Error handler was not invoked on API failure"
    args = dummy_error_handler.calls[0]
    # In PolicyReasoningAgent, error_handler is called with three args ("PolicyReasoningAgent", "generate_explanation", exception)
    # Our DummyErrorHandlerAgent will format this into an error string
    assert args[0] == "PolicyReasoningAgent" or args[1] == "generate_explanation"
    assert isinstance(args[-1], Exception) and "OpenAI API error" in str(args[-1])
    # The result should contain the error message from dummy_error_handler
    assert result.get("error"), "No error output returned on API exception"
    assert "OpenAI API error" in result["error"]
