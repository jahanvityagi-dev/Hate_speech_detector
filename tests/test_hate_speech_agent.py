
import json
import pytest
from fastapi import HTTPException
import app.agents.hate_speech_agent as hs_module
from app.agents.hate_speech_agent import HateSpeechDetectionAgent

# Reuse DummyOpenAIClient and DummyOpenAIResponse from conftest via the fixture
def test_classify_valid_label(monkeypatch, dummy_openai_client_factory):
    """classify should return the parsed JSON with label and explanation for a valid response."""
    # Prepare dummy OpenAI client to return a valid JSON classification
    response_content = json.dumps({"label": "Hate", "explanation": "Test explanation"})
    dummy_client = dummy_openai_client_factory(response_content)
    monkeypatch.setattr(hs_module, "client", dummy_client)
    agent = HateSpeechDetectionAgent(deployment="dummy-deployment")
    result = agent.classify("Some hateful content")
    # The result should directly be the parsed JSON from the dummy response
    assert result["label"] == "Hate"
    assert result["explanation"] == "Test explanation"

def test_classify_invalid_label(monkeypatch, dummy_openai_client_factory):
    """If the model returns an invalid label, classify should raise HTTPException."""
    # Dummy response with a label not in the allowed LABELS list
    bad_content = json.dumps({"label": "NotValid", "explanation": "Irrelevant"})
    dummy_client = dummy_openai_client_factory(bad_content)
    monkeypatch.setattr(hs_module, "client", dummy_client)
    agent = HateSpeechDetectionAgent(deployment="dummy")
    # Now check that HTTPException is raised
    with pytest.raises(HTTPException) as excinfo:
        agent.classify("Test text")

    exc = excinfo.value
    assert exc.status_code == 500
    assert "Invalid label returned" in str(exc.detail)


def test_classify_response_not_json(monkeypatch, dummy_openai_client_factory):
    """If the model returns non-JSON content, classify should raise an HTTPException via error_handler."""
    # Dummy response content that is not valid JSON (to trigger json.loads exception)
    dummy_client = dummy_openai_client_factory("Not a JSON")
    monkeypatch.setattr(hs_module, "client", dummy_client)
    agent = HateSpeechDetectionAgent(deployment="dummy")
    # Use the real error handler (which should raise HTTPException)
    with pytest.raises(HTTPException) as excinfo:
        agent.classify("text that causes JSON error")
    exc = excinfo.value
    # The HTTPException should indicate an internal error; ensure original JSON error message is mentioned
    assert exc.status_code == 500
    # Because of how handle_error is called with wrong params, detail may contain UnknownAgent or the error text
    detail_str = str(exc.detail)
    # Check that some JSON decode error indication is in the detail (e.g., 'Expecting value')
    assert "Expecting value" in detail_str or "JSONDecodeError" in detail_str, "JSON parse error not reflected in exception detail"

def test_classify_api_exception(monkeypatch):
    """If the OpenAI API call fails, classify should raise HTTPException."""
    # Dummy client whose create raises exception
    class FailingClient:
        def __init__(self):
            self.chat = self
            self.completions = self
        def create(self, **kwargs):
            raise RuntimeError("API failure")

    monkeypatch.setattr(hs_module, "client", FailingClient())
    agent = HateSpeechDetectionAgent(deployment="dummy")

    with pytest.raises(HTTPException) as excinfo:
        agent.classify("some input text")

    exc = excinfo.value
    assert exc.status_code == 500
    assert "API failure" in str(exc.detail)

