# from app.agents.hate_speech_agent import HateSpeechDetectionAgent

# def test_classification_output_structure():
#     agent = HateSpeechDetectionAgent()
#     result = agent.classify("I hate all XYZ people.")
#     assert "label" in result
#     assert "explanation" in result
#     assert isinstance(result["label"], str)
#     assert isinstance(result["explanation"], str)
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

def test_classify_invalid_label(monkeypatch, dummy_openai_client_factory, dummy_error_handler):
    """If the model returns an invalid label, classify should call error_handler and return its output."""
    # Dummy response with a label not in the allowed LABELS list
    bad_content = json.dumps({"label": "NotValid", "explanation": "Irrelevant"})
    dummy_client = dummy_openai_client_factory(bad_content)
    monkeypatch.setattr(hs_module, "client", dummy_client)
    agent = HateSpeechDetectionAgent(deployment="dummy")
    # Inject dummy error handler to capture error handling without raising
    agent.error_handler = dummy_error_handler
    result = agent.classify("Test text")
    # It should have called error_handler.handle_error due to invalid label
    assert dummy_error_handler.calls, "Error handler was not called for invalid label"
    call_args = dummy_error_handler.calls[0]
    # The HateSpeechDetectionAgent code calls handle_error("HateSpeechAgent", error_message)
    assert call_args[0] == "HateSpeechAgent"
    assert "Invalid label returned" in call_args[1]
    # The result should be the dummy error dict returned by DummyErrorHandlerAgent
    assert result.get("error"), "No error output returned for invalid label"
    assert "Invalid label returned" in result["error"]

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

def test_classify_api_exception(monkeypatch, dummy_error_handler):
    """If the OpenAI API call fails, classify should use error_handler and return an error structure."""
    # Create a dummy client whose create() method raises an exception to simulate API failure
    class FailingClient:
        def __init__(self):
            self.chat = self
            self.completions = self
        def create(self, **kwargs):
            raise RuntimeError("API failure")
    monkeypatch.setattr(hs_module, "client", FailingClient())
    agent = HateSpeechDetectionAgent(deployment="dummy")
    agent.error_handler = dummy_error_handler  # inject dummy handler to capture output
    result = agent.classify("some input text")
    # Ensure the dummy error handler was invoked due to the exception
    assert dummy_error_handler.calls, "Error handler not called on API exception"
    call_args = dummy_error_handler.calls[0]
    assert call_args[0] == "HateSpeechAgent"  # context passed to error handler
    assert "API failure" in call_args[1]      # error message passed to error handler
    # The result should be the dummy error dictionary
    assert result.get("error") is not None
    assert "API failure" in result["error"]
