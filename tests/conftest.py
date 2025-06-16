import io
import json
import numpy as np
import pytest

class DummyOpenAIResponse:
    """Dummy response object mimicking AzureOpenAI chat completion response."""
    def __init__(self, content: str):
        # Simulate the nested response structure: choices -> message -> content
        message = type("Message", (), {"content": content})
        choice = type("Choice", (), {"message": message})
        self.choices = [choice]

class DummyOpenAIClient:
    """Dummy AzureOpenAI client with a chat.completions.create method."""
    def __init__(self, response: DummyOpenAIResponse):
        self._response = response
        self.chat = self  # .chat and .completions both reference self for simplicity
        self.completions = self

    def create(self, **kwargs):
        # Return the predefined dummy response regardless of input
        return self._response

class DummyErrorHandlerAgent:
    """Dummy ErrorHandlerAgent to capture handle_error calls without raising exceptions."""
    def __init__(self):
        self.calls = []  # Store calls for inspection

    def handle_error(self, *args):
        """Capture calls and return a dummy error structure instead of raising."""
        self.calls.append(args)
        # Determine how to format the error output based on arguments
        if len(args) == 1:
            # Single argument (likely an Exception object)
            e = args[0]
            return {"error": str(e)}
        elif len(args) == 2:
            # Two arguments: assume (context, error_message)
            context, message = args
            return {"error": f"{context}: {message}"}
        elif len(args) == 3:
            # Three arguments: could be (error, agent_name, method) or (agent_name, method, error)
            if isinstance(args[0], Exception):
                error, agent_name, method = args
            else:
                agent_name, method, error = args
            return {"error": f"[{agent_name}] {method}: {str(error)}"}
        # If unexpected usage, just return generic error string
        return {"error": "Unknown error format"}

@pytest.fixture
def dummy_openai_client_factory():
    """
    Fixture to create a dummy AzureOpenAI client for a given response content.
    Usage: dummy_client = dummy_openai_client_factory(content_str)
    Then monkeypatch the target module's `client` with this dummy_client.
    """
    def _make_client(content: str):
        response = DummyOpenAIResponse(content)
        return DummyOpenAIClient(response)
    return _make_client

@pytest.fixture
def dummy_error_handler():
    """
    Fixture providing a DummyErrorHandlerAgent instance to inject into agents.
    """
    return DummyErrorHandlerAgent()

@pytest.fixture
def dummy_whisper_model(monkeypatch):
    """
    Fixture to replace whisper.load_model with a dummy model 
    that returns a fixed transcription result.
    """
    # Define dummy Whisper model class
    class DummyWhisperModel:
        def transcribe(self, audio_path: str):
            return {"text": "dummy transcription result"}  # Simulated transcription text

    # Monkeypatch the whisper.load_model to return our dummy model
    import app.services.audio_transcript_service as ats_module
    monkeypatch.setattr(ats_module.whisper, "load_model", lambda model_name="base": DummyWhisperModel())
    return DummyWhisperModel  # Not strictly used by tests, but returned for completeness

