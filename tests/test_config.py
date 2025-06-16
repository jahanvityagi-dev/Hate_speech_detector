import os
import importlib
import pytest
import sys

@pytest.fixture(autouse=True)
def reset_env_and_reload():
    """Reload config after env change to reflect updates."""
    yield
    if "app.config" in sys.modules:
        importlib.reload(sys.modules["app.config"])


def test_openai_api_key_set(monkeypatch):
    """Should read OPENAI_API_KEY from environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    # Force reload after setting env
    import app.config as config
    importlib.reload(config)

    assert config.OPENAI_API_KEY == "fake-key"
