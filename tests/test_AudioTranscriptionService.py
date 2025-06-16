import os
import pytest
from app.services.audio_transcript_service import AudioTranscriptionService

class DummyWhisperModel:
    """Dummy Whisper model to simulate transcription without actual model."""
    def transcribe(self, audio_path: str):
        return {"text": "Transcribed audio text"}

def test_transcribe_success(monkeypatch, tmp_path, dummy_whisper_model):
    """transcribe should return the transcription text when the file exists."""
    # Create a dummy audio file path
    dummy_audio = tmp_path / "audio.mp3"
    dummy_audio.write_bytes(b"dummy audio content")
    # Ensure os.path.exists returns True for this path
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    # The dummy_whisper_model fixture already patched whisper.load_model to DummyWhisperModel
    service = AudioTranscriptionService(model_name="base")
    # Also monkeypatch the model instance to our DummyWhisperModel (to ensure transcribe is dummy)
    service.model = DummyWhisperModel()
    result_text = service.transcribe(str(dummy_audio))
    assert result_text == "Transcribed audio text", "Transcription text did not match expected dummy text"

def test_transcribe_file_not_found(monkeypatch, dummy_whisper_model):
    """transcribe should raise FileNotFoundError if the audio file path does not exist."""
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    service = AudioTranscriptionService(model_name="base")
    with pytest.raises(FileNotFoundError) as excinfo:
        service.transcribe("nonexistent_file.mp3")
    exc = excinfo.value
    assert "Audio file not found" in str(exc), "Expected FileNotFoundError for missing audio file"
