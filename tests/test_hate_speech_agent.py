from app.agents.hate_speech_agent import HateSpeechDetectionAgent

def test_classification_output_structure():
    agent = HateSpeechDetectionAgent()
    result = agent.classify("I hate all XYZ people.")
    assert "label" in result
    assert "explanation" in result
    assert isinstance(result["label"], str)
    assert isinstance(result["explanation"], str)
