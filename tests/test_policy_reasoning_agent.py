from app.agents.policy_reasoning_agent import PolicyReasoningAgent

def test_generate_explanation_format():
    agent = PolicyReasoningAgent()
    explanation = agent.generate_explanation(
        input_text="XYZ people are criminals.",
        label="Hate",
        policy_snippets=[{
            "text": "Platform prohibits hate based on nationality.",
            "source_file": "meta_policy.txt",
            "score": 0.9
        }]
    )
    assert isinstance(explanation, str)
    assert "hate" in explanation.lower()
