from app.agents.hate_speech_agent import HateSpeechDetectionAgent
from app.agents.hybrid_retriever_agent import HybridRetrieverAgent
from app.agents.policy_reasoning_agent import PolicyReasoningAgent
from app.agents.action_recommender_agent import ActionRecommenderAgent

class ModerationPipeline:
    """
    Orchestrates the full moderation process:
    1. Classify input
    2. Retrieve policies
    3. Generate reasoning
    4. Recommend action
    """

    def __init__(self):
        self.classifier = HateSpeechDetectionAgent()
        self.retriever = HybridRetrieverAgent()
        self.reasoner = PolicyReasoningAgent()
        self.recommender = ActionRecommenderAgent()

    def run(self, input_text: str) -> dict:
        """
        Executes the full moderation flow and returns structured result.
        """
        classification = self.classifier.classify(input_text)
        label = classification["label"]

        policy_chunks = self.retriever.retrieve(input_text)

        explanation = self.reasoner.generate_explanation(
            input_text=input_text,
            label=label,
            policy_snippets=policy_chunks
        )

        action_info = self.recommender.recommend(label)

        return {
            "input": input_text,
            "label": label,
            "classification_reason": classification["explanation"],
            "policy_snippets": policy_chunks,
            "explanation": explanation,
            "action": action_info["action"],
            "action_reason": action_info["reason"]
        }
