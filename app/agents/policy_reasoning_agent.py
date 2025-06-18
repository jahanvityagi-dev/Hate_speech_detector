import os
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

from app.agents.error_handler_agent import ErrorHandlerAgent

class PolicyReasoningAgent:
    """
    Uses Azure OpenAI (ChatGPT) to generate a reasoning explanation based on the input text, its label, and relevant policies.
    """
    error_handler = ErrorHandlerAgent()  # for decorator usage

    def __init__(self, deployment: str = None):
        # Determine which Azure OpenAI deployment to use (defaults to env setting)
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        # Initialize Azure OpenAI chat model using environment vars (API key & endpoint)
        self.llm = AzureChatOpenAI(
            azure_deployment=self.deployment,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
            temperature=0.3
        )
        self.error_handler = ErrorHandlerAgent()

    @error_handler.handle_errors(agent_name="PolicyReasoningAgent", method="generate_explanation")
    def generate_explanation(self, input_text: str, label: str, policy_snippets: list[dict]) -> str:
        """
        Generate a short explanation of why the content was classified with `label`, using the retrieved policy snippets.
        """
        # Combine policy snippets into a formatted context string
        snippets_text = "\n\n".join(
            f"- {chunk['text']} (Source: {chunk['source_file']})" for chunk in policy_snippets
        )
        # Construct the prompt for the Azure OpenAI model
        prompt = (
            "You are a content policy advisor.\n\n"
            "Given the following:\n"
            "1. The user-submitted text\n"
            "2. Its classification label\n"
            "3. Relevant policy definitions\n\n"
            "Generate a brief, professional explanation for why the content was classified this way.\n\n"
            f"Text:\n\"\"\"{input_text}\"\"\"\n\n"
            f"Label: {label}\n\n"
            "Relevant Policies:\n"
            f"{snippets_text}\n\n"
            "Respond with a single paragraph explaining the reasoning."
        )
        # Invoke Azure OpenAI chat model (as a single user message)
        response_msg = self.llm.invoke([HumanMessage(content=prompt)])
        return response_msg.content.strip()
