import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from app.agents.error_handler_agent import ErrorHandlerAgent

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

class PolicyReasoningAgent:
    """
    This agent uses OpenAI to justify a classification decision
    based on the input and matching policy texts.
    """
    error_handler = ErrorHandlerAgent()

    def __init__(self, model=None, deployment=None):
        self.model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.error_handler = ErrorHandlerAgent()

    @error_handler.handle_errors(agent_name="PolicyReasoningAgent", method="generate_explanation")
    def generate_explanation(self, input_text: str, label: str, policy_snippets: list[dict]) -> str:
        """
        Combine input + label + retrieved policy to generate reasoning using OpenAI.
        """
        snippets = "\n\n".join([f"- {chunk['text']} (Source: {chunk['source_file']})" for chunk in policy_snippets])

        prompt = f"""
You are a content policy advisor.

Given the following:
1. The user-submitted text
2. Its classification label
3. Retrieved policy definitions

Generate a short, professional explanation of why the content was classified this way.

Text:
\"\"\"{input_text}\"\"\"

Label: {label}

Relevant Policies:
{snippets}

Respond with a paragraph explaining the reasoning.
"""

        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

        # except Exception as e:
            # return self.error_handler.handle_error("PolicyReasoningAgent", "generate_explanation", e)
