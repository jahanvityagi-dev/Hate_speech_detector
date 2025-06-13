import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from app.agents.error_handler_agent import ErrorHandlerAgent

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

LABELS = ["Hate", "Toxic", "Offensive", "Neutral", "Ambiguous"]

class HateSpeechDetectionAgent:
    def __init__(self, deployment=None):
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.error_handler = ErrorHandlerAgent()

    def classify(self, input_text: str) -> dict:
        prompt = f"""
You are a content moderation assistant. Classify the following text into one of the following categories:
- Hate
- Toxic
- Offensive
- Neutral
- Ambiguous

Also provide a short explanation for your classification.

Text:
\"\"\"{input_text}\"\"\"

Return your output in this JSON format:
{{
  "label": "<one of: Hate, Toxic, Offensive, Neutral, Ambiguous>",
  "explanation": "<brief reason>"
}}
"""
        try:
            response = client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            result = json.loads(content)

            if result["label"] not in LABELS:
                raise ValueError("Invalid label returned")

            return result

        except Exception as e:
            # return {
            #     "label": "Ambiguous",
            #     "explanation": f"Error: {str(e)}"
            # }
            return self.error_handler.handle_error("HateSpeechAgent", str(e))
