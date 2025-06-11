from pydantic import BaseModel, Field
from typing import List

class ModerationRequest(BaseModel):
    text: str 

class PolicySnippet(BaseModel):
    text: str
    source_file: str
    score: float

class ModerationResponse(BaseModel):
    input: str
    label: str
    classification_reason: str
    policy_snippets: List[PolicySnippet]
    explanation: str
    action: str
    action_reason: str
