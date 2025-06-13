from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModerationResult:
    label: str
    classification_reason: str
    explanation: str
    policy_snippets: List[Dict[str, Any]]
    action: str
    action_reason: str
    input_text: str = ""
