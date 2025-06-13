import io
import csv
from core.models import ModerationResult
def _generate_export_csv(result: ModerationResult, transcription: str = None) -> str:
        """Generate CSV string from moderation result."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            "Input Text", "Transcription", "Label", "Classification Reason",
            "Explanation", "Action", "Action Reason", "Policy Snippets"
        ])

        # Combine policy snippet texts
        policy_texts = " | ".join([p.get("text", "") for p in result.policy_snippets])

        # Write data row
        writer.writerow([
            result.input_text,
            transcription or "",
            result.label,
            result.classification_reason,
            result.explanation,
            result.action,
            result.action_reason,
            policy_texts
        ])

        return output.getvalue()
