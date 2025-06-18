from app.services.moderation_pipeline import ModerationPipeline
import json

pipeline = ModerationPipeline()

input_text = "People from XYZ are a threat to the nation."
result = pipeline.run(input_text)

print(json.dumps(result, indent=2))
