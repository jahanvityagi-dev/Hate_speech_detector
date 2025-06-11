# from app.agents.hate_speech_agent import HateSpeechDetectionAgent
# from app.agents.hybrid_retriever_agent import HybridRetrieverAgent
# from app.agents.policy_reasoning_agent import PolicyReasoningAgent
# from app.agents.action_recommender_agent import ActionRecommenderAgent
# # 1. User input
# input_text = "People from XYZ are a threat to the nation."

# # 2. Step 1 - Classify
# classifier = HateSpeechDetectionAgent()
# classification = classifier.classify(input_text)
# print("🔖 Classification Result:", classification)

# # 3. Step 2 - Retrieve policies
# retriever = HybridRetrieverAgent()
# retrieved_chunks = retriever.retrieve(input_text)
# print(f"\n📚 Retrieved {len(retrieved_chunks)} policy chunks.")

# # 4. Step 3 - Generate Explanation
# reasoner = PolicyReasoningAgent()
# explanation = reasoner.generate_explanation(
#     input_text=input_text,
#     label=classification["label"],
#     policy_snippets=retrieved_chunks
# )

# print("\n🧾 Justification:\n", explanation)

# recommender = ActionRecommenderAgent()
# recommendation = recommender.recommend(classification["label"])

# print("\n🚦 Moderation Action:")
# print("Action:", recommendation["action"])
# print("Reason:", recommendation["reason"])
from app.services.moderation_pipeline import ModerationPipeline
import json

pipeline = ModerationPipeline()

input_text = "People from XYZ are a threat to the nation."
result = pipeline.run(input_text)

print(json.dumps(result, indent=2))
