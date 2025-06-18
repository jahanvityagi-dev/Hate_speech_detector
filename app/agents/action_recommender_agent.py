from app.agents.error_handler_agent import ErrorHandlerAgent
class ActionRecommenderAgent:
    """
    Maps classification labels to moderation actions.
    """
    error_handler = ErrorHandlerAgent()

    def __init__(self):
      

        self.action_map = {
            "Hate": {
                "action": "Ban",
                "reason": "Content promotes hate or dehumanization, which violates platform policy."
            },
            "Toxic": {
                "action": "Warn",
                "reason": "Content is toxic in tone but not directly hateful."
            },
            "Offensive": {
                "action": "Flag",
                "reason": "Content is offensive and should be reviewed by a moderator."
            },
            "Neutral": {
                "action": "Allow",
                "reason": "No harmful content detected."
            },
            "Ambiguous": {
                "action": "Manual Review",
                "reason": "Content is unclear; requires human evaluation."
            }
            
        }
        
    @error_handler.handle_errors(agent_name="ActionRecommenderAgent", method="recommend")
    def recommend(self, label: str) -> dict:
        """
        Returns recommended action and explanation for a classification label.
        """
       
        
        return self.action_map.get(label, {
            "action": "Manual Review",
            "reason": "Unknown label; escalate to a moderator."
        })
    