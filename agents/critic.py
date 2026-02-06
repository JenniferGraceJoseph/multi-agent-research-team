from typing import Dict, Any
from agents.base_agent import BaseAgent
from datetime import datetime

class CriticAgent(BaseAgent):
    """
    Evaluates research quality.
    Identifies gaps and provides a consensus score.
    """
    def __init__(self):
        super().__init__(name="Critic", temperature=0.1)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Get the latest research (either from messages or compressed summary)
        latest_research = state.get("compressed_summary") or state["messages"][-1]["content"]
        
        self.log("Critiquing latest research findings...")

        prompt = f"""
        You are a Peer Reviewer. Evaluate the following research for:
        1. Logical gaps or missing information.
        2. Potential factual inaccuracies.
        3. Clarity and depth.

        RESEARCH TO EVALUATE:
        {latest_research}

        Provide your critique and a Confidence Score between 0.0 and 1.0.
        If the research is excellent, score it > 0.8. 
        If it needs more data, score it < 0.5.

        Format:
        Critique: [Your feedback]
        Score: [0.0-1.0]
        """

        response = self.llm.invoke(prompt)
        content = response.content
        
        # Extract score
        score = 0.5
        if "Score:" in content:
            try:
                score = float(content.split("Score:")[1].strip()[:3])
            except:
                pass

        message = self.format_response(content=content, confidence=score, sources=[])
        message["timestamp"] = datetime.utcnow().isoformat()

        # Update the confidence_scores dict for the Orchestrator to see
        new_scores = state.get("confidence_scores", {}).copy()
        new_scores["Critic"] = score

        return {
            "messages": [message],
            "confidence_scores": new_scores,
            "current_node": "critic"
        }