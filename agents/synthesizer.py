from typing import Dict, Any
from agents.base_agent import BaseAgent

class SynthesizerAgent(BaseAgent):
    """
    Merges research findings and critiques into a unified knowledge base.
    """
    def __init__(self):
        super().__init__(name="Synthesizer", temperature=0.2)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log("Synthesizing final research report...")
        
        all_messages = "\n\n".join([f"{m['sender']}: {m['content']}" for m in state["messages"]])

        prompt = f"""
        You are a Master Synthesizer. Combine the following research and critiques into a single, 
        comprehensive, and logically structured final knowledge document.
        
        HISTORY:
        {all_messages}

        Ensure the final output is professional, cites the sources mentioned in the history, 
        and addresses the gaps raised by the Critic.
        """

        response = self.llm.invoke(prompt)
        
        message = self.format_response(content=response.content, confidence=1.0, sources=[])
        
        return {
            "messages": [message],
            "current_node": "synthesizer"
        }