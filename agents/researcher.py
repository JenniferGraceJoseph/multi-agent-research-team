from typing import Dict, Any
from agents.base_agent import BaseAgent
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime
import json

class ResearcherAgent(BaseAgent):
    """
    Primary research agent.
    Gathers factual information using real-time web search.
    """

    def __init__(self):
        super().__init__(name="Researcher", temperature=0.2)
        # Initialize the search tool
        self.search_tool = TavilySearchResults(k=5) # Returns top 5 results

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        topic = state["research_topic"]
        self.log(f"Conducting web research on: {topic}")

        # 1. Execute Real-World Search
        search_results = self.search_tool.invoke({"query": topic})
        
        # 2. Extract URLs for Source Attribution
        sources = [res.get("url") for res in search_results]

        # 3. Use LLM to synthesize search results into a report
        prompt = f"""
        You are a Senior Researcher. Synthesize the following raw search results into a concise factual report.
        
        Topic: {topic}
        Search Results: {json.dumps(search_results)}

        Rules:
        - Use only the data provided in search results.
        - List key facts, statistics, and technical details.
        - If the results are contradictory, note the discrepancy.
        - Assign a confidence score (0.0 to 1.0) based on the quantity and quality of sources.

        Output Format:
        Report: [Your Synthesis]
        Confidence: [0.0-1.0]
        """

        response = self.llm.invoke(prompt)
        
        # Parsing confidence from LLM output (simple extraction)
        content = response.content
        confidence = 0.8 # Default fallback
        if "Confidence:" in content:
            try:
                confidence = float(content.split("Confidence:")[1].strip()[:3])
            except:
                pass

        # 4. Format the final message for the Orchestrator
        message = self.format_response(
            content=content,
            confidence=confidence,
            sources=sources
        )
        message["timestamp"] = datetime.utcnow().isoformat()

        return {
            "messages": [message],
            "iteration_count": state.get("iteration_count", 0) + 1,
            "current_node": "researcher"
        }