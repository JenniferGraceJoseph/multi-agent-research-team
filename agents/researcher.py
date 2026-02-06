import os
import json
from agents.base_agent import BaseAgent
from langchain_community.tools.tavily_search import TavilySearchResults
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearcherAgent(BaseAgent):
    """Research agent using Tavily Search for real-time data gathering"""
    
    def __init__(self):
        super().__init__("Researcher")
        self.search_tool = TavilySearchResults(
            max_results=5,
            api_key=os.getenv("TAVILY_API_KEY")
        )
    
    def run(self, state: dict) -> dict:
        """
        Execute researcher logic:
        1. Extract research topic from state
        2. Perform multi-keyword searches
        3. Aggregate and score results
        4. Return JSON: {data, confidence, sources}
        """
        
        topic = state.get("research_topic", "")
        if not topic:
            return state
        
        try:
            # Perform search
            search_results = self.search_tool.invoke(topic)
            
            # Extract content and sources
            findings = []
            sources = []
            
            for result in search_results:
                content = result.get("content", "")
                url = result.get("url", "")
                
                if content:
                    findings.append(content)
                if url:
                    sources.append(url)
            
            research_data = " ".join(findings)
            
            # Calculate confidence based on result count and source diversity
            confidence = min(0.95, len(search_results) / 10.0 + 0.5)
            
            # Create message
            message = {
                "sender": self.agent_name,
                "content": research_data,
                "timestamp": "",
                "confidence": confidence,
                "sources": sources
            }
            
            # Update state
            new_state = {
                **state,
                "messages": state.get("messages", []) + [message],
                "confidence_scores": {
                    **state.get("confidence_scores", {}),
                    self.agent_name: confidence
                },
                "context": research_data
            }
            
            logger.info(f"Researcher found {len(sources)} sources with confidence {confidence:.2f}")
            return new_state
            
        except Exception as e:
            logger.error(f"Researcher error: {str(e)}")
            return state