import os
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all research agents"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM from environment variables"""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        if provider == "anthropic":
            return ChatAnthropic(
                model="claude-3-opus-20240229",
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            return ChatOpenAI(
                model="gpt-4-turbo",
                api_key=os.getenv("OPENAI_API_KEY")
            )
    
    @abstractmethod
    def run(self, state: dict) -> dict:
        """Execute agent logic. Must return state update."""
        pass
    
    def format_response(self, data: str, confidence: float, sources: list) -> dict:
        """Standardized response format"""
        return {
            "data": data,
            "confidence": max(0.0, min(1.0, confidence)),
            "sources": sources
        }