import os
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from typing import Any, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all research agents.
    Provides LLM initialization, standardized logging, and response formatting.
    """
    
    def __init__(self, name: str, temperature: float = 0.2):
        self.name = name
        self.temperature = temperature
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM based on environment variables"""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        if provider == "anthropic":
            self.log("Initializing Anthropic Claude-3")
            return ChatAnthropic(
                model="claude-3-opus-20240229",
                temperature=self.temperature
            )
        else:
            self.log("Initializing OpenAI GPT-4-Turbo")
            return ChatOpenAI(
                model="gpt-4-turbo",
                temperature=self.temperature
            )
    
    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent logic. Must be implemented by subclasses."""
        pass
    
    def log(self, message: str):
        """Standardized agent logging"""
        logger.info(f"[{self.name}] {message}")

    def format_response(self, content: str, confidence: float, sources: List[str]) -> Dict[str, Any]:
        """
        Ensures all agents return the standard format required for 
        ScaleDown and the Consensus Engine.
        """
        return {
            "sender": self.name,
            "content": content,
            "confidence": max(0.0, min(1.0, confidence)),
            "sources": sources,
            "timestamp": None # Will be set by orchestrator
        }