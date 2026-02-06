from typing import TypedDict, Annotated, Any
from langgraph.graph import StateGraph, END
from langgraph.types import Send
import operator
from datetime import datetime

class Message(TypedDict):
    """Individual message structure"""
    sender: str
    content: str
    timestamp: str
    confidence: float
    sources: list[str]

class AgentState(TypedDict):
    """Main state passed between agents in the workflow"""
    messages: Annotated[list[Message], operator.add]  # Accumulate messages
    context: str  # Current accumulated research context
    compressed_summary: str  # Output from ScaleDown node
    confidence_scores: dict[str, float]  # {agent_name: score}
    iteration_count: int
    research_topic: str
    message_queue: list[dict]  # Centralized handoff tracking
    compression_metadata: dict
    consensus_reached: bool
    current_node: str

def create_research_graph():
    """
    Build the LangGraph StateGraph with workflow:
    Researcher -> ScaleDown -> Critic -> Consensus -> (Loop or Synthesizer)
    """
    graph = StateGraph(AgentState)
    
    # Define nodes (to be implemented in respective agent files)
    # graph.add_node("researcher", researcher_node)
    # graph.add_node("scaledown", compress_context_node)
    # graph.add_node("critic", critic_node)
    # graph.add_node("consensus", consensus_check_node)
    
    # Define edges
    # graph.add_edge("researcher", "scaledown")
    # graph.add_edge("scaledown", "critic")
    # graph.add_conditional_edges(
    #     "consensus",
    #     consensus_check_node,
    #     {
    #         "revise": "researcher",
    #         "proceed": "synthesizer"
    #     }
    # )
    
    # graph.set_entry_point("researcher")
    # graph.set_finish_point("writer")
    
    return graph

# Initialize the graph
# workflow = create_research_graph()
# app = workflow.compile()