from typing import TypedDict, Annotated

class AgentState(TypedDict):
    message_history: Annotated[list[str], "History of messages exchanged between agents"]
    node_connections: Annotated[dict[str, list[str]], "Connections between nodes in the workflow"]

# Example of defining the Researcher -> ScaleDown -> Critic workflow
workflow = AgentState(
    message_history=[],
    node_connections={
        "Researcher": ["ScaleDown"],
        "ScaleDown": ["Critic"],
        "Critic": []
    }
)