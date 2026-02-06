import os
from dotenv import load_dotenv
from core.orchestrator import create_research_team

# Load API Keys
load_dotenv()

def main():
    topic = input("Enter research topic: ")
    
    # Initialize the LangGraph
    app = create_research_team()
    
    # Run the graph
    inputs = {"research_topic": topic, "iteration_count": 0, "messages": []}
    
    print(f"\n--- Starting Research Team for: {topic} ---\n")
    
    for output in app.stream(inputs):
        # This will print updates as agents finish their work
        for node_name, state_update in output.items():
            print(f"\n[NODE COMPLETED]: {node_name}")
            if "messages" in state_update:
                last_msg = state_update["messages"][-1]
                print(f"Confidence: {last_msg.get('confidence', 'N/A')}")

if __name__ == "__main__":
    main()