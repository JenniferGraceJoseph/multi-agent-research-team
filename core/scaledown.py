import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compress_context_node(state: dict) -> dict:
    """
    LangGraph node: Compress last 3 messages into factual summary.
    
    Execution:
    1. Extract last 3 messages
    2. Use ChatOpenAI to extract core factual claims
    3. Target 70% token reduction
    4. Replace message history with compressed version
    
    Returns:
        Updated state with compressed_summary and trimmed messages
    """
    
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Get last 3 messages
    messages = state.get("messages", [])
    last_messages = messages[-3:] if len(messages) >= 3 else messages
    
    if not last_messages:
        return {**state, "compressed_summary": "", "messages": messages}
    
    # Format messages
    message_text = "\n\n".join([
        f"{msg.get('sender', 'unknown')}: {msg.get('content', '')}"
        for msg in last_messages
    ])
    
    # Create compression prompt
    prompt = ChatPromptTemplate.from_template("""You are a research data compression specialist.

EXTRACT ONLY:
- Core factual claims
- Research data and findings
- Source citations and URLs
- Numerical data and statistics

REMOVE:
- Conversational filler
- Meta-commentary
- Confidence assessments

TARGET: 70% reduction in length

INPUT MESSAGES:
{messages}

OUTPUT (JSON):
{{
    "key_findings": ["finding1", "finding2"],
    "data_points": ["data1", "data2"],
    "sources": ["url1", "url2"],
    "compressed_text": "Brief summary"
}}""")
    
    chain = prompt | llm
    response = chain.invoke({"messages": message_text})
    
    try:
        json_str = response.content
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        compressed_data = json.loads(json_str)
        compressed_summary = compressed_data.get("compressed_text", "")
    except:
        compressed_summary = response.content
    
    # Create new state with compressed messages
    preserved_messages = messages[:-3] if len(messages) > 3 else []
    compressed_message = {
        "sender": "ScaleDown",
        "content": compressed_summary,
        "timestamp": "",
        "confidence": 1.0,
        "sources": []
    }
    
    return {
        **state,
        "messages": preserved_messages + [compressed_message],
        "compressed_summary": compressed_summary,
        "compression_metadata": {
            "original_length": len(message_text),
            "compressed_length": len(compressed_summary),
            "ratio": (1 - len(compressed_summary) / len(message_text)) * 100,
            "messages_processed": len(last_messages)
        }
    }