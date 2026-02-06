import os
import json
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AgentState from orchestrator
from core.orchestrator import AgentState

def compress_context_node(state: AgentState) -> dict:
    """
    Compress the last 3 messages from state into a concise summary.
    
    Strategy:
    1. Extract last 3 messages from message history
    2. Use LLM to extract only core factual claims and research data
    3. Remove conversational filler and redundancy
    4. Replace message history with compressed summary
    5. Target 70% token reduction
    
    Args:
        state: Current AgentState containing messages and context
        
    Returns:
        Updated state dict with 'compressed_summary' field and trimmed messages
    """
    
    try:
        # Initialize ChatOpenAI with API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.3,  # Low temperature for factual extraction
            api_key=api_key
        )
        
        # Extract last 3 messages
        messages = state.get("messages", [])
        last_messages = messages[-3:] if len(messages) >= 3 else messages
        
        if not last_messages:
            logger.warning("No messages to compress")
            return {
                **state,
                "compressed_summary": "",
                "messages": messages  # Keep original if nothing to compress
            }
        
        # Format messages for compression
        message_text = "\n\n".join([
            f"Agent ({msg.get('sender', 'unknown')}): {msg.get('content', '')}"
            for msg in last_messages
        ])
        
        # Create compression prompt
        compression_prompt = ChatPromptTemplate.from_template(
            """You are a research data compression specialist. Your task is to extract ONLY the core factual claims and research data from the following agent messages.

CRITICAL RULES:
1. Extract only factual claims, research findings, and data points
2. Remove ALL conversational filler, greetings, and meta-commentary
3. Preserve all source citations and URLs
4. Maintain numerical data and statistics exactly as stated
5. Remove confidence assessments (we handle those separately)
6. Use bullet points for clarity
7. Target 70% reduction in length while keeping all critical information
8. Format output as structured key findings

INPUT MESSAGES:
{messages}

OUTPUT FORMAT:
Return a JSON object with this structure:
{{
    "key_findings": ["finding1", "finding2", ...],
    "data_points": ["data1", "data2", ...],
    "sources": ["source1", "source2", ...],
    "compressed_text": "A brief paragraph summary"
}}

Extract and compress the above messages now:"""
        )
        
        # Run compression
        chain = compression_prompt | llm
        response = chain.invoke({"messages": message_text})
        
        # Parse LLM response
        try:
            # Extract JSON from response
            response_text = response.content
            
            # Try to parse as JSON
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```)[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```)[1].split("```)[0].strip()
            else:
                json_str = response_text
            
            compressed_data = json.loads(json_str)
            compressed_summary = compressed_data.get("compressed_text", "")
            
            logger.info(f"Compression successful. Original: {len(message_text)} chars, Compressed: {len(compressed_summary)} chars")
            logger.info(f"Compression ratio: {(1 - len(compressed_summary) / len(message_text)) * 100:.1f}%")
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response, using raw content")
            compressed_summary = response.content
        
        # Create new messages list with compressed summary
        # Keep all messages except last 3, add compressed version
        preserved_messages = messages[:-3] if len(messages) > 3 else []
        
        compressed_message = {
            "sender": "ScaleDown",
            "content": compressed_summary,
            "type": "compressed_summary",
            "original_message_count": len(last_messages),
            "compression_ratio": (1 - len(compressed_summary) / len(message_text)) * 100 if message_text else 0
        }
        
        new_messages = preserved_messages + [compressed_message]
        
        # Return updated state
        return {
            **state,
            "messages": new_messages,
            "compressed_summary": compressed_summary,
            "compression_metadata": {
                "original_length": len(message_text),
                "compressed_length": len(compressed_summary),
                "ratio": (1 - len(compressed_summary) / len(message_text)) * 100 if message_text else 0,
                "messages_processed": len(last_messages)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in compress_context_node: {str(e)}")
        # Graceful fallback: return state with minimal compression
        return {
            **state,
            "compressed_summary": "",
            "compression_error": str(e)
        }