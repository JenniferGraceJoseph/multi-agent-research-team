import os
import json
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Import AgentState
from core.orchestrator import AgentState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compress_context_node(state: AgentState) -> dict:
    """
    LangGraph node: Compress the last 3 messages into a concise factual summary.

    Strategy:
    1. Extract last 3 messages
    2. Use LLM to extract only core factual claims and data
    3. Remove conversational filler
    4. Target ~70% token reduction
    """

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.3,
            api_key=api_key
        )

        messages = state.get("messages", [])
        last_messages = messages[-3:] if len(messages) >= 3 else messages

        if not last_messages:
            logger.warning("No messages available for compression")
            return {
                **state,
                "compressed_summary": "",
                "messages": messages
            }

        message_text = "\n\n".join(
            f"Agent ({msg.get('sender', 'unknown')}): {msg.get('content', '')}"
            for msg in last_messages
        )

        compression_prompt = ChatPromptTemplate.from_template(
            """You are a research data compression specialist.

EXTRACT ONLY:
- Core factual claims
- Research findings
- Numerical data and statistics
- Source citations and URLs

REMOVE:
- Conversational filler
- Greetings and meta-commentary
- Confidence scores

RULES:
- Preserve factual accuracy
- Use bullet points
- Target ~70% reduction

Return output as JSON:
{{
  "key_findings": [],
  "data_points": [],
  "sources": [],
  "compressed_text": ""
}}

INPUT:
{messages}
"""
        )

        chain = compression_prompt | llm
        response = chain.invoke({"messages": message_text})

        try:
            response_text = response.content
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            compressed_data = json.loads(response_text)
            compressed_summary = compressed_data.get("compressed_text", "")

            logger.info(
                f"Compression complete | Original: {len(message_text)} chars | "
                f"Compressed: {len(compressed_summary)} chars"
            )

        except json.JSONDecodeError:
            logger.warning("JSON parsing failed, using raw response")
            compressed_summary = response.content

        preserved_messages = messages[:-3] if len(messages) > 3 else []

        compressed_message = {
            "sender": "ScaleDown",
            "content": compressed_summary,
            "type": "compressed_summary"
        }

        return {
            **state,
            "messages": preserved_messages + [compressed_message],
            "compressed_summary": compressed_summary,
            "compression_metadata": {
                "original_length": len(message_text),
                "compressed_length": len(compressed_summary),
                "ratio": (
                    (1 - len(compressed_summary) / len(message_text)) * 100
                    if message_text else 0
                ),
                "messages_processed": len(last_messages)
            }
        }

    except Exception as e:
        logger.error(f"ScaleDown compression failed: {str(e)}")
        return {
            **state,
            "compressed_summary": "",
            "compression_error": str(e)
        }
