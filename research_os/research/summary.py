"""research/summary.py - RAG-based initial summary generation."""
import os
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from prompts.summary_prompts import DEEP_RESEARCH_PROMPT
from utils.logging import setup_logger

logger = setup_logger("Summary")


def generate_summary(query: str, evidence: str) -> str:
    """
    Generate an initial deep research briefing grounded in retrieved evidence chunks.
    Uses local Qwen3 via Ollama.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:latest")

    llm = ChatOllama(base_url=base_url, model=chat_model, temperature=0)
    chain = DEEP_RESEARCH_PROMPT | llm | StrOutputParser()

    logger.info("Generating initial deep research briefing...")
    try:
        result = chain.invoke({"query": query, "evidence": evidence})
        return result
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return f"ERROR: Summary generation failed — {e}"
