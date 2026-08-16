"""research/regeneration.py - Feedback-driven summary revision chain."""
import os
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from prompts.regeneration_prompts import REGENERATION_PROMPT
from utils.logging import setup_logger

logger = setup_logger("Regeneration")


def regenerate_summary(
    query: str,
    previous_summary: str,
    evidence: str,
    evaluation: dict,
) -> str:
    """
    Regenerate a research briefing incorporating evaluator feedback.
    Receives the full evaluation dict and formats revision instructions for the LLM.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:latest")

    llm = ChatOllama(base_url=base_url, model=chat_model, temperature=0)
    chain = REGENERATION_PROMPT | llm | StrOutputParser()

    # Format revision instructions into a human-readable block
    revision_instructions = evaluation.get("revision_instructions", [])
    if revision_instructions:
        revision_text = "\n".join(f"- {inst}" for inst in revision_instructions)
    else:
        revision_text = "No specific revision instructions were generated; improve overall quality."

    # Include all issue categories in the evaluation block
    eval_lines = [f"Score: {evaluation.get('score', 'N/A')}/10\n"]
    for category in [
        "unsupported_claims",
        "missing_limitations",
        "missing_quantitative_evidence",
        "redundancy_issues",
        "contradictions",
        "missing_citations",
    ]:
        items = evaluation.get(category, [])
        if items:
            eval_lines.append(f"{category.upper().replace('_', ' ')}:")
            for item in items:
                eval_lines.append(f"  - {item}")
    evaluation_text = "\n".join(eval_lines)

    logger.info("Regenerating summary with evaluator feedback...")
    try:
        result = chain.invoke(
            {
                "query": query,
                "previous_summary": previous_summary,
                "evidence": evidence,
                "evaluation": evaluation_text,
                "revision_instructions": revision_text,
            }
        )
        return result
    except Exception as e:
        logger.error(f"Regeneration chain failed: {e}")
        return previous_summary  # Preserve previous best version on failure
