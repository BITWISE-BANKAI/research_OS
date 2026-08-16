"""research/evaluator.py - Evaluates a generated research briefing against ground-truth evidence."""
import os
import re
from typing import Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from prompts.evaluator_prompts import EVALUATOR_PROMPT
from utils.logging import setup_logger

logger = setup_logger("Evaluator")


def _parse_section(text: str, section_name: str, next_sections: list) -> list:
    """Extract a bullet-list section from structured evaluator output."""
    if next_sections:
        next_pattern = "|".join(re.escape(s) for s in next_sections)
        pattern = rf"{re.escape(section_name)}\s*:\s*(.*?)(?=\n(?:{next_pattern})\s*:|$)"
    else:
        pattern = rf"{re.escape(section_name)}\s*:\s*(.*?)$"

    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return []

    content = match.group(1).strip()
    if content.upper().strip() in ("NONE", "NONE.", "N/A"):
        return []

    return [
        line.lstrip("-•*").strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().upper().startswith("NONE")
    ]


def parse_evaluation(text: str) -> Dict[str, Any]:
    """
    Parse structured evaluator output into a python dict.
    Returns None score on parse failure — never returns score=0 on failure.
    """
    score: Optional[float] = None
    score_match = re.search(r"SCORE\s*:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))

    sections = [
        "UNSUPPORTED_CLAIMS",
        "MISSING_LIMITATIONS",
        "MISSING_QUANTITATIVE_EVIDENCE",
        "REDUNDANCY_ISSUES",
        "CONTRADICTIONS",
        "MISSING_CITATIONS",
        "REVISION_INSTRUCTIONS",
    ]

    return {
        "score": score,
        "raw": text,
        "unsupported_claims": _parse_section(text, "UNSUPPORTED_CLAIMS", sections[1:]),
        "missing_limitations": _parse_section(text, "MISSING_LIMITATIONS", sections[2:]),
        "missing_quantitative_evidence": _parse_section(text, "MISSING_QUANTITATIVE_EVIDENCE", sections[3:]),
        "redundancy_issues": _parse_section(text, "REDUNDANCY_ISSUES", sections[4:]),
        "contradictions": _parse_section(text, "CONTRADICTIONS", sections[5:]),
        "missing_citations": _parse_section(text, "MISSING_CITATIONS", sections[6:]),
        "revision_instructions": _parse_section(text, "REVISION_INSTRUCTIONS", []),
    }


def evaluate_summary(summary: str, evidence: str) -> Dict[str, Any]:
    """
    Run the evaluator chain against a research briefing.
    Returns a parsed evaluation dict. On failure, returns status='evaluator_failed'.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:latest")

    llm = ChatOllama(base_url=base_url, model=chat_model, temperature=0)
    chain = EVALUATOR_PROMPT | llm | StrOutputParser()

    logger.info("Evaluating research briefing...")
    try:
        raw_text = chain.invoke({"summary": summary, "evidence": evidence})
        parsed = parse_evaluation(raw_text)

        if parsed["score"] is None:
            logger.warning("Evaluator returned output but could not parse a score.")
            return {**parsed, "status": "score_parse_failed"}

        logger.info(f"Evaluation complete. Score: {parsed['score']}/10")
        return {**parsed, "status": "success"}

    except Exception as e:
        logger.error(f"Evaluator chain failed: {e}")
        return {
            "score": None,
            "raw": "",
            "status": "evaluator_failed",
            "error": str(e),
            "unsupported_claims": [],
            "missing_limitations": [],
            "missing_quantitative_evidence": [],
            "redundancy_issues": [],
            "contradictions": [],
            "missing_citations": [],
            "revision_instructions": [],
        }
