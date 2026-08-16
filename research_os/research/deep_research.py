"""research/deep_research.py - Iterative deep research pipeline with evaluation loop."""
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from rag.chroma_store import ChromaStore
from rag.mongo_store import MongoStore
from research.summary import generate_summary
from research.evaluator import evaluate_summary
from research.regeneration import regenerate_summary
from utils.logging import setup_logger

logger = setup_logger("DeepResearch")


def _mongo_chunk_to_document(c: dict) -> Document:
    return Document(
        page_content=c.get("content", ""),
        metadata={
            "paper_id": c.get("paper_id", ""),
            "title": c.get("title", "Unknown"),
            "authors": c.get("authors", []),
            "year": c.get("year"),
            "section": c.get("section", "General"),
            "page": c.get("page", "N/A"),
            "source": c.get("source", "MongoDB"),
        },
    )


def _get_chunks_for_paper(
    store: ChromaStore,
    mongo_store: MongoStore,
    pid: str,
    query: str,
    k: int,
    embedding_healthy: bool,
) -> List[Document]:
    """
    Resilient per-paper chunk retrieval, mirroring RAGRetriever's fallback cascade:
      1. ChromaDB similarity search (only when Ollama embeddings are healthy)
      2. MongoDB / local JSON chunk store (currently empty until the indexer
         dual-writes there, but checked in case that changes)
      3. Raw ChromaDB pull — bypasses embeddings entirely, reads whatever text
         is physically stored regardless of Ollama's state
      4. Paper abstract as an absolute last resort, so a paper is never silently
         dropped from the evidence set just because full-text chunks are missing
    """
    # 1. Real similarity search
    if embedding_healthy:
        chunks = store.similarity_search(query, paper_ids=[pid], k=k)
        if chunks:
            return chunks
    else:
        logger.warning(f"Ollama embeddings unavailable — skipping vector search for paper {pid}.")

    # 2. MongoDB / local JSON chunk fallback
    mongo_chunks = mongo_store.get_chunks(pid)
    if mongo_chunks:
        logger.info(f"Recovered {len(mongo_chunks)} chunks for paper {pid} from MongoDB/JSON store.")
        return [_mongo_chunk_to_document(c) for c in mongo_chunks]

    # 3. Raw ChromaDB pull (no embedding needed — reads stored text directly)
    try:
        raw = store.collection.get(where={"paper_id": pid})
        docs = raw.get("documents", [])
        metas = raw.get("metadatas", [])
        if docs:
            out = []
            for i, text in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                authors_raw = meta.get("authors", "")
                authors_val = (
                    [a.strip() for a in authors_raw.split(",") if a.strip()]
                    if isinstance(authors_raw, str)
                    else (authors_raw or [])
                )
                out.append(Document(
                    page_content=text or "",
                    metadata={
                        "paper_id": meta.get("paper_id", pid),
                        "title": meta.get("title", "Unknown"),
                        "authors": authors_val,
                        "year": meta.get("year"),
                        "section": meta.get("section", "General"),
                        "page": meta.get("page", "N/A"),
                        "source": meta.get("source", "ChromaDB-raw"),
                    },
                ))
            logger.info(f"Recovered {len(out)} raw chunks for paper {pid} directly from ChromaDB (bypassing embeddings).")
            return out[:k]
    except Exception as e:
        logger.warning(f"Raw ChromaDB fetch failed for paper {pid}: {e}")

    # 4. Abstract as last resort
    meta = store.get_paper_metadata(pid) or mongo_store.get_paper(pid)
    if meta:
        abstract = meta.get("abstract")
        if abstract and abstract != "N/A":
            logger.info(f"Using abstract as last-resort evidence for paper {pid} — no chunks found anywhere.")
            return [Document(
                page_content=f"Title: {meta.get('title', '')}\n\nAbstract: {abstract}",
                metadata={
                    "paper_id": pid,
                    "title": meta.get("title", "Unknown"),
                    "authors": meta.get("authors", []),
                    "year": meta.get("year"),
                    "section": "Abstract",
                    "page": "N/A",
                    "source": "Paper-Metadata",
                },
            )]

    return []


def build_evidence_context(paper_ids: List[str], query: str, k_per_paper: int = 8) -> str:
    """
    Retrieve and assemble an evidence context string for each requested paper,
    using a resilient fallback cascade so the pipeline still produces evidence
    when Ollama embeddings are down, or when a paper's chunks only made it
    into one of the storage backends.
    """
    store = ChromaStore()
    mongo_store = MongoStore()
    embedding_healthy = store.is_embedding_healthy()
    if not embedding_healthy:
        logger.warning("Ollama embeddings unavailable — evidence retrieval will use non-embedding fallbacks only.")

    context_blocks: List[str] = []

    # Per-paper targeted retrieval, with fallback cascade
    for pid in paper_ids:
        meta = store.get_paper_metadata(pid)
        title = meta.get("title", "Unknown") if meta else "Unknown"
        year = meta.get("year", "N/A") if meta else "N/A"

        chunks = _get_chunks_for_paper(store, mongo_store, pid, query, k_per_paper, embedding_healthy)

        if not chunks:
            logger.warning(f"No chunks found for paper {pid} in any store (Chroma, Mongo, raw, or abstract) — likely never indexed.")
            continue

        paper_block = f"### PAPER: {title} ({year})\n"
        for doc in chunks:
            sec = doc.metadata.get("section", "General")
            pg = doc.metadata.get("page", "N/A")
            paper_block += f"[Section: {sec} | Page: {pg}]\n{doc.page_content}\n\n"
        context_blocks.append(paper_block)

    # Cross-paper broad retrieval pass — only meaningful with real embeddings;
    # skipped when Ollama is down since a mismatched-vector search adds noise,
    # not signal, on top of what the per-paper fallbacks already gathered.
    if embedding_healthy:
        broad_chunks = store.similarity_search(query, paper_ids=paper_ids, k=10)
        if broad_chunks:
            broad_block = "### CROSS-PAPER EVIDENCE (Broad Retrieval)\n"
            for doc in broad_chunks:
                t = doc.metadata.get("title", "Unknown")
                sec = doc.metadata.get("section", "General")
                pg = doc.metadata.get("page", "N/A")
                broad_block += f"[{t} | Section: {sec} | Page: {pg}]\n{doc.page_content}\n\n"
            context_blocks.append(broad_block)

    return "\n---\n".join(context_blocks)


def iterative_summarize(
    query: str,
    paper_ids: List[str],
    max_iterations: int = 3,
    score_threshold: int = 8,
) -> Dict[str, Any]:
    """
    Iterative summarisation loop:
      1. Build RAG evidence context from ChromaDB.
      2. Generate initial briefing.
      3. Evaluate — if score >= threshold, return.
      4. Otherwise regenerate with feedback.
      5. Repeat up to max_iterations.
      6. Return best available briefing + audit trail and persist to MongoDB.

    Returns a dict:
      {
        "summary": str,
        "final_score": float | None,
        "iterations_run": int,
        "evaluation_history": list,
        "status": str,
        "evidence_context": str,
      }
    """
    logger.info(f"Starting deep research pipeline for query: '{query}' over {len(paper_ids)} papers.")
    mongo_store = MongoStore()

    # 1. Build evidence corpus
    evidence = build_evidence_context(paper_ids, query)
    if not evidence.strip():
        result = {
            "summary": "Error: No indexed evidence found for the selected papers. Please index the papers first.",
            "final_score": None,
            "iterations_run": 0,
            "evaluation_history": [],
            "status": "no_evidence",
            "evidence_context": "",
        }
        return result

    # 2. Initial generation
    summary = generate_summary(query, evidence)
    if summary.startswith("ERROR:"):
        result = {
            "summary": summary,
            "final_score": None,
            "iterations_run": 0,
            "evaluation_history": [],
            "status": "generation_failed",
            "evidence_context": evidence,
        }
        return result

    evaluation_history: List[Dict] = []
    best_summary = summary
    best_score: Optional[float] = None
    final_status = "max_iterations_reached"
    iterations_run = 0

    # 3. Iterative refinement loop
    for iteration in range(1, max_iterations + 1):
        iterations_run = iteration
        logger.info(f"Evaluation pass {iteration}/{max_iterations}...")
        evaluation = evaluate_summary(summary, evidence)
        evaluation_history.append({"iteration": iteration, **evaluation})

        # Handle evaluator failures gracefully
        if evaluation["status"] in ("evaluator_failed", "score_parse_failed"):
            logger.warning(f"Evaluator failed on iteration {iteration}. Stopping loop.")
            final_status = evaluation["status"]
            break

        score = evaluation["score"]
        logger.info(f"  → Score: {score}/10")

        # Track best version
        if best_score is None or (score is not None and score > best_score):
            best_score = score
            best_summary = summary

        if score is not None and score >= score_threshold:
            logger.info(f"Score {score} meets threshold {score_threshold}. Stopping.")
            final_status = "threshold_met"
            break

        # Only regenerate if not on last iteration
        if iteration < max_iterations:
            logger.info(f"  → Regenerating with feedback ({len(evaluation.get('revision_instructions', []))} instructions)...")
            summary = regenerate_summary(query, summary, evidence, evaluation)

    result = {
        "query": query,
        "paper_ids": paper_ids,
        "summary": best_summary,
        "final_score": best_score,
        "iterations_run": iterations_run,
        "evaluation_history": evaluation_history,
        "status": final_status,
        "evidence_context": evidence,
    }

    # Save to MongoDB
    try:
        mongo_store.save_briefing(result)
        logger.info("Deep research briefing successfully saved to MongoDB.")
    except Exception as e:
        logger.error(f"Failed to save briefing to MongoDB: {e}")

    return result