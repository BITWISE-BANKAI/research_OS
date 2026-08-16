"""app.py - FastAPI server for RESEARCHOS."""
import os
import sys
from pathlib import Path

# ── Make sure research_os/ is importable when run from research_os/ ──────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from tools.semantic_scholar import search_papers
from tools.paper_indexer import index_paper
from rag.chroma_store import ChromaStore
from rag.mongo_store import MongoStore
from rag.retriever import RAGRetriever
from research.deep_research import iterative_summarize
from tools.comparison_tools import compare_papers
from models.schemas import (
    SearchQueryRequest,
    IndexPaperRequest,
    ManualPaperRequest,
    ChatRequest,
    CompareRequest,
    DeepResearchRequest,
)
from utils.logging import setup_logger

logger = setup_logger("App")

app = FastAPI(
    title="RESEARCHOS",
    description="Local AI Research Intelligence & RAG Chatbot with MongoDB & ChromaDB",
    version="1.0.0",
)


# Allow CORS for local development so browser-based clients can call the API
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Singletons (initialised lazily on first use to avoid startup cost) ────────
_retriever: Optional[RAGRetriever] = None
_store: Optional[ChromaStore] = None
_mongo_store: Optional[MongoStore] = None


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


def get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store


def get_mongo_store() -> MongoStore:
    global _mongo_store
    if _mongo_store is None:
        _mongo_store = MongoStore()
    return _mongo_store


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    mongo = get_mongo_store()
    return {
        "status": "ok",
        "service": "RESEARCHOS",
        "mongo_connected": mongo.is_connected,
        "mongo_engine": "MongoDB" if mongo.is_connected else "Local JSON Backup",
    }


# ── Search papers ─────────────────────────────────────────────────────────────
@app.post("/api/search")
async def search(body: SearchQueryRequest):
    try:
        results = search_papers.invoke({"query": body.query})
        if not results:
            return {"papers": [], "message": "No papers found or Semantic Scholar unavailable."}
        return {"papers": results}
    except Exception as e:
        logger.error(f"/api/search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Index a paper ─────────────────────────────────────────────────────────────
@app.post("/api/index")
async def index(body: IndexPaperRequest):
    """Index endpoint: downloads, chunks, embeds and stores in MongoDB + ChromaDB.

    Accepts:
      - paper_id: Semantic Scholar paper ID or arXiv ID
      - paper: Complete dictionary of paper metadata (optional)
    """

    try:
        pid = body.paper_id
        paper_obj = body.paper

        # ---------------------------------------------------------
        # 1. Get paper_id
        # ---------------------------------------------------------
        if not pid and isinstance(paper_obj, dict):
            pid = paper_obj.get("paper_id")

        if isinstance(pid, str):
            pid = pid.strip()

        if not pid or str(pid).lower() == "string":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing or invalid paper_id in request body. "
                    "Provide a valid paper_id or use /api/papers/manual."
                )
            )

        pid = str(pid)

        # ---------------------------------------------------------
        # 2. Normalize paper metadata
        # ---------------------------------------------------------
        if isinstance(paper_obj, dict):
            paper_obj = dict(paper_obj)

            # The request paper_id is the canonical ID.
            paper_obj["paper_id"] = pid

        # ---------------------------------------------------------
        # 3. Save metadata to MongoDB first
        # ---------------------------------------------------------
        if isinstance(paper_obj, dict):
            try:
                mongo = get_mongo_store()
                mongo.save_paper(paper_obj)

                logger.info(
                    f"Pre-saved paper metadata for paper_id={pid}"
                )

            except Exception as e:
                # Metadata persistence failure should not prevent
                # the indexing pipeline from attempting to run.
                logger.warning(
                    f"Could not pre-save paper metadata: {e}"
                )

        # ---------------------------------------------------------
        # 4. ACTUALLY RUN THE INDEXING PIPELINE
        # ---------------------------------------------------------
        result_msg = index_paper.invoke({
            "paper_id": pid,
            "paper_data": paper_obj
        })

        logger.info(
            f"Indexing completed for paper_id={pid}: {result_msg}"
        )

        return {
            "message": result_msg,
            "paper_id": pid
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            f"/api/index error for paper_id={body.paper_id}: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ── Add manual paper ──────────────────────────────────────────────────────────
@app.post("/api/papers/manual")
async def add_manual_paper(body: ManualPaperRequest):
    """Directly index a manual research paper or note into MongoDB and ChromaDB without Semantic Scholar."""
    try:
        import uuid
        pid = body.paper_id.strip() if body.paper_id else f"manual_{uuid.uuid4().hex[:10]}"
        
        paper_dict = {
            "paper_id": pid,
            "title": body.title.strip(),
            "abstract": body.abstract.strip() if body.abstract else "No abstract provided.",
            "authors": body.authors or ["Unknown Author"],
            "year": body.year,
            "url": body.url or "",
            "content": body.content or body.abstract or body.title,
            "citation_count": 0,
            "influential_citation_count": 0,
            "fields_of_study": ["Manual Entry"],
        }
        
        mongo = get_mongo_store()
        mongo.save_paper(paper_dict)
        
        result_msg = index_paper.invoke({"paper_id": str(pid), "paper_data": paper_obj})
        return {
            "message": result_msg,
            "paper_id": str(pid),
            "paper": paper_dict
        }
    except Exception as e:
        logger.error(f"/api/papers/manual error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ── List indexed papers ───────────────────────────────────────────────────────
@app.get("/api/papers")
async def list_papers():
    try:
        mongo = get_mongo_store()
        chroma = get_store()
        papers_mongo = mongo.get_all_papers()
        papers_chroma = chroma.get_all_papers()

        # Merge unique papers by paper_id
        seen = {}
        for p in papers_mongo + papers_chroma:
            pid = p.get("paper_id")
            if pid and pid not in seen:
                seen[pid] = p
        return {"papers": list(seen.values())}
    except Exception as e:
        logger.error(f"/api/papers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Delete paper from index & MongoDB ─────────────────────────────────────────
@app.delete("/api/papers/{paper_id}")
async def delete_paper(paper_id: str):
    try:
        chroma = get_store()
        mongo = get_mongo_store()
        ok_chroma = chroma.delete_paper(paper_id)
        ok_mongo = mongo.delete_paper(paper_id)
        if ok_chroma or ok_mongo:
            return {"message": f"Paper {paper_id} deleted from ChromaDB and MongoDB."}
        raise HTTPException(status_code=404, detail="Paper not found in index.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/api/papers DELETE error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── RAG Chat ──────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(body: ChatRequest):
    try:
        retriever = get_retriever()
        result = retriever.generate_answer(
            query=body.query,
            mode=body.mode,
            paper_ids=body.paper_ids if body.paper_ids else None,
            k=body.k,
        )
        return result
    except Exception as e:
        logger.error(f"/api/chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Compare papers ────────────────────────────────────────────────────────────
@app.post("/api/compare")
async def compare(body: CompareRequest):
    try:
        if len(body.paper_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 paper IDs are required.")
        result = compare_papers.invoke({"paper_ids": body.paper_ids})
        return {"comparison": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/api/compare error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Deep Research ─────────────────────────────────────────────────────────────
@app.post("/api/deep-research")
async def deep_research(body: DeepResearchRequest):
    try:
        if not body.paper_ids:
            raise HTTPException(status_code=400, detail="At least one paper_id is required.")
        result = iterative_summarize(
            query=body.query,
            paper_ids=body.paper_ids,
            max_iterations=body.max_iterations,
            score_threshold=body.score_threshold,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/api/deep-research error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── MongoDB JSON Storage Endpoints ────────────────────────────────────────────
@app.get("/api/mongo/stats")
async def mongo_stats():
    """Get statistics for JSON documents and collections stored in MongoDB."""
    try:
        mongo = get_mongo_store()
        return mongo.get_stats()
    except Exception as e:
        logger.error(f"/api/mongo/stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mongo/briefings")
async def get_stored_briefings(limit: int = 50):
    """Retrieve all deep research briefings stored in MongoDB."""
    try:
        mongo = get_mongo_store()
        return {"briefings": mongo.get_briefings(limit=limit)}
    except Exception as e:
        logger.error(f"/api/mongo/briefings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mongo/chunks/{paper_id}")
async def get_paper_chunks(paper_id: str):
    """Retrieve all chunk JSON documents for a specific paper from MongoDB."""
    try:
        mongo = get_mongo_store()
        chunks = mongo.get_chunks(paper_id=paper_id)
        return {"paper_id": paper_id, "chunks": chunks}
    except Exception as e:
        logger.error(f"/api/mongo/chunks/{paper_id} error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mongo/history")
async def get_chat_history(limit: int = 100):
    """Retrieve grounded QA chat history from MongoDB."""
    try:
        mongo = get_mongo_store()
        return {"history": mongo.get_chat_history(limit=limit)}
    except Exception as e:
        logger.error(f"/api/mongo/history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mongo/comparisons")
async def get_comparisons(limit: int = 50):
    """Retrieve past comparisons stored in MongoDB."""
    try:
        mongo = get_mongo_store()
        return {"comparisons": mongo.get_comparisons(limit=limit)}
    except Exception as e:
        logger.error(f"/api/mongo/comparisons error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

