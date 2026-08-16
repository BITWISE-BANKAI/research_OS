"""tests/test_rag.py — Basic unit + integration tests for RESEARCHOS."""
import os
import sys
import json
import pytest

# Make research_os importable from test run location
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# Ensure 'tools' package resolves to research_os/tools
if "tools" in sys.modules and not hasattr(sys.modules["tools"], "__path__"):
    del sys.modules["tools"]

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE, ".env"))

from tools.semantic_scholar import search_papers
from rag.chunker import split_documents
from rag.embeddings import get_embeddings
from utils.deduplication import is_duplicate
from research.evaluator import parse_evaluation
from utils.citations import format_citation

# ── 1. Semantic Scholar normalisation ──────────────────────────────
def test_search_papers_returns_list():
    """search_papers should return a list (may be empty on rate-limit)."""
    result = search_papers.invoke({"query": "attention is all you need"})
    assert isinstance(result, list), "Expected a list from search_papers"

def test_search_paper_fields():
    """Each returned paper should contain required keys."""
    papers = search_papers.invoke({"query": "BERT language model"})
    if not papers:
        pytest.skip("No results returned (likely rate-limited)")
    required_keys = {"paper_id","title","abstract","authors","year","citation_count","url"}
    for p in papers[:3]:
        assert required_keys.issubset(p.keys()), f"Missing keys in paper: {p.keys()}"


# ── 2. Chunker ─────────────────────────────────────────────────────
def test_chunker_basic():
    """Chunker must split a document and preserve metadata."""
    from langchain_core.documents import Document
    from rag.chunker import split_documents

    doc = Document(
        page_content="A" * 2500,  # longer than chunk_size
        metadata={
            "paper_id": "test_pid",
            "title": "Test Paper",
            "authors": ["Author One"],
            "year": 2024,
            "source": "test_source",
        }
    )
    chunks = split_documents([doc], chunk_size=1200, chunk_overlap=200)
    assert len(chunks) >= 2, "Should produce at least 2 chunks from 2500 chars"

def test_chunker_metadata_preservation():
    """Every chunk must carry the required metadata fields."""
    from langchain_core.documents import Document
    from rag.chunker import split_documents

    doc = Document(
        page_content="This is a test abstract. " * 80,
        metadata={
            "paper_id": "meta_pid",
            "title": "Meta Test Paper",
            "authors": ["Jane Doe", "John Smith"],
            "year": 2023,
            "source": "https://example.com/paper.pdf",
        }
    )
    chunks = split_documents([doc])
    required = {"paper_id","title","authors","year","source","chunk_id","page","section"}
    for chunk in chunks:
        missing = required - set(chunk.metadata.keys())
        assert not missing, f"Chunk missing metadata keys: {missing}"

def test_chunker_assigns_unique_chunk_ids():
    """Each chunk must get a unique chunk_id."""
    from langchain_core.documents import Document
    from rag.chunker import split_documents

    doc = Document(
        page_content="word " * 1000,
        metadata={"paper_id": "uid_pid", "title": "UID Test", "authors": [], "year": 2023, "source": "src"}
    )
    chunks = split_documents([doc])
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

def test_chunker_does_not_embed():
    """split_documents should return Document objects, not vectors."""
    from langchain_core.documents import Document
    from rag.chunker import split_documents

    doc = Document(page_content="Simple text.", metadata={"paper_id": "noembed", "title": "T", "authors": [], "year": 2024, "source": "s"})
    chunks = split_documents([doc])
    assert all(isinstance(c, Document) for c in chunks), "Chunker must return Document objects"


# ── 3. Embeddings initialisation ───────────────────────────────────
def test_embeddings_initialise():
    """Embeddings object must initialise without errors."""
    from rag.embeddings import get_embeddings
    emb = get_embeddings()
    assert emb is not None


# ── 4. Deduplication ───────────────────────────────────────────────
def test_dedup_by_paper_id():
    from utils.deduplication import is_duplicate
    existing = [{"paper_id": "abc123", "doi": None, "arxiv_id": None, "title": "Some Paper"}]
    new_paper = {"paper_id": "abc123", "doi": None, "arxiv_id": None, "title": "Some Paper"}
    assert is_duplicate(new_paper, existing) is True

def test_dedup_by_doi():
    from utils.deduplication import is_duplicate
    existing = [{"paper_id": "xyz", "doi": "10.1234/test", "arxiv_id": None, "title": "X"}]
    new_paper = {"paper_id": "other", "doi": "10.1234/test", "arxiv_id": None, "title": "Y"}
    assert is_duplicate(new_paper, existing) is True

def test_dedup_by_arxiv_id():
    from utils.deduplication import is_duplicate
    existing = [{"paper_id": "a", "doi": None, "arxiv_id": "2301.00001", "title": "A"}]
    new_paper = {"paper_id": "b", "doi": None, "arxiv_id": "2301.00001", "title": "B"}
    assert is_duplicate(new_paper, existing) is True

def test_dedup_by_normalized_title():
    from utils.deduplication import is_duplicate
    existing = [{"paper_id": "a", "doi": None, "arxiv_id": None, "title": "Attention Is All You Need"}]
    new_paper = {"paper_id": "b", "doi": None, "arxiv_id": None, "title": "attention is all you need!!"}
    assert is_duplicate(new_paper, existing) is True

def test_no_dedup_different_papers():
    from utils.deduplication import is_duplicate
    existing = [{"paper_id": "a", "doi": "10.1/a", "arxiv_id": "2301.00001", "title": "Paper A"}]
    new_paper = {"paper_id": "b", "doi": "10.1/b", "arxiv_id": "2301.99999", "title": "Paper B"}
    assert is_duplicate(new_paper, existing) is False


# ── 5. Evaluation parser ───────────────────────────────────────────
def test_parse_evaluation_score():
    from research.evaluator import parse_evaluation
    text = "SCORE: 7\nUNSUPPORTED_CLAIMS:\n- Claim A\nMISSING_LIMITATIONS:\nNONE\nMISSING_QUANTITATIVE_EVIDENCE:\nNONE\nREDUNDANCY_ISSUES:\nNONE\nCONTRADICTIONS:\nNONE\nMISSING_CITATIONS:\nNONE\nREVISION_INSTRUCTIONS:\n- Fix claim A"
    result = parse_evaluation(text)
    assert result["score"] == 7.0
    assert "Claim A" in result["unsupported_claims"][0]
    assert len(result["revision_instructions"]) >= 1

def test_parse_evaluation_none_score_on_missing():
    from research.evaluator import parse_evaluation
    result = parse_evaluation("No score here, just some text.")
    assert result["score"] is None, "Missing SCORE: should return None, not 0"


# ── 6. Citation formatter ──────────────────────────────────────────
def test_citation_format_with_page():
    from utils.citations import format_citation
    meta = {"title": "BERT", "page": 3, "section": "Methods", "source": "https://arxiv.org"}
    cit = format_citation(meta)
    assert "BERT" in cit
    assert "3" in cit

def test_citation_no_fabricated_page():
    from utils.citations import format_citation
    meta = {"title": "GPT-4", "page": None, "section": "General", "source": "https://openai.com"}
    cit = format_citation(meta)
    assert "openai.com" in cit or "GPT-4" in cit
    # Should NOT contain a fabricated page number
    assert "Page: None" not in cit


# ── 7. ChromaDB (integration — requires Ollama running) ────────────
@pytest.mark.integration
def test_chroma_add_and_retrieve():
    """End-to-end: add a document to ChromaDB and retrieve it."""
    import tempfile
    from langchain_core.documents import Document
    from rag.chroma_store import ChromaStore

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["CHROMA_PERSIST_DIR"] = tmpdir
        store = ChromaStore()

        doc = Document(
            page_content="The transformer architecture uses multi-head self-attention.",
            metadata={
                "paper_id": "chroma_test_01",
                "title": "Attention Is All You Need",
                "authors": "Vaswani et al.",
                "year": 2017,
                "section": "Introduction",
                "page": 1,
                "chunk_id": "chroma_test_01_chunk_1",
                "source": "https://arxiv.org/abs/1706.03762",
            }
        )
        ok = store.add_documents([doc])
        assert ok, "add_documents should return True"

        # Check exists
        assert store.paper_exists("chroma_test_01")

        # Similarity search
        results = store.similarity_search("attention self-attention mechanism", k=1)
        assert len(results) >= 1, "Should retrieve at least 1 chunk"

        # Delete
        store.delete_paper("chroma_test_01")
        assert not store.paper_exists("chroma_test_01")


# ── 8. Integration smoke test ──────────────────────────────────────
@pytest.mark.integration
def test_full_pipeline_smoke():
    """
    Smoke test: search → (fake) chunk/embed → store → retrieve.
    Does NOT download a real PDF to keep test deterministic.
    """
    import tempfile
    from langchain_core.documents import Document
    from rag.chunker import split_documents
    from rag.chroma_store import ChromaStore

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["CHROMA_PERSIST_DIR"] = tmpdir

        raw_doc = Document(
            page_content="Large language models tend to hallucinate by generating plausible but false information. "
                         "RLHF and RAG are two key mitigation strategies. " * 30,
            metadata={
                "paper_id": "smoke_test_pid",
                "title": "LLM Hallucination Survey",
                "authors": ["Test Author"],
                "year": 2024,
                "source": "https://arxiv.org/abs/0000.00000",
                "page": 0,
            }
        )
        chunks = split_documents([raw_doc], chunk_size=400, chunk_overlap=50)
        assert len(chunks) > 0

        store = ChromaStore()
        ok = store.add_documents(chunks)
        assert ok

        results = store.similarity_search("what causes LLM hallucination", k=3)
        assert len(results) > 0, "Should find relevant chunks"
        assert store.paper_exists("smoke_test_pid")

        papers = store.get_all_papers()
        assert any(p["paper_id"] == "smoke_test_pid" for p in papers)
