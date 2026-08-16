"""tests/test_mongo.py - Unit tests for MongoDB & local JSON fallback storage."""
import os
import sys
import tempfile
import pytest

# Make research_os importable
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from rag.mongo_store import MongoStore


@pytest.fixture
def temp_mongo_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["MONGO_FALLBACK_DIR"] = tmpdir
        store = MongoStore()
        yield store


def test_mongo_save_and_get_paper(temp_mongo_store):
    paper_data = {
        "paper_id": "test_paper_123",
        "title": "Attention Is All You Need",
        "authors": ["Vaswani et al."],
        "year": 2017,
        "citation_count": 100000,
        "fields_of_study": ["Computer Science"],
        "url": "https://arxiv.org/abs/1706.03762",
    }
    ok = temp_mongo_store.save_paper(paper_data)
    assert ok is True

    fetched = temp_mongo_store.get_paper("test_paper_123")
    assert fetched is not None
    assert fetched["title"] == "Attention Is All You Need"
    assert fetched["paper_id"] == "test_paper_123"
    assert temp_mongo_store.paper_exists("test_paper_123") is True


def test_mongo_save_and_get_chunks(temp_mongo_store):
    chunks = [
        {
            "chunk_id": "test_paper_123_chunk_1",
            "content": "Self-attention mechanism relates different positions of a single sequence.",
            "section": "Introduction",
            "page": 1,
            "source": "https://arxiv.org/pdf/1706.03762.pdf",
        },
        {
            "chunk_id": "test_paper_123_chunk_2",
            "content": "Multi-head attention allows the model to jointly attend to information.",
            "section": "Model Architecture",
            "page": 3,
            "source": "https://arxiv.org/pdf/1706.03762.pdf",
        },
    ]
    ok = temp_mongo_store.save_chunks("test_paper_123", chunks)
    assert ok is True

    fetched_chunks = temp_mongo_store.get_chunks("test_paper_123")
    assert len(fetched_chunks) == 2
    assert any(c["chunk_id"] == "test_paper_123_chunk_1" for c in fetched_chunks)


def test_mongo_save_and_get_briefing(temp_mongo_store):
    briefing = {
        "id": "briefing_test_001",
        "query": "LLM hallucination mechanisms",
        "paper_ids": ["test_paper_123"],
        "summary": "Deep analysis of hallucination causes...",
        "final_score": 8.5,
        "iterations_run": 2,
        "status": "threshold_met",
    }
    bid = temp_mongo_store.save_briefing(briefing)
    assert bid == "briefing_test_001"

    briefings = temp_mongo_store.get_briefings()
    assert len(briefings) >= 1
    assert any(b["id"] == "briefing_test_001" for b in briefings)


def test_mongo_save_and_get_chat_history(temp_mongo_store):
    chat = {
        "id": "chat_001",
        "query": "What is the Transformer architecture?",
        "mode": "library",
        "answer": "The transformer relies entirely on an attention mechanism.",
        "sources": [{"title": "Attention Is All You Need", "year": 2017}],
        "status": "success",
    }
    cid = temp_mongo_store.save_chat_interaction(chat)
    assert cid == "chat_001"

    history = temp_mongo_store.get_chat_history()
    assert len(history) >= 1
    assert any(h["id"] == "chat_001" for h in history)


def test_mongo_save_and_get_comparison(temp_mongo_store):
    comp = {
        "id": "comp_001",
        "paper_ids": ["paper_a", "paper_b"],
        "comparison": "Paper A uses CNNs whereas Paper B uses Transformers.",
    }
    cid = temp_mongo_store.save_comparison(comp)
    assert cid == "comp_001"

    comps = temp_mongo_store.get_comparisons()
    assert len(comps) >= 1
    assert any(c["id"] == "comp_001" for c in comps)


def test_mongo_delete_paper(temp_mongo_store):
    paper_data = {"paper_id": "del_test_pid", "title": "To Delete"}
    temp_mongo_store.save_paper(paper_data)
    temp_mongo_store.save_chunks("del_test_pid", [{"chunk_id": "del_test_pid_chunk_1", "content": "Sample"}])

    assert temp_mongo_store.paper_exists("del_test_pid") is True
    deleted = temp_mongo_store.delete_paper("del_test_pid")
    assert deleted is True
    assert temp_mongo_store.paper_exists("del_test_pid") is False


def test_mongo_stats(temp_mongo_store):
    stats = temp_mongo_store.get_stats()
    assert "connected" in stats
    assert "papers_count" in stats
    assert "chunks_count" in stats
    assert "briefings_count" in stats
