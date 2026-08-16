"""rag/mongo_store.py - MongoDB storage for JSON documents, metadata, briefings, and logs."""
import os
import json
import uuid
from datetime import datetime, timezone

from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from utils.logging import setup_logger

load_dotenv()
logger = setup_logger("MongoStore")

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


class MongoStore:
    """
    MongoDB JSON Storage manager for ResearchOS.
    Stores:
      - `papers`: Paper metadata JSON records
      - `chunks`: Full-text document chunk JSON records
      - `research_briefings`: Deep research briefings and evaluation audit histories
      - `chat_history`: Grounded QA interactions, citations, and retrieved evidence
      - `comparisons`: Synthesized cross-paper comparative analyses

    Includes automatic local JSON fallback if MongoDB server is offline or unreachable.
    """

    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = os.getenv("MONGO_DB_NAME", "research_os")
        self.fallback_dir = os.getenv("MONGO_FALLBACK_DIR", "./data/mongodb_json_backup")
        self.is_connected = False
        self.client = None
        self.db = None

        os.makedirs(self.fallback_dir, exist_ok=True)
        for col in ["papers", "chunks", "research_briefings", "chat_history", "comparisons"]:
            os.makedirs(os.path.join(self.fallback_dir, col), exist_ok=True)

        self._connect()

    def _connect(self):
        """Attempt connection to MongoDB with quick timeout."""
        if not PYMONGO_AVAILABLE:
            logger.warning("pymongo is not installed. Using local JSON file store fallback.")
            self.is_connected = False
            return

        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=2000)
            # Test connection
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self.is_connected = True
            logger.info(f"Connected to MongoDB at {self.mongo_uri} (database: {self.db_name})")

            # Create unique indexes
            self.db.papers.create_index("paper_id", unique=True)
            self.db.chunks.create_index("chunk_id", unique=True)
            self.db.chunks.create_index("paper_id")
        except Exception as e:
            self.is_connected = False
            logger.warning(f"MongoDB connection failed ({e}). Utilizing local JSON file fallback at {self.fallback_dir}")

    # ── Fallback Helpers ──────────────────────────────────────────────────────────
    def _fallback_save(self, collection: str, key: str, data: dict):
        path = os.path.join(self.fallback_dir, collection, f"{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _fallback_get(self, collection: str, key: str) -> Optional[dict]:
        path = os.path.join(self.fallback_dir, collection, f"{key}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading fallback file {path}: {e}")
        return None

    def _fallback_get_all(self, collection: str) -> List[dict]:
        dir_path = os.path.join(self.fallback_dir, collection)
        results = []
        if os.path.exists(dir_path):
            for fname in os.listdir(dir_path):
                if fname.endswith(".json"):
                    fpath = os.path.join(dir_path, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            results.append(json.load(f))
                    except Exception as e:
                        logger.error(f"Error reading fallback file {fpath}: {e}")
        return results

    def _fallback_delete(self, collection: str, key: str) -> bool:
        path = os.path.join(self.fallback_dir, collection, f"{key}.json")
        if os.path.exists(path):
            try:
                os.remove(path)
                return True
            except Exception as e:
                logger.error(f"Error deleting fallback file {path}: {e}")
        return False

    # ── Papers Collection ─────────────────────────────────────────────────────────
    def save_paper(self, paper_data: dict) -> bool:
        """Save or update paper metadata JSON."""
        try:
            pid = paper_data.get("paper_id")
            if not pid:
                logger.error("Cannot save paper without paper_id")
                return False

            doc = dict(paper_data)
            doc["updated_at"] = datetime.now(timezone.utc).isoformat()
            if "created_at" not in doc:
                doc["created_at"] = doc["updated_at"]

            if self.is_connected:
                self.db.papers.update_one({"paper_id": pid}, {"$set": doc}, upsert=True)
            else:
                self._fallback_save("papers", pid, doc)

            logger.info(f"Saved paper metadata JSON for '{pid}' ({doc.get('title', 'N/A')})")
            return True
        except Exception as e:
            logger.error(f"Error saving paper to MongoDB: {e}")
            try:
                self._fallback_save("papers", paper_data.get("paper_id", "unknown"), paper_data)
                return True
            except Exception:
                return False

    def get_paper(self, paper_id: str) -> Optional[dict]:
        """Fetch paper metadata JSON by paper_id."""
        try:
            if self.is_connected:
                doc = self.db.papers.find_one({"paper_id": paper_id}, {"_id": 0})
                if doc:
                    return doc
            return self._fallback_get("papers", paper_id)
        except Exception as e:
            logger.error(f"Error fetching paper {paper_id}: {e}")
            return self._fallback_get("papers", paper_id)

    def get_all_papers(self) -> List[dict]:
        """Fetch all stored paper metadata JSON records."""
        try:
            if self.is_connected:
                papers = list(self.db.papers.find({}, {"_id": 0}))
                if papers:
                    return papers
            return self._fallback_get_all("papers")
        except Exception as e:
            logger.error(f"Error fetching all papers from MongoDB: {e}")
            return self._fallback_get_all("papers")

    def delete_paper(self, paper_id: str) -> bool:
        """Delete paper metadata and its chunks from MongoDB."""
        try:
            deleted = False
            if self.is_connected:
                res1 = self.db.papers.delete_one({"paper_id": paper_id})
                res2 = self.db.chunks.delete_many({"paper_id": paper_id})
                deleted = res1.deleted_count > 0 or res2.deleted_count > 0
            
            fb_del = self._fallback_delete("papers", paper_id)
            chunks_dir = os.path.join(self.fallback_dir, "chunks")
            if os.path.exists(chunks_dir):
                for fname in list(os.listdir(chunks_dir)):
                    if fname.startswith(f"{paper_id}_"):
                        try:
                            os.remove(os.path.join(chunks_dir, fname))
                        except Exception:
                            pass
            return deleted or fb_del
        except Exception as e:
            logger.error(f"Error deleting paper {paper_id}: {e}")
            return False

    def paper_exists(self, paper_id: str) -> bool:
        """Check if paper is stored in MongoDB."""
        try:
            if self.is_connected:
                return self.db.papers.count_documents({"paper_id": paper_id}) > 0
            return self._fallback_get("papers", paper_id) is not None
        except Exception as e:
            logger.error(f"Error checking paper exists: {e}")
            return self._fallback_get("papers", paper_id) is not None

    # ── Chunks Collection ─────────────────────────────────────────────────────────
    def save_chunks(self, paper_id: str, chunks_data: List[dict]) -> bool:
        """Save full-text chunk JSON documents."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            for chunk in chunks_data:
                chunk["paper_id"] = paper_id
                chunk["created_at"] = now
                cid = chunk.get("chunk_id", str(uuid.uuid4()))
                chunk["chunk_id"] = cid

                if self.is_connected:
                    self.db.chunks.update_one({"chunk_id": cid}, {"$set": chunk}, upsert=True)
                else:
                    self._fallback_save("chunks", cid, chunk)

            logger.info(f"Saved {len(chunks_data)} chunk JSON records for paper {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving chunks for paper {paper_id}: {e}")
            return False

    def get_chunks(self, paper_id: Optional[str] = None) -> List[dict]:
        """Fetch all chunks or chunks for a specific paper."""
        try:
            if self.is_connected:
                query = {"paper_id": paper_id} if paper_id else {}
                return list(self.db.chunks.find(query, {"_id": 0}))
            all_chunks = self._fallback_get_all("chunks")
            if paper_id:
                return [c for c in all_chunks if c.get("paper_id") == paper_id]
            return all_chunks
        except Exception as e:
            logger.error(f"Error fetching chunks: {e}")
            return []

    # ── Deep Research Briefings Collection ────────────────────────────────────────
    def save_briefing(self, briefing_data: dict) -> str:
        """Save a deep research briefing and audit trail JSON."""
        try:
            briefing_id = briefing_data.get("id", str(uuid.uuid4()))
            doc = dict(briefing_data)
            doc["id"] = briefing_id
            doc["created_at"] = datetime.now(timezone.utc).isoformat()

            if self.is_connected:
                self.db.research_briefings.update_one({"id": briefing_id}, {"$set": doc}, upsert=True)
            else:
                self._fallback_save("research_briefings", briefing_id, doc)

            logger.info(f"Saved deep research briefing JSON: {briefing_id}")
            return briefing_id
        except Exception as e:
            logger.error(f"Error saving briefing to MongoDB: {e}")
            return ""

    def get_briefings(self, limit: int = 50) -> List[dict]:
        """Fetch past research briefing JSON records."""
        try:
            if self.is_connected:
                return list(self.db.research_briefings.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
            return sorted(self._fallback_get_all("research_briefings"), key=lambda x: x.get("created_at", ""), reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Error fetching briefings: {e}")
            return []

    # ── Chat History Collection ───────────────────────────────────────────────────
    def save_chat_interaction(self, chat_data: dict) -> str:
        """Save a grounded Q&A chat interaction JSON record."""
        try:
            interaction_id = chat_data.get("id", str(uuid.uuid4()))
            doc = dict(chat_data)
            doc["id"] = interaction_id
            doc["created_at"] = datetime.now(timezone.utc).isoformat()

            if self.is_connected:
                self.db.chat_history.insert_one(doc)
                if "_id" in doc:
                    del doc["_id"]
            else:
                self._fallback_save("chat_history", interaction_id, doc)

            return interaction_id
        except Exception as e:
            logger.error(f"Error saving chat interaction: {e}")
            return ""

    def get_chat_history(self, limit: int = 100) -> List[dict]:
        """Fetch chat history JSON records."""
        try:
            if self.is_connected:
                return list(self.db.chat_history.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
            return sorted(self._fallback_get_all("chat_history"), key=lambda x: x.get("created_at", ""), reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Error fetching chat history: {e}")
            return []

    # ── Comparisons Collection ────────────────────────────────────────────────────
    def save_comparison(self, comparison_data: dict) -> str:
        """Save a paper comparison report JSON."""
        try:
            cid = comparison_data.get("id", str(uuid.uuid4()))
            doc = dict(comparison_data)
            doc["id"] = cid
            doc["created_at"] = datetime.now(timezone.utc).isoformat()

            if self.is_connected:
                self.db.comparisons.update_one({"id": cid}, {"$set": doc}, upsert=True)
            else:
                self._fallback_save("comparisons", cid, doc)

            return cid
        except Exception as e:
            logger.error(f"Error saving comparison: {e}")
            return ""

    def get_comparisons(self, limit: int = 50) -> List[dict]:
        """Fetch past paper comparison JSON reports."""
        try:
            if self.is_connected:
                return list(self.db.comparisons.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
            return sorted(self._fallback_get_all("comparisons"), key=lambda x: x.get("created_at", ""), reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Error fetching comparisons: {e}")
            return []

    # ── System Stats ──────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics for MongoDB stored JSON records."""
        try:
            papers_count = len(self.get_all_papers())
            chunks_count = len(self.get_chunks())
            briefings_count = len(self.get_briefings())
            chat_count = len(self.get_chat_history())
            comparisons_count = len(self.get_comparisons())

            return {
                "connected": self.is_connected,
                "engine": "MongoDB" if self.is_connected else "Local JSON Backup",
                "database": self.db_name if self.is_connected else self.fallback_dir,
                "papers_count": papers_count,
                "chunks_count": chunks_count,
                "briefings_count": briefings_count,
                "chat_count": chat_count,
                "comparisons_count": comparisons_count,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "connected": False,
                "engine": "Error",
                "papers_count": 0,
                "chunks_count": 0,
                "briefings_count": 0,
                "chat_count": 0,
                "comparisons_count": 0,
            }
