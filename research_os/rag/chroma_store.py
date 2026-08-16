import os
import json
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from rag.embeddings import get_embeddings
from utils.logging import setup_logger

load_dotenv()
logger = setup_logger("ChromaStore")

class ChromaStore:
    def __init__(self):
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        self.collection_name = "research_papers"
        self.embeddings = get_embeddings()
        
        # Ensure persist directory exists
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.collection = self.db._collection
        logger.info(f"Persistent ChromaDB initialized at {self.persist_dir} (collection: {self.collection_name})")

    def is_embedding_healthy(self) -> bool:
        """Check whether Ollama embeddings are reachable. Returns False when only hash fallback is available."""
        if hasattr(self.embeddings, 'check_health'):
            return self.embeddings.check_health()
        return True

    def _serialize_metadata(self, metadata: dict) -> dict:
        """Serialize metadata dictionary to ensure all values are compatible with Chroma (str, int, float, bool)."""
        clean_meta = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, list):
                # Convert list (e.g. authors, fields) to JSON string or comma-separated string
                if k == "authors":
                    clean_meta[k] = ", ".join(v)
                elif k == "fields_of_study" or k == "fields":
                    clean_meta[k] = ", ".join(v)
                else:
                    clean_meta[k] = json.dumps(v)
            elif isinstance(v, (dict, set)):
                clean_meta[k] = json.dumps(v)
            elif isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        return clean_meta

    def _deserialize_metadata(self, metadata: dict) -> dict:
        """Deserialize metadata fields from Chroma storage back to python types."""
        clean_meta = dict(metadata)
        if "authors" in clean_meta and isinstance(clean_meta["authors"], str):
            clean_meta["authors"] = [a.strip() for a in clean_meta["authors"].split(",") if a.strip()]
        if "fields_of_study" in clean_meta and isinstance(clean_meta["fields_of_study"], str):
            clean_meta["fields_of_study"] = [f.strip() for f in clean_meta["fields_of_study"].split(",") if f.strip()]
        return clean_meta

    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents (chunks) to the ChromaDB.
        Automatically serializes metadata.
        """
        try:
            # Prepare docs with clean metadatas
            clean_docs = []
            for doc in documents:
                clean_meta = self._serialize_metadata(doc.metadata)
                clean_docs.append(Document(page_content=doc.page_content, metadata=clean_meta))
                
            # Add to DB
            self.db.add_documents(clean_docs)
            # Persist changes to disk so other instances can see them
            if hasattr(self.db, "persist"):
                try:
                    self.db.persist()
                    logger.info("ChromaDB persisted to storage.")
                except Exception as persist_err:
                    logger.warning(f"ChromaDB persist failed: {persist_err}")
            logger.info(f"Successfully added {len(clean_docs)} document chunks to ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return False

    def similarity_search(
        self,
        query: str,
        paper_ids: Optional[List[str]] = None,
        k: int = 5
    ) -> List[Document]:
        """
        Perform similarity search on the collection.
        Allows filtering by specific paper IDs.
        """
        try:
            filter_dict = None
            if paper_ids and len(paper_ids) > 0:
                if len(paper_ids) == 1:
                    filter_dict = {"paper_id": paper_ids[0]}
                else:
                    filter_dict = {"paper_id": {"$in": paper_ids}}
                    
            raw_results = self.db.similarity_search(query, k=k, filter=filter_dict)
            
            # Deserialize metadatas
            deserialized_results = []
            for doc in raw_results:
                clean_meta = self._deserialize_metadata(doc.metadata)
                deserialized_results.append(Document(page_content=doc.page_content, metadata=clean_meta))
                
            return deserialized_results
        except Exception as e:
            logger.error(f"Error executing similarity search: {e}")
            return []

    def delete_paper(self, paper_id: str) -> bool:
        """Delete all document chunks corresponding to the paper_id."""
        try:
            # Delete using collection direct access
            self.collection.delete(where={"paper_id": paper_id})
            logger.info(f"Successfully deleted paper with ID: {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting paper {paper_id}: {e}")
            return False

    def paper_exists(self, paper_id: str) -> bool:
        """Check if a paper is already indexed in the vector database."""
        try:
            results = self.collection.get(where={"paper_id": paper_id}, limit=1)
            return len(results.get("ids", [])) > 0
        except Exception as e:
            logger.error(f"Error checking if paper exists: {e}")
            return False

    def get_paper_metadata(self, paper_id: str) -> Optional[dict]:
        """Retrieve metadata for a specific indexed paper."""
        try:
            results = self.collection.get(where={"paper_id": paper_id}, limit=1)
            metadatas = results.get("metadatas", [])
            if metadatas:
                return self._deserialize_metadata(metadatas[0])
            return None
        except Exception as e:
            logger.error(f"Error fetching paper metadata: {e}")
            return None

    def get_all_papers(self) -> List[dict]:
        """Fetch all unique indexed papers from the collection."""
        try:
            results = self.collection.get()
            papers_map = {}
            for meta in results.get("metadatas", []):
                pid = meta.get("paper_id")
                if pid and pid not in papers_map:
                    deserialized = self._deserialize_metadata(meta)
                    papers_map[pid] = {
                        "paper_id": pid,
                        "title": deserialized.get("title", "Unknown"),
                        "authors": deserialized.get("authors", []),
                        "year": deserialized.get("year"),
                        "url": deserialized.get("url", ""),
                        "doi": deserialized.get("doi"),
                        "arxiv_id": deserialized.get("arxiv_id"),
                        "open_access_url": deserialized.get("open_access_url")
                    }
            return list(papers_map.values())
        except Exception as e:
            logger.error(f"Error fetching all papers: {e}")
            return []
