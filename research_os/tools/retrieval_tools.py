from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from rag.chroma_store import ChromaStore
from utils.logging import setup_logger

logger = setup_logger("RetrievalTool")

@tool
def retrieve_evidence(query: str, paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Retrieves the top evidence chunks from the indexed database for a given query.
    Optionally filters search to specific paper_ids.
    """
    try:
        store = ChromaStore()
        # Retrieve chunks (k=5 by default)
        docs = store.similarity_search(query, paper_ids=paper_ids, k=5)
        
        results = []
        for doc in docs:
            results.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
            
        logger.info(f"Retrieved {len(results)} evidence chunks for query '{query}'")
        return results
    except Exception as e:
        logger.error(f"Error in retrieve_evidence tool: {e}")
        return [{"error": f"Failed to retrieve evidence: {str(e)}"}]
