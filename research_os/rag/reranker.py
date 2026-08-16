from typing import List
from langchain_core.documents import Document
from utils.logging import setup_logger

logger = setup_logger("Reranker")

class Reranker:
    """
    Reranker stub interface.
    Currently acts as a pass-through, but designed to allow easy integration
    of BGE-Reranker, Cohere, or cross-encoders later.
    """
    def __init__(self, model_name: str = "none"):
        self.model_name = model_name
        logger.info("Reranker initialized (Pass-Through mode).")

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """
        Rerank a list of documents relative to the search query.
        For MVP, simply returns the top_n documents directly.
        """
        logger.info(f"Reranking skipped: returning first {top_n} out of {len(documents)} docs.")
        return documents[:top_n]
