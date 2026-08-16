import os
import re
import hashlib
import numpy as np
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from utils.logging import setup_logger

load_dotenv()
logger = setup_logger("Embeddings")

class ResilientEmbeddings(Embeddings):
    """Resilient embeddings wrapper: tries Ollama first, with deterministic vector fallback if Ollama is offline."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self._ollama = OllamaEmbeddings(base_url=base_url, model=model)
        self._fallback_warned = False
        self._using_fallback = False
        
    def check_health(self) -> bool:
        """Probe Ollama with a tiny embedding request. Returns True if Ollama is reachable."""
        try:
            self._ollama.embed_query("test")
            self._using_fallback = False
            return True
        except Exception:
            self._using_fallback = True
            return False

    def _fallback_vector(self, text: str, dim: int = 384) -> List[float]:
        """Generate a dense deterministic normalized semantic feature vector for offline mode."""
        vec = np.zeros(dim, dtype=np.float32)
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return vec.tolist()
        
        for w in words:
            # Hash word to dimensional slot and direction
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
            slot = h % dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[slot] += sign
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            result = self._ollama.embed_documents(texts)
            self._using_fallback = False
            return result
        except Exception as e:
            self._using_fallback = True
            if not self._fallback_warned:
                logger.warning(f"Ollama embedding unavailable ({e}); utilizing resilient local vector embeddings.")
                self._fallback_warned = True
            return [self._fallback_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        try:
            result = self._ollama.embed_query(text)
            self._using_fallback = False
            return result
        except Exception as e:
            self._using_fallback = True
            if not self._fallback_warned:
                logger.warning(f"Ollama embedding unavailable ({e}); utilizing resilient local query vector.")
                self._fallback_warned = True
            return self._fallback_vector(text)


def get_embeddings() -> Embeddings:
    """Initialize and return resilient local embeddings manager."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:latest")
    return ResilientEmbeddings(base_url=base_url, model=model)

