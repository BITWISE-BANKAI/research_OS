from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PaperMetadataSchema(BaseModel):
    paper_id: str
    title: str
    abstract: Optional[str] = "N/A"
    authors: List[str] = []
    year: Optional[int] = None
    citation_count: int = 0
    influential_citation_count: int = 0
    fields_of_study: List[str] = []
    url: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    open_access_url: Optional[str] = None

class SearchQueryRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    papers: List[PaperMetadataSchema]

class IndexPaperRequest(BaseModel):
    paper_id: Optional[str] = Field(
        default=None,
        description="Semantic Scholar paper_id or arXiv ID (e.g. '1706.03762')",
        json_schema_extra={"example": "1706.03762"}
    )
    paper: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Complete paper metadata dictionary if indexing without Semantic Scholar API",
        json_schema_extra={"example": {
            "paper_id": "manual_001",
            "title": "Attention Is All You Need",
            "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "url": "https://arxiv.org/abs/1706.03762"
        }}
    )

class ManualPaperRequest(BaseModel):
    title: str = Field(..., description="Title of the research paper", json_schema_extra={"example": "Attention Is All You Need"})
    abstract: Optional[str] = Field("No abstract provided.", description="Paper abstract or summary", json_schema_extra={"example": "We propose the Transformer architecture based solely on attention mechanisms."})
    authors: Optional[List[str]] = Field(default_factory=lambda: ["Unknown Author"], description="List of author names", json_schema_extra={"example": ["Ashish Vaswani", "Noam Shazeer"]})
    year: Optional[int] = Field(2024, description="Publication year", json_schema_extra={"example": 2017})
    content: Optional[str] = Field(None, description="Full text or section paragraphs to chunk and embed", json_schema_extra={"example": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. The Transformer uses self-attention."})
    paper_id: Optional[str] = Field(None, description="Optional custom paper ID", json_schema_extra={"example": "manual_1706_03762"})
    url: Optional[str] = Field(None, description="Optional reference URL", json_schema_extra={"example": "https://arxiv.org/abs/1706.03762"})



class ChatRequest(BaseModel):
    query: str
    mode: str = "library" # "single", "library", "compare"
    paper_ids: Optional[List[str]] = None
    k: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    status: Optional[str] = None

class CompareRequest(BaseModel):
    paper_ids: List[str]
    query: Optional[str] = "Compare methodologies, datasets, findings, and limitations."

class DeepResearchRequest(BaseModel):
    query: str
    paper_ids: List[str]
    max_iterations: int = 3
    score_threshold: int = 8
