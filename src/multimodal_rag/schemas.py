"""Pydantic schemas that define the public FastAPI contract."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str = Field(default="default", min_length=1, max_length=100)
    history: List[str] = Field(default_factory=list, max_length=10)


class DocumentInfo(BaseModel):
    filename: str
    pages: int
    chunks: int
    file_size_bytes: Optional[int] = None


class UploadResponse(BaseModel):
    uploaded: List[str]
    total_chunks: int
    documents: List[DocumentInfo]


class DeleteDocumentResponse(BaseModel):
    deleted: str
    documents: List[DocumentInfo]


class SourceResponse(BaseModel):
    chunk_id: str
    score: float
    source: str
    page: int
    text: str
    table: str = ""
    figure_caption: str = ""
    section: str = ""


class QueryResponse(BaseModel):
    answer: str
    unsupported: bool
    confidence: float
    sources: List[SourceResponse]
    retrieval_scores: List[float]
    latency_ms: float
    session_id: str
    history: List[str]


class EvaluationSummary(BaseModel):
    recall_at_5: float
    mrr: float
    citation_accuracy: float
    answer_faithfulness: float
    latency_ms: float
    comparison: Dict[str, Dict[str, float]]
