"""Pydantic schemas that define the public FastAPI contract."""

from typing import Dict, List, Literal, Optional

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
    status: Literal["queued", "extracting", "embedding", "indexed", "failed", "cancelled"] = "indexed"
    progress: int = Field(default=100, ge=0, le=100)
    message: str = "Indexed."
    error: Optional[str] = None
    job_id: Optional[str] = None


class IngestionJobResponse(BaseModel):
    job_id: str
    filename: str
    status: Literal["queued", "extracting", "embedding", "indexed", "failed", "cancelled"]
    progress: int = Field(ge=0, le=100)
    message: str
    created_at: str
    updated_at: str
    error: Optional[str] = None


class UploadResponse(BaseModel):
    uploaded: List[str]
    total_chunks: int
    documents: List[DocumentInfo]
    jobs: List[IngestionJobResponse]


class DeleteDocumentResponse(BaseModel):
    deleted: str
    documents: List[DocumentInfo]


class EquationResponse(BaseModel):
    latex: str = ""
    status: Literal["needs_verification", "source_only"]
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: List[float] = Field(min_length=4, max_length=4)


class SourceResponse(BaseModel):
    chunk_id: str
    score: float
    source: str
    page: int
    text: str
    table: str = ""
    figure_caption: str = ""
    section: str = ""
    equations: List[EquationResponse] = Field(default_factory=list)


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


class MathOcrStatus(BaseModel):
    enabled: bool
    checkpoint_available: bool
    checkpoint_path: str
    mode: str
