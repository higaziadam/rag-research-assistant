from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List


@dataclass
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    artifacts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts")
    results_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "results")
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_local_files_only: bool = field(default_factory=lambda: os.getenv("MODEL_LOCAL_FILES_ONLY", "true").lower() == "true")
    faiss_index_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "faiss_index.index")
    metadata_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "metadata.jsonl")
    documents_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "documents.json")
    jobs_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "jobs.json")
    uploads_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "uploads")
    max_upload_bytes: int = 100 * 1024 * 1024
    max_upload_files: int = 10
    ingestion_worker_count: int = 1
    max_session_history: int = 10
    max_session_count: int = 100
    retrieval_candidate_k: int = 30
    rerank_candidate_k: int = 12
    reranker_batch_size: int = 16
    summary_candidate_k: int = 80
    summary_rerank_candidate_k: int = 32
    max_answer_sentences: int = 7
    max_answer_sentence_characters: int = 750
    math_ocr_enabled: bool = field(default_factory=lambda: os.getenv("MATH_OCR_ENABLED", "true").lower() == "true")
    math_ocr_checkpoint: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "MATH_OCR_CHECKPOINT",
                str(Path(__file__).resolve().parents[2] / "artifacts" / "math_ocr" / "checkpoints" / "weights.pth"),
            )
        )
    )
    max_equations_per_page: int = 4
    max_ocr_equations_per_document: int = 50
    text_chunk_characters: int = 650
    text_chunk_overlap_characters: int = 120
    max_table_characters: int = 8_000
    cors_origins: List[str] = field(
        default_factory=lambda: [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]
    )


settings = Settings()
