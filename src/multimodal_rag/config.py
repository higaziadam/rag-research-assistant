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
    faiss_index_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "faiss_index.index")
    metadata_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "metadata.jsonl")
    documents_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "documents.json")
    uploads_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "uploads")
    max_upload_bytes: int = 25 * 1024 * 1024
    max_upload_files: int = 10
    max_session_history: int = 10
    max_answer_sentences: int = 7
    max_answer_sentence_characters: int = 750
    max_context_tokens: int = 1500
    cors_origins: List[str] = field(
        default_factory=lambda: [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]
    )


settings = Settings()
