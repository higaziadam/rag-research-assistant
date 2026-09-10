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
    model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
        or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    # Set RERANKER_REVISION to an empty string when loading a locally trained
    # checkpoint. Hugging Face revisions apply to remote repositories only.
    reranker_revision: str | None = field(
        default_factory=lambda: os.getenv("RERANKER_REVISION", "233902d25c440f23af6f7d6e94d2946bac0bee0a").strip() or None
    )
    model_local_files_only: bool = field(default_factory=lambda: os.getenv("MODEL_LOCAL_FILES_ONLY", "true").lower() == "true")
    faiss_index_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "faiss_index.index")
    metadata_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "metadata.jsonl")
    documents_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "documents.json")
    jobs_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "jobs.json")
    uploads_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "uploads")
    max_upload_bytes: int = 100 * 1024 * 1024
    max_upload_batch_bytes: int = 250 * 1024 * 1024
    max_upload_files: int = 10
    max_filename_characters: int = 180
    max_pdf_pages: int = 2_500
    max_pdf_page_area_points: float = 25_000_000.0
    max_preview_pixels: int = 12_000_000
    ingestion_worker_count: int = 1
    inference_concurrency: int = field(default_factory=lambda: max(1, int(os.getenv("INFERENCE_CONCURRENCY", "1"))))
    max_terminal_jobs: int = field(default_factory=lambda: max(10, int(os.getenv("MAX_TERMINAL_JOBS", "250"))))
    max_session_history: int = 10
    max_session_count: int = 100
    # Keep enough dense and lexical candidates for the cross-encoder to
    # recover direct evidence from long documents before truncating to top-k.
    retrieval_candidate_k: int = 60
    sparse_candidate_k: int = 60
    reciprocal_rank_fusion_constant: int = 60
    rerank_candidate_k: int = 30
    hybrid_dense_backfill_k: int = 30
    reranker_batch_size: int = 16
    summary_candidate_k: int = 80
    summary_rerank_candidate_k: int = 32
    max_answer_sentences: int = 7
    max_answer_sentence_characters: int = 750
    # Synthesis is opt-in so a missing local model never prevents grounded
    # retrieval from returning an answer. Set ANSWER_SYNTHESIS_ENABLED=true
    # after Ollama and the configured model are available locally.
    answer_synthesis_enabled: bool = field(
        default_factory=lambda: os.getenv("ANSWER_SYNTHESIS_ENABLED", "false").lower() == "true"
    )
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    answer_synthesis_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("ANSWER_SYNTHESIS_TIMEOUT_SECONDS", "20"))
    )
    answer_synthesis_max_evidence: int = field(
        default_factory=lambda: int(os.getenv("ANSWER_SYNTHESIS_MAX_EVIDENCE", "7"))
    )
    answer_synthesis_max_evidence_characters: int = field(
        default_factory=lambda: int(os.getenv("ANSWER_SYNTHESIS_MAX_EVIDENCE_CHARACTERS", "1400"))
    )
    allow_remote_synthesis: bool = field(
        default_factory=lambda: os.getenv("ALLOW_REMOTE_SYNTHESIS", "false").lower() == "true"
    )
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
    api_key: str = field(default_factory=lambda: os.getenv("RAG_API_KEY", "").strip())


settings = Settings()
