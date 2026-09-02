from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    artifacts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts")
    results_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "results")
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    faiss_index_path: str = "artifacts/faiss_index.index"
    metadata_path: str = "artifacts/metadata.jsonl"
    max_context_tokens: int = 1500


settings = Settings()
