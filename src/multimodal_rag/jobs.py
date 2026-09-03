"""Persistent ingestion job records for background document processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

JobStatus = Literal["queued", "extracting", "embedding", "indexed", "failed", "cancelled"]
ACTIVE_JOB_STATUSES = {"queued", "extracting", "embedding"}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IngestionJob:
    job_id: str
    filename: str
    status: JobStatus
    progress: int
    message: str
    created_at: str
    updated_at: str
    error: str | None = None

    @classmethod
    def create(cls, filename: str) -> "IngestionJob":
        timestamp = utc_timestamp()
        return cls(
            job_id=uuid4().hex,
            filename=filename,
            status="queued",
            progress=0,
            message="Queued for indexing.",
            created_at=timestamp,
            updated_at=timestamp,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IngestionJob":
        return cls(
            job_id=str(payload["job_id"]),
            filename=str(payload["filename"]),
            status=payload["status"],
            progress=int(payload.get("progress", 0)),
            message=str(payload.get("message", "")),
            created_at=str(payload.get("created_at", utc_timestamp())),
            updated_at=str(payload.get("updated_at", utc_timestamp())),
            error=payload.get("error"),
        )

    def update(self, status: JobStatus, progress: int, message: str, error: str | None = None) -> None:
        self.status = status
        self.progress = max(0, min(progress, 100))
        self.message = message
        self.error = error
        self.updated_at = utc_timestamp()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
