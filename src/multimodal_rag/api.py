from __future__ import annotations

import copy
import hmac
import json
import logging
import math
import os
import re
import shutil
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from threading import BoundedSemaphore, Lock, RLock
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .config import settings
from .data_models import DocumentChunk, RetrievalResult
from .document_extraction import StructuredPdfExtractor, render_pdf_region, validate_pdf
from .embeddings import EmbeddingStore
from .math_extraction import LocalMathExtractor
from .jobs import ACTIVE_JOB_STATUSES, IngestionJob
from .retrieval import FAISSRetriever
from .reranker import Reranker
from .sparse_retrieval import BM25Retriever, reciprocal_rank_fusion
from .synthesis import OllamaSynthesisClient
from .schemas import (
    DeleteDocumentResponse,
    DocumentInfo,
    EvaluationSummary,
    IngestionJobResponse,
    MathOcrStatus,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self._recover_interrupted_state()
        self.embedding_store = EmbeddingStore(
            settings.model_name,
            local_files_only=settings.model_local_files_only,
            revision=settings.model_revision,
        )
        self.retriever = FAISSRetriever(embedding_dim=self.embedding_store.embedding_dimension)
        self.sparse_retriever = BM25Retriever([])
        self.reranker = Reranker(
            settings.reranker_model,
            local_files_only=settings.model_local_files_only,
            batch_size=settings.reranker_batch_size,
            revision=settings.reranker_revision,
        )
        self.synthesizer = OllamaSynthesisClient(
            enabled=settings.answer_synthesis_enabled,
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.answer_synthesis_timeout_seconds,
            max_evidence=settings.answer_synthesis_max_evidence,
            max_evidence_characters=settings.answer_synthesis_max_evidence_characters,
            allow_remote=settings.allow_remote_synthesis,
        )
        self.math_extractor = LocalMathExtractor(
            enabled=settings.math_ocr_enabled,
            checkpoint_path=settings.math_ocr_checkpoint,
            max_equations_per_page=settings.max_equations_per_page,
            max_ocr_equations_per_document=settings.max_ocr_equations_per_document,
        )
        self.pdf_extractor = StructuredPdfExtractor(max_table_characters=settings.max_table_characters)
        self.documents: List[Dict[str, Any]] = []
        self.jobs: Dict[str, IngestionJob] = self._load_persisted_jobs()
        self.job_executor = ThreadPoolExecutor(
            max_workers=settings.ingestion_worker_count,
            thread_name_prefix="document-ingestion",
        )
        self.job_futures: Dict[str, Future[None]] = {}
        self.session_history: Dict[str, List[str]] = defaultdict(list)
        self.storage_lock = RLock()
        self.inference_gate = BoundedSemaphore(settings.inference_concurrency)
        if self._has_persisted_index():
            self._restore_persisted_state()
        elif settings.documents_path.exists():
            self._restore_documents_without_index()
        else:
            self._bootstrap_demo_docs()
        self._rebuild_sparse_retriever()
        self._resume_pending_jobs()

    def close(self) -> None:
        """Release background workers during graceful application shutdown."""
        executor = getattr(self, "job_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _has_persisted_index() -> bool:
        index_exists = settings.faiss_index_path.exists()
        metadata_exists = settings.metadata_path.exists()
        if index_exists != metadata_exists:
            raise RuntimeError("Persistent index is incomplete: both FAISS index and metadata files are required.")
        return index_exists

    def _restore_persisted_state(self) -> None:
        self.retriever = FAISSRetriever(
            embedding_dim=self.embedding_store.embedding_dimension,
            index_path=str(settings.faiss_index_path),
            metadata_path=str(settings.metadata_path),
        )
        if settings.documents_path.exists():
            documents = json.loads(settings.documents_path.read_text(encoding="utf-8"))
            if not isinstance(documents, list):
                raise RuntimeError("Persistent document manifest must contain a list of documents.")
            self.documents = documents
        else:
            self.documents = self._documents_from_chunks()

    def _restore_documents_without_index(self) -> None:
        """Recover queued uploads even if a previous run stopped before index persistence."""
        documents = json.loads(settings.documents_path.read_text(encoding="utf-8"))
        if not isinstance(documents, list):
            raise RuntimeError("Persistent document manifest must contain a list of documents.")
        self.documents = documents
        for document in self.documents:
            if document.get("status", "indexed") == "indexed":
                document.update(
                    {
                        "status": "failed",
                        "progress": 0,
                        "message": "The saved search index is unavailable. Retry to rebuild this document.",
                        "error": "Persistent search index is unavailable.",
                    }
                )

    def _documents_from_chunks(self) -> List[Dict[str, Any]]:
        documents: Dict[str, Dict[str, Any]] = {}
        for chunk in self.retriever.chunks:
            document = documents.setdefault(
                chunk.source,
                {
                    "filename": chunk.source,
                    "pages": 0,
                    "chunks": 0,
                    "status": "indexed",
                    "progress": 100,
                    "message": "Indexed.",
                },
            )
            document["pages"] = max(document["pages"], int(chunk.metadata.get("page", 0)))
            document["chunks"] += 1
        return list(documents.values())

    @staticmethod
    def _load_persisted_jobs() -> Dict[str, IngestionJob]:
        if not settings.jobs_path.exists():
            return {}
        payload = json.loads(settings.jobs_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError("Persistent ingestion job manifest must contain a list of jobs.")
        return {job.job_id: job for item in payload if (job := IngestionJob.from_dict(item))}

    def _persist_documents(self, documents: List[Dict[str, Any]] | None = None) -> None:
        settings.documents_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = settings.documents_path.with_suffix(f"{settings.documents_path.suffix}.tmp")
        temporary_manifest.write_text(json.dumps(self.documents if documents is None else documents, indent=2), encoding="utf-8")
        temporary_manifest.replace(settings.documents_path)
        self._restrict_file_permissions(settings.documents_path)

    def _prune_terminal_jobs(
        self,
        jobs: Dict[str, IngestionJob] | None = None,
        documents: List[Dict[str, Any]] | None = None,
    ) -> None:
        target_jobs = self.jobs if jobs is None else jobs
        target_documents = self.documents if documents is None else documents
        protected_ids = {
            str(document.get("job_id"))
            for document in target_documents
            if document.get("job_id")
        }
        terminal = sorted(
            (
                job
                for job in target_jobs.values()
                if job.status not in ACTIVE_JOB_STATUSES and job.job_id not in protected_ids
            ),
            key=lambda job: job.updated_at,
            reverse=True,
        )
        for job in terminal[settings.max_terminal_jobs :]:
            target_jobs.pop(job.job_id, None)

    def _persist_jobs(
        self,
        jobs: Dict[str, IngestionJob] | None = None,
        documents: List[Dict[str, Any]] | None = None,
    ) -> None:
        settings.jobs_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_jobs = settings.jobs_path.with_suffix(f"{settings.jobs_path.suffix}.tmp")
        persisted_jobs = getattr(self, "jobs", {}) if jobs is None else jobs
        self._prune_terminal_jobs(persisted_jobs, documents)
        serialized_jobs = [job.to_dict() for job in persisted_jobs.values()]
        temporary_jobs.write_text(json.dumps(serialized_jobs, indent=2), encoding="utf-8")
        temporary_jobs.replace(settings.jobs_path)
        self._restrict_file_permissions(settings.jobs_path)

    @staticmethod
    def _restrict_file_permissions(path: Path) -> None:
        try:
            path.chmod(0o600)
        except OSError:
            logger.warning("Could not restrict filesystem permissions for %s", path)

    @staticmethod
    def _state_paths() -> List[Path]:
        return [settings.faiss_index_path, settings.metadata_path, settings.documents_path, settings.jobs_path]

    @classmethod
    def _recover_interrupted_state(cls) -> None:
        marker = settings.artifacts_dir / "state-transaction.json"
        if not marker.exists():
            return
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            existed = payload.get("existed") if isinstance(payload, dict) else None
            expected_names = {path.name for path in cls._state_paths()}
            if not isinstance(existed, dict) or set(existed) != expected_names:
                raise ValueError("The transaction marker is malformed.")
            for path in cls._state_paths():
                backup = path.with_name(f".{path.name}.rollback")
                if bool(existed.get(path.name)) and backup.exists():
                    os.replace(backup, path)
                elif not bool(existed.get(path.name)):
                    path.unlink(missing_ok=True)
                backup.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
        except Exception:
            logger.exception("Could not recover an interrupted persistent-state transaction")
            raise RuntimeError("Persistent state recovery failed; preserve the artifacts directory for inspection.")

    def _persist_state(
        self,
        *,
        retriever: FAISSRetriever | None = None,
        documents: List[Dict[str, Any]] | None = None,
        jobs: Dict[str, IngestionJob] | None = None,
    ) -> None:
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        persisted_retriever = self.retriever if retriever is None else retriever
        persisted_documents = self.documents if documents is None else documents
        persisted_jobs = getattr(self, "jobs", {}) if jobs is None else jobs
        paths = self._state_paths()
        marker = settings.artifacts_dir / "state-transaction.json"
        marker_tmp = marker.with_suffix(".tmp")
        existed = {path.name: path.exists() for path in paths}
        for path in paths:
            backup = path.with_name(f".{path.name}.rollback")
            backup.unlink(missing_ok=True)
            if path.exists():
                try:
                    os.link(path, backup)
                except OSError:
                    shutil.copy2(path, backup)
        marker_tmp.write_text(json.dumps({"existed": existed}), encoding="utf-8")
        os.replace(marker_tmp, marker)
        try:
            persisted_retriever.save(str(settings.faiss_index_path), str(settings.metadata_path))
            self._persist_documents(persisted_documents)
            self._persist_jobs(persisted_jobs, persisted_documents)
            self._restrict_file_permissions(settings.faiss_index_path)
            self._restrict_file_permissions(settings.metadata_path)
        except Exception:
            self._recover_interrupted_state()
            raise
        else:
            marker.unlink(missing_ok=True)
            for path in paths:
                path.with_name(f".{path.name}.rollback").unlink(missing_ok=True)

    def _rebuild_sparse_retriever(self) -> None:
        """Rebuild the local BM25 postings from the authoritative chunk list."""
        self.sparse_retriever = BM25Retriever(self.retriever.chunks)

    @staticmethod
    def _persist_upload(filename: str, data: bytes) -> None:
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        try:
            settings.uploads_dir.chmod(0o700)
        except OSError:
            logger.warning("Could not restrict filesystem permissions for %s", settings.uploads_dir)
        destination = settings.uploads_dir / filename
        temporary = destination.with_suffix(f"{destination.suffix}.uploading")
        temporary.write_bytes(data)
        temporary.replace(destination)
        RAGService._restrict_file_permissions(destination)

    @staticmethod
    def _validated_upload_filename(raw_filename: str | None) -> str:
        filename = (raw_filename or "").strip()
        if not filename or filename != Path(filename).name:
            raise HTTPException(status_code=400, detail="The upload contains an invalid filename.")
        if len(filename) > settings.max_filename_characters:
            raise HTTPException(status_code=400, detail="The PDF filename is too long.")
        if not filename.casefold().endswith(".pdf"):
            raise HTTPException(status_code=415, detail=f"{filename} is not a PDF file.")
        if filename.endswith((".", " ")) or any(ord(character) < 32 for character in filename):
            raise HTTPException(status_code=400, detail="The PDF filename contains unsupported characters.")
        if any(character in filename for character in '<>:"/\\|?*'):
            raise HTTPException(status_code=400, detail="The PDF filename contains unsupported characters.")
        reserved_names = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
        if Path(filename).stem.casefold() in reserved_names:
            raise HTTPException(status_code=400, detail="The PDF filename is reserved by the operating system.")
        return filename

    @staticmethod
    def _validate_pdf_payload(filename: str, data: bytes) -> None:
        try:
            validate_pdf(
                data,
                max_pages=settings.max_pdf_pages,
                max_page_area_points=settings.max_pdf_page_area_points,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"{filename}: {exc}") from exc

    def _bootstrap_demo_docs(self):
        demo_chunks = [
            DocumentChunk(
                chunk_id="demo_001",
                text="The multimodal retrieval pipeline ingests text, tables, and figure captions from research documents before building a FAISS vector index.",
                table="| Method | Recall@5 |\n| --- | ---: |\n| Baseline | 0.58 |",
                figure_caption="Figure 1: Retrieval performance by chunking strategy.",
                source="technical_report.pdf",
                section="System Overview",
                metadata={"page": 2, "document_type": "technical_report"},
                type="text",
            ),
            DocumentChunk(
                chunk_id="demo_002",
                text="Dense retrieval with normalized embeddings provides high recall, while a cross-encoder reranker improves ranking quality and answer grounding.",
                table="| Method | MRR |\n| --- | ---: |\n| Dense + rerank | 0.68 |",
                figure_caption="Figure 2: Latency vs retrieval quality trade-off.",
                source="technical_report.pdf",
                section="Retrieval Methods",
                metadata={"page": 5, "document_type": "technical_report"},
                type="text",
            ),
            DocumentChunk(
                chunk_id="demo_003",
                text="LoRA fine-tuning of the reranker improves relevance ordering, citation faithfulness, and overall retrieval precision under domain-specific queries.",
                table="| Model | Recall@5 |\n| --- | ---: |\n| LoRA reranker | 0.84 |",
                figure_caption="Figure 3: Effect of PEFT reranking on technical QA accuracy.",
                source="technical_report.pdf",
                section="Fine-tuning",
                metadata={"page": 10, "document_type": "technical_report"},
                type="text",
            ),
        ]
        self._index_chunks(demo_chunks)
        self.documents.append(
            {
                "filename": "technical_report.pdf",
                "pages": 10,
                "chunks": len(demo_chunks),
                "status": "indexed",
                "progress": 100,
                "message": "Indexed.",
            }
        )

    def _split_text(self, text: str, chunk_length: int | None = None) -> List[str]:
        chunk_length = chunk_length or settings.text_chunk_characters
        pieces: List[str] = []
        paragraph_blocks = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        for block in paragraph_blocks:
            sentences = re.split(r"(?<=[.!?])\s+", block)
            current = ""
            for sentence in sentences:
                if len((current + " " + sentence).strip()) <= chunk_length:
                    current = (current + " " + sentence).strip()
                else:
                    if current:
                        pieces.append(current)
                    current = ""
                    while len(sentence) > chunk_length:
                        split_at = sentence.rfind(" ", 0, chunk_length + 1)
                        split_at = split_at if split_at > 0 else chunk_length
                        pieces.append(sentence[:split_at].strip())
                        sentence = sentence[split_at:].strip()
                    current = sentence
            if current:
                pieces.append(current)
        if len(pieces) < 2 or settings.text_chunk_overlap_characters <= 0:
            return pieces or [text[:chunk_length]]

        overlapped_pieces = [pieces[0]]
        for piece in pieces[1:]:
            previous_tail = overlapped_pieces[-1][-settings.text_chunk_overlap_characters :].split(" ", 1)[-1]
            overlapped_pieces.append(f"{previous_tail} {piece}".strip())
        return overlapped_pieces

    def _parse_pdf_to_chunks(self, filename: str, file_bytes: bytes) -> List[DocumentChunk]:
        validate_pdf(
            file_bytes,
            max_pages=settings.max_pdf_pages,
            max_page_area_points=settings.max_pdf_page_area_points,
        )
        chunks: List[DocumentChunk] = []
        layout_pages = self.pdf_extractor.extract_pages(file_bytes, equation_extractor=self.math_extractor)
        for layout_page in layout_pages:
            for element_index, element in enumerate(layout_page.elements):
                text_parts = [element.text] if element.type != "text" else self._split_text(element.text)
                for part_index, text in enumerate(text_parts):
                    chunk_id = f"{filename}-{layout_page.page_number}-{element.type}-{element_index}-{part_index}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            text=text,
                            table=element.table,
                            figure_caption=element.figure_caption,
                            source=filename,
                            section=element.section,
                            metadata={
                                "page": layout_page.page_number,
                                "document_type": "pdf",
                                "bounding_box": element.bounding_box,
                                "quality_flags": element.quality_flags,
                            },
                            equations=self._equations_for_element(layout_page.equations, element.bounding_box),
                            type=element.type,
                        )
                    )
            for equation_index, equation in enumerate(layout_page.equations):
                bounding_box = equation.get("bounding_box", [])
                if len(bounding_box) != 4:
                    continue
                latex = str(equation.get("latex", "")).strip()
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{filename}-{layout_page.page_number}-equation-{equation_index}",
                        text=(f"Mathematical expression: {latex}" if latex else "Mathematical expression in the cited PDF."),
                        source=filename,
                        metadata={
                            "page": layout_page.page_number,
                            "document_type": "pdf",
                            "bounding_box": bounding_box,
                            "quality_flags": ["verify_original_math_notation"],
                        },
                        equations=[equation],
                        type="equation",
                    )
                )
        return chunks

    @staticmethod
    def _equations_for_element(equations: List[Dict[str, Any]], bounding_box: List[float]) -> List[Dict[str, Any]]:
        """Attach overlapping or immediately-following equations to explanatory prose."""
        x0, y0, x1, y1 = bounding_box
        matching_equations = []
        for equation in equations:
            equation_box = equation.get("bounding_box", [])
            if len(equation_box) != 4:
                continue
            ex0, ey0, ex1, ey1 = equation_box
            overlaps = max(x0, ex0) <= min(x1, ex1) and max(y0, ey0) <= min(y1, ey1)
            # Display equations are often centered or indented, so their
            # horizontal bounds need not overlap the preceding paragraph.
            follows_explanation = 0 <= ey0 - y1 <= 72
            if overlaps or follows_explanation:
                matching_equations.append(equation)
        return matching_equations

    def _index_chunks(self, chunks: List[DocumentChunk]) -> None:
        searchable_chunks, embeddings = self._embed_chunks(chunks)
        if not searchable_chunks:
            return
        self.retriever.add_chunks(searchable_chunks, embeddings)
        self._rebuild_sparse_retriever()

    def _embed_chunks(self, chunks: List[DocumentChunk]) -> tuple[List[DocumentChunk], Any]:
        """Create embeddings without holding the storage lock.

        Embedding a large textbook can take minutes on CPU. Keeping it outside
        the lock allows new uploads and document-status requests to proceed.
        """
        searchable_chunks = [chunk for chunk in chunks if chunk.to_text_for_search()]
        if not searchable_chunks:
            return [], None
        gate = getattr(self, "inference_gate", nullcontext())
        with gate:
            embeddings = self.embedding_store.encode([chunk.to_text_for_search() for chunk in searchable_chunks])
        return searchable_chunks, embeddings

    @staticmethod
    def _content_counts(chunks: List[DocumentChunk]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for chunk in chunks:
            counts[chunk.type] += 1
        return dict(counts)

    def _document_for_filename(self, filename: str) -> Dict[str, Any] | None:
        return next((document for document in self.documents if document["filename"] == filename), None)

    def _update_job(
        self,
        job_id: str,
        status: str,
        progress: int,
        message: str,
        error: str | None = None,
    ) -> bool:
        """Persist a job update and mirror it on the matching document record."""
        job = self.jobs.get(job_id)
        if job is None or job.status == "cancelled":
            return False
        job.update(status=status, progress=progress, message=message, error=error)
        document = self._document_for_filename(job.filename)
        if document is not None:
            document.update(
                {
                    "status": job.status,
                    "progress": job.progress,
                    "message": job.message,
                    "error": job.error,
                    "job_id": job.job_id,
                }
            )
        self._persist_documents()
        self._persist_jobs()
        return True

    def _schedule_job(self, job_id: str) -> None:
        future = self.job_futures.get(job_id)
        if future is None or future.done():
            future = self.job_executor.submit(self._run_ingestion_job, job_id)
            self.job_futures[job_id] = future

            def remove_completed_job(completed_future: Future[None]) -> None:
                with self.storage_lock:
                    if self.job_futures.get(job_id) is completed_future:
                        self.job_futures.pop(job_id, None)

            future.add_done_callback(remove_completed_job)

    def _resume_pending_jobs(self) -> None:
        """Resume uploads that were queued when the backend last stopped."""
        with self.storage_lock:
            for job in self.jobs.values():
                if job.status not in ACTIVE_JOB_STATUSES:
                    continue
                if (settings.uploads_dir / job.filename).is_file():
                    job.update(status="queued", progress=0, message="Queued after backend restart.")
                else:
                    job.update(status="failed", progress=0, message="Upload file is missing.", error="Upload file is missing.")
                    document = self._document_for_filename(job.filename)
                    if document is not None:
                        document.update({"status": "failed", "message": job.message, "error": job.error})
                        continue
                document = self._document_for_filename(job.filename)
                if document is not None:
                    document.update(
                        {
                            "status": job.status,
                            "progress": job.progress,
                            "message": job.message,
                            "error": job.error,
                            "job_id": job.job_id,
                        }
                    )
            self._persist_documents()
            self._persist_jobs()
            pending_job_ids = [job.job_id for job in self.jobs.values() if job.status == "queued"]
        for job_id in pending_job_ids:
            self._schedule_job(job_id)

    def _run_ingestion_job(self, job_id: str) -> None:
        try:
            with self.storage_lock:
                job = self.jobs.get(job_id)
                if job is None or job.status == "cancelled":
                    return
                filename = job.filename
                self._update_job(job_id, "extracting", 10, "Extracting text and equation regions.")

            file_bytes = (settings.uploads_dir / filename).read_bytes()
            chunks = self._parse_pdf_to_chunks(filename, file_bytes)
            if not chunks:
                raise ValueError("No extractable text was found. Scanned PDFs need OCR before upload.")

            with self.storage_lock:
                if not self._update_job(job_id, "embedding", 65, "Creating embeddings and updating the search index."):
                    return

            searchable_chunks, embeddings = self._embed_chunks(chunks)
            if not searchable_chunks:
                raise ValueError("No searchable text, table, figure, or equation evidence was found.")

            with self.storage_lock:
                job = self.jobs.get(job_id)
                if job is None or job.status == "cancelled":
                    return
                document = self._document_for_filename(filename)
                if document is None:
                    return
                staged_retriever = self.retriever.clone()
                staged_retriever.add_chunks(searchable_chunks, embeddings)
                staged_sparse_retriever = BM25Retriever(staged_retriever.chunks)
                staged_documents = copy.deepcopy(self.documents)
                staged_jobs = copy.deepcopy(self.jobs)
                staged_document = next(item for item in staged_documents if item["filename"] == filename)
                staged_job = staged_jobs[job_id]
                staged_document.update(
                    {
                        "pages": max((int(chunk.metadata.get("page", 0)) for chunk in chunks), default=0),
                        "chunks": len(chunks),
                        "content_counts": self._content_counts(chunks),
                    }
                )
                staged_job.update(status="indexed", progress=100, message="Indexed.")
                staged_document.update(
                    {
                        "status": "indexed",
                        "progress": 100,
                        "message": "Indexed.",
                        "error": None,
                        "job_id": staged_job.job_id,
                    }
                )
                self._persist_state(
                    retriever=staged_retriever,
                    documents=staged_documents,
                    jobs=staged_jobs,
                )
                self.retriever = staged_retriever
                self.sparse_retriever = staged_sparse_retriever
                self.documents = staged_documents
                self.jobs = staged_jobs
        except Exception as exc:  # pragma: no cover - depends on malformed external PDFs
            logger.exception("Document ingestion failed for job %s", job_id)
            with self.storage_lock:
                self._update_job(
                    job_id,
                    "failed",
                    0,
                    "Indexing failed. Review the backend log for the diagnostic details.",
                    error="The PDF could not be indexed safely.",
                )

    def upload_documents(self, files: List[UploadFile]) -> UploadResponse:
        prepared_uploads: List[tuple[str, bytes]] = []
        batch_size = 0
        submitted_names: set[str] = set()
        with self.storage_lock:
            existing_names = {document["filename"].casefold() for document in self.documents}
        for file in files:
            filename = self._validated_upload_filename(file.filename)
            canonical_name = filename.casefold()
            if canonical_name in existing_names:
                raise HTTPException(status_code=409, detail=f"{filename} is already indexed. Use a new filename to replace it.")
            if canonical_name in submitted_names:
                raise HTTPException(status_code=409, detail=f"{filename} appears more than once in this upload.")
            data = file.file.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"{filename} exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB upload limit.",
                )
            batch_size += len(data)
            if batch_size > settings.max_upload_batch_bytes:
                raise HTTPException(status_code=413, detail="The combined PDF upload is too large.")
            self._validate_pdf_payload(filename, data)
            submitted_names.add(canonical_name)
            prepared_uploads.append((filename, data))

        jobs: List[IngestionJob] = []
        with self.storage_lock:
            uploaded_names: List[str] = []
            indexed_filenames = {document["filename"].casefold() for document in self.documents}
            conflicts = [filename for filename, _ in prepared_uploads if filename.casefold() in indexed_filenames]
            if conflicts:
                raise HTTPException(status_code=409, detail=f"{conflicts[0]} is already indexed. Use a new filename to replace it.")
            previous_documents = copy.deepcopy(self.documents)
            previous_jobs = copy.deepcopy(self.jobs)
            persisted_uploads: List[Path] = []
            try:
                for filename, data in prepared_uploads:
                    if filename.casefold() in indexed_filenames:
                        # Defensive check if this method is changed to release
                        # the storage lock between individual files.
                        raise HTTPException(status_code=409, detail=f"{filename} is already indexed. Use a new filename.")
                    uploaded_names.append(filename)
                    indexed_filenames.add(filename.casefold())
                    self._persist_upload(filename, data)
                    persisted_uploads.append(settings.uploads_dir / filename)
                    job = IngestionJob.create(filename)
                    self.jobs[job.job_id] = job
                    jobs.append(job)
                    self.documents.append(
                        {
                            "filename": filename,
                            "pages": 0,
                            "chunks": 0,
                            "status": job.status,
                            "progress": job.progress,
                            "message": job.message,
                            "job_id": job.job_id,
                            "content_counts": {},
                        }
                    )
                self._persist_documents()
                self._persist_jobs()
            except Exception:
                self.documents = previous_documents
                self.jobs = previous_jobs
                for upload_path in persisted_uploads:
                    upload_path.unlink(missing_ok=True)
                self._persist_documents()
                self._persist_jobs()
                raise
            response_documents = self._documents_with_file_sizes(self.documents)

        for job in jobs:
            self._schedule_job(job.job_id)
        return UploadResponse(
            uploaded=uploaded_names,
            total_chunks=0,
            documents=response_documents,
            jobs=[job.to_dict() for job in jobs],
        )

    def list_documents(self) -> List[Dict[str, Any]]:
        return self._documents_with_file_sizes(self.documents)

    @staticmethod
    def _documents_with_file_sizes(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        documents_with_sizes = []
        for document in documents:
            filename = Path(document["filename"]).name
            upload_path = settings.uploads_dir / filename
            file_size_bytes = upload_path.stat().st_size if upload_path.is_file() else None
            public_document = {**document, "file_size_bytes": file_size_bytes}
            if public_document.get("error"):
                public_document["error"] = "The document could not be indexed. Review the backend log for details."
            documents_with_sizes.append(public_document)
        return documents_with_sizes

    @staticmethod
    def load_persisted_documents() -> List[Dict[str, Any]]:
        if not settings.documents_path.exists():
            return []
        documents = json.loads(settings.documents_path.read_text(encoding="utf-8"))
        if not isinstance(documents, list):
            raise RuntimeError("Persistent document manifest must contain a list of documents.")
        return RAGService._documents_with_file_sizes(documents)

    def delete_document(self, filename: str) -> Dict[str, Any]:
        """Remove a document from the index, manifest, and persisted uploads."""
        safe_filename = self._validated_upload_filename(filename)

        with self.storage_lock:
            document = self._document_for_filename(safe_filename)
            if document is None:
                raise HTTPException(status_code=404, detail=f"{safe_filename} was not found.")

            if document.get("status") in ACTIVE_JOB_STATUSES:
                job_id = document.get("job_id")
                job = self.jobs.get(job_id) if job_id else None
                if job is not None:
                    job.update(status="cancelled", progress=0, message="Cancelled and removed.")
                self.documents = [item for item in self.documents if item["filename"] != safe_filename]
                upload_path = settings.uploads_dir / safe_filename
                if upload_path.is_file():
                    upload_path.unlink()
                self._persist_documents()
                self._persist_jobs()
                return {"deleted": safe_filename, "documents": self._documents_with_file_sizes(self.documents)}

            previous_retriever = self.retriever
            previous_sparse_retriever = self.sparse_retriever
            previous_documents = self.documents
            remaining_documents = [document for document in self.documents if document["filename"] != safe_filename]
            filtered_retriever = self.retriever.without_sources({safe_filename})

            upload_path = settings.uploads_dir / safe_filename
            temporary_upload_path = upload_path.with_suffix(f"{upload_path.suffix}.deleting")
            if upload_path.exists():
                upload_path.replace(temporary_upload_path)

            self.retriever = filtered_retriever
            self._rebuild_sparse_retriever()
            self.documents = remaining_documents
            try:
                self._persist_state()
            except Exception:
                self.retriever = previous_retriever
                self.sparse_retriever = previous_sparse_retriever
                self.documents = previous_documents
                if temporary_upload_path.exists():
                    temporary_upload_path.replace(upload_path)
                raise

            if temporary_upload_path.exists():
                temporary_upload_path.unlink()

            return {"deleted": safe_filename, "documents": self._documents_with_file_sizes(self.documents)}

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self.storage_lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Ingestion job was not found.")
            payload = job.to_dict()
            if payload.get("error"):
                payload["error"] = "The document could not be indexed. Review the backend log for details."
            return payload

    def retry_document(self, filename: str) -> Dict[str, Any]:
        safe_filename = self._validated_upload_filename(filename)

        with self.storage_lock:
            document = self._document_for_filename(safe_filename)
            if document is None:
                raise HTTPException(status_code=404, detail=f"{safe_filename} was not found.")
            if document.get("status") != "failed":
                raise HTTPException(status_code=409, detail="Only failed documents can be retried.")
            if not (settings.uploads_dir / safe_filename).is_file():
                raise HTTPException(status_code=404, detail="The original PDF is not available for retry.")

            job = IngestionJob.create(safe_filename)
            self.jobs[job.job_id] = job
            document.update(
                {
                    "pages": 0,
                    "chunks": 0,
                    "status": job.status,
                    "progress": job.progress,
                    "message": job.message,
                    "error": None,
                    "job_id": job.job_id,
                }
            )
            self._persist_documents()
            self._persist_jobs()
        self._schedule_job(job.job_id)
        return job.to_dict()

    @staticmethod
    def _prepare_query(query: str, history: List[str]) -> str:
        """Use history only for a clear follow-up, not every independent question."""
        follow_up_terms = r"\b(?:it|they|them|that|those|this|these|former|latter|previous|above|same)\b"
        if history and re.search(follow_up_terms, query, re.IGNORECASE):
            prior = " ".join(history[-2:])
            return f"Previous questions: {prior}\nFollow-up question: {query}"
        return query

    @staticmethod
    def _format_citations(evidence: List[Dict[str, Any]]) -> str:
        pages_by_source: Dict[str, List[int]] = defaultdict(list)
        for item in evidence:
            pages_by_source[item["source"]].append(item["page"])
        labels = []
        for source, pages in pages_by_source.items():
            unique_pages = sorted(set(pages))
            page_label = f"p. {unique_pages[0]}" if len(unique_pages) == 1 else f"pp. {', '.join(map(str, unique_pages))}"
            labels.append(f"{source} ({page_label})")
        return "; ".join(labels)

    @staticmethod
    def _is_readable_prose(sentence: str) -> bool:
        """Reject PDF extraction fragments that are dominated by broken equation tokens."""
        untrusted_math_glyphs = "∫∑√≤≥≈≠±×÷⎛⎝⎞⎠〈〉‖"
        if any(glyph in sentence for glyph in untrusted_math_glyphs):
            return False
        if sentence.count("\n") >= 2 and re.search(r"[=+*/^]", sentence):
            return False

        tokens = sentence.split()
        if len(tokens) < 5:
            return False

        word_count = len(re.findall(r"[A-Za-z]{2,}", sentence))
        noisy_token_count = sum(
            bool(re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", token))
            or (len(re.sub(r"[^A-Za-z]", "", token)) == 1 and token.isalpha())
            for token in tokens
        )
        visible_characters = [character for character in sentence if not character.isspace()]
        letter_ratio = sum(character.isalpha() for character in visible_characters) / max(len(visible_characters), 1)

        return word_count >= 5 and noisy_token_count / len(tokens) <= 0.35 and letter_ratio >= 0.55

    @staticmethod
    def _extract_summary_sentences(
        query: str,
        evidence: List[Dict[str, Any]],
        max_sentence_characters: int,
    ) -> List[str]:
        query_terms = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
        candidates: list[tuple[int, int, int, str]] = []
        seen_sentences = set()
        for evidence_index, item in enumerate(evidence):
            item_candidates = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", item["text"])
                if (
                    len(sentence.strip()) >= 30
                    and sentence.strip().endswith((".", "!", "?"))
                    and RAGService._is_readable_prose(sentence.strip())
                )
            ]
            for position, sentence in enumerate(item_candidates):
                normalized = re.sub(r"\W+", " ", sentence.lower()).strip()
                if normalized in seen_sentences:
                    continue
                seen_sentences.add(normalized)
                sentence_terms = set(re.findall(r"[a-zA-Z]{3,}", sentence.lower()))
                relevance = len(query_terms.intersection(sentence_terms))
                phrase_bonus = int(any(" ".join(query.lower().split()[start : start + 2]) in sentence.lower() for start in range(len(query.split()) - 1)))
                candidates.append((relevance + phrase_bonus, -evidence_index, -position, sentence))

        summaries = []
        for _, _, _, sentence in sorted(candidates, key=lambda candidate: candidate[:3], reverse=True):
            if len(summaries) >= settings.max_answer_sentences:
                break
            if len(sentence) > max_sentence_characters:
                sentence = f"{sentence[:max_sentence_characters].rsplit(' ', 1)[0]}..."
            summaries.append(sentence)
        return summaries

    @staticmethod
    def _citation_label(item: Dict[str, Any]) -> str:
        return f"[{item['source']}, p. {item['page']}]"

    @staticmethod
    def _clean_claim(text: str) -> str:
        """Keep readable prose while removing a following untrusted formula."""
        normalized = re.sub(r"\s+", " ", text).strip().lstrip("• ")
        math_start = re.search(r"[∫∑√≤≥≈≠±×÷⎛⎝⎞⎠〈〉‖]", normalized)
        if math_start:
            normalized = normalized[: math_start.start()].strip()
        normalized = re.sub(r"^[\s:;|.-]+", "", normalized)
        normalized = re.sub(
            r"^((?:GV|MP|MS|MG)-?\d+(?:\.\d+)?-\d+)\s*;\s*:\s*",
            r"\1: ",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"\.{2,}", ".", normalized)
        # PDF page numbers can be prepended to a sentence when a text block
        # crosses a footer. They are not part of the claim itself.
        normalized = re.sub(r"^\d{1,3}\s+(?=[A-Za-z])", "", normalized)
        normalized = re.sub(r"\b(?:[A-Za-z][\w']*|[A-Za-z][\w']*\([^)]*\))\s*=\s*$", "", normalized).strip()
        first_sentence = re.split(r"(?<=[.!?])\s+", normalized)[0].strip()
        if first_sentence and first_sentence[0].islower():
            first_sentence = first_sentence[0].upper() + first_sentence[1:]
        if first_sentence and first_sentence[-1].isalnum():
            first_sentence += "."
        return first_sentence

    @staticmethod
    def _is_useful_claim(claim: str) -> bool:
        words = re.findall(r"[A-Za-z]{2,}", claim)
        if len(words) < 5:
            return False
        if len(words) >= 4 and sum(word[0].isupper() for word in words) / len(words) >= 0.75:
            return False
        filler_prefixes = (
            "then ",
            "let's ",
            "at this point",
            "in particular",
            "the fundamental theorem",
            "we now have",
            "an important topic related to",
            "which is the formula",
            "according to the",
            "solve the following",
            "find the following",
            "table ",
            "table:",
            "http://",
            "https://",
            "www.",
        )
        boilerplate_markers = (
            "additional information",
            "acknowledgments",
            "commercial entities",
            "contact information",
            "copyright",
            "disclaimer",
            "glossary of terms",
            "not intended to imply",
            "recommendation or endorsement",
            "publication history",
        )
        normalized = claim.lower()
        return not normalized.startswith(filler_prefixes) and not any(marker in normalized for marker in boilerplate_markers)

    @staticmethod
    def _is_overview_question(query: str) -> bool:
        """Identify questions that ask for a topic-level explanation.

        These questions should prefer definitions and core descriptions over
        related concepts, exercises, or isolated examples.
        """
        normalized = re.sub(r"\s+", " ", query.lower()).strip(" ?.")
        return normalized.startswith((
            "tell me about ",
            "give me an overview of ",
            "give an overview of ",
            "explain ",
        ))

    @classmethod
    def _question_intent(cls, query: str) -> str:
        """Classify presentation intent without making a subject-specific assumption."""
        normalized = re.sub(r"\s+", " ", query.lower()).strip(" ?.")
        if re.search(
            r"\b(?:compare|comparison|difference|differentiate|versus|vs|both|common themes?|similarities)\b",
            normalized,
        ):
            return "comparison"
        if re.search(r"\b(?:summari[sz]e|summary|overview)\b", normalized):
            return "summary"
        if re.search(r"\b(?:figure|chart|graph|diagram|image|table)\b", normalized):
            return "visual"
        # Declarative questions can contain words such as "calculate" but
        # still ask for an explanation (for example, "What can double
        # integrals calculate?").  Classify their leading question form
        # before looking for imperative procedure language.
        if normalized.startswith(("what is ", "define ", "what does ", "what can ")) or re.search(
            r"\bwhat\s+does\b.*\b(?:mean|represent)\b", normalized
        ):
            return "definition"
        if re.search(r"\b(?:how do i|how can i|steps? to|procedure|solve|calculate|derive)\b", normalized):
            return "procedure"
        if cls._is_overview_question(normalized):
            return "explanation"
        return "explanation"

    @classmethod
    def _intent_search_query(cls, original_query: str, contextual_query: str) -> str:
        """Steer retrieval toward the evidence shape needed for the question."""
        intent = cls._question_intent(original_query)
        # Comparison is the dominant intent even when the wording also asks
        # what each source recommends. Otherwise a query such as "compare how
        # both reports recommend responses" is incorrectly routed toward one
        # structured recommendation table.
        if intent == "comparison":
            return (
                f"{contextual_query}\nPrioritize evidence for every named or selected source, then identify "
                "their distinct approaches, shared themes, and differences."
            )
        if cls._is_document_purpose_question(original_query):
            return (
                f"{contextual_query}\nPrioritize the introduction's explicit purpose, scope, intended use, "
                "and document overview. Exclude front matter, disclaimers, contact details, and publication metadata."
            )
        if cls._asks_for_recommendations(original_query):
            return (
                f"{contextual_query}\nPrioritize explicit suggested actions, recommendations, controls, "
                "and risk-management guidance. Prefer structured recommendation tables when available."
            )
        guidance = {
            "procedure": "Prioritize method steps, rules, prerequisites, and worked examples.",
            "comparison": "Prioritize direct comparisons, distinctions, and supporting evidence for each item.",
            "summary": "Prioritize central concepts, findings, and concise overview passages.",
            "visual": "Prioritize figure captions, tables, charts, diagrams, and their explanatory text.",
            "definition": "Prioritize an introductory definition, core idea, and main formula when relevant.",
            "explanation": "Prioritize an introductory definition, core idea, and main supporting explanation.",
        }
        return f"{contextual_query}\n{guidance[intent]}"

    @staticmethod
    def _compound_aspects(query: str) -> List[str]:
        """Return explicit coordinated aspects without inventing query terms."""
        normalized = re.sub(r"\s+", " ", query).strip()
        match = re.search(
            r"\b(?:across|including|covering|in terms of)\s+(.+?)(?:[?.]|$)",
            normalized,
            re.IGNORECASE,
        )
        raw_aspects: List[str] = []
        if match:
            raw_aspects = re.split(r"\s*,\s*|\s+and\s+", match.group(1), flags=re.IGNORECASE)
        else:
            verb = r"(?:identify|assess|evaluate|measure|recommend|monitor|mitigate|manage)\w*"
            serial = re.search(
                rf"\b({verb})\s*,\s*({verb})\s*,?\s+and\s+({verb}(?:\s+[A-Za-z][A-Za-z-]*){{0,4}})",
                normalized,
                re.IGNORECASE,
            )
            if serial:
                raw_aspects = list(serial.groups())
                final_words = re.findall(r"[A-Za-z][A-Za-z-]*", raw_aspects[-1])
                topic = final_words[-1] if len(final_words) > 1 else ""
                if topic:
                    raw_aspects[0] = f"{raw_aspects[0]} {topic}"
                    raw_aspects[1] = f"{raw_aspects[1]} {topic}"

        aspects = []
        for raw_aspect in raw_aspects[:4]:
            aspect = re.sub(r"^(?:and|or)\s+", "", raw_aspect, flags=re.IGNORECASE).strip(" ,:;-.?")
            words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", aspect)
            if words and len(words) <= 6:
                aspects.append(" ".join(words))
        return list(dict.fromkeys(aspects)) if len(aspects) >= 2 else []

    @classmethod
    def _compound_aspect_queries(cls, query: str) -> List[str]:
        """Expand each explicit aspect into a bounded retrieval query."""
        normalized = re.sub(r"\s+", " ", query).strip()
        aspects = cls._compound_aspects(normalized)
        if not aspects:
            return []
        boundary = re.search(r"\b(?:across|including|covering|in terms of)\b", normalized, re.IGNORECASE)
        if boundary is None:
            first_aspect = re.search(rf"\b{re.escape(aspects[0].split()[0])}\w*\b", normalized, re.IGNORECASE)
            boundary = first_aspect
        prefix = normalized[: boundary.start()].strip(" ,:;-") if boundary else normalized
        if cls._question_intent(query) == "comparison":
            # Retrieval already runs independently within each source. Keeping
            # document names in every BM25 subquery promotes front matter over
            # the requested identify/assess/recommend operations.
            prefix = ""
        expanded_queries = []
        for aspect in aspects:
            additions = ""
            if re.search(r"\bidentif", aspect, re.IGNORECASE):
                additions = "characterize detect classification hazards exposure vulnerability impacts"
            elif re.search(r"\b(?:assess|evaluat|measur)", aspect, re.IGNORECASE):
                additions = "evaluate measure evidence confidence likelihood severity"
            elif re.search(r"\b(?:recommend|monitor|mitigat|manag)", aspect, re.IGNORECASE):
                additions = "actions guidance response strategies options mitigate mitigation adaptation reduce prevention"
            expanded_queries.append(" ".join(part for part in (prefix, aspect, additions) if part))
        return expanded_queries

    @classmethod
    def _expanded_aspect_terms(cls, aspect: str) -> set[str]:
        """Add retrieval-only equivalents for comparison operations."""
        terms = cls._support_terms(aspect)
        if re.search(r"\bidentif", aspect, re.IGNORECASE):
            terms.update({"characterize", "detect", "classification", "hazard", "exposure", "vulnerability", "impact"})
        elif re.search(r"\b(?:assess|evaluat|measur)", aspect, re.IGNORECASE):
            terms.update({"assess", "evaluate", "measure", "evidence", "confidence", "likelihood", "severity"})
        elif re.search(r"\b(?:recommend|monitor|mitigat|manag)", aspect, re.IGNORECASE):
            terms.update({"action", "guidance", "response", "strategy", "option", "mitigate", "mitigation", "adaptation", "reduce", "prevention"})
        return terms

    @staticmethod
    def _is_document_purpose_question(query: str) -> bool:
        """Identify requests for a source document's purpose or scope."""
        normalized = re.sub(r"\s+", " ", query.casefold())
        return bool(
            re.search(r"\b(?:purpose|scope|aim|objective)\b", normalized)
            and re.search(r"\b(?:this|the)?\s*(?:document|report|paper|guide)\b", normalized)
        )

    @staticmethod
    def _asks_for_recommendations(query: str) -> bool:
        """Identify questions that require action-oriented source evidence."""
        return bool(re.search(r"\b(?:recommend(?:ation|ations|ed)?|suggested actions?|what should|guidance|controls?)\b", query, re.IGNORECASE))

    @staticmethod
    def _is_summary_candidate(chunk: DocumentChunk) -> bool:
        """Exclude navigation and citation boilerplate from document summaries."""
        if chunk.type != "text":
            return False
        quality_flags = set(chunk.metadata.get("quality_flags", []))
        if quality_flags.intersection({"no_extractable_text", "limited_extractable_text"}):
            return False

        content = " ".join(part for part in (chunk.text, chunk.table, chunk.figure_caption) if part).strip()
        normalized = re.sub(r"\s+", " ", content).lower()
        word_count = len(re.findall(r"[a-zA-Z]{2,}", normalized))
        if word_count < 12:
            return False
        if re.search(r"\b(?:table of contents|contents|bibliography|references)\b", normalized):
            return False
        if re.search(r"\.{3,}\s*\d+\b", normalized) or normalized.count("http") + normalized.count("www.") >= 2:
            return False
        if normalized.startswith(
            ("doi:", "see also", "copyright", "all rights reserved", "additional information", "disclaimer:", "contact information")
        ):
            return False
        return not any(marker in normalized for marker in ("commercial entities", "publication history"))

    @staticmethod
    def _has_summary_coverage(evidence: List[Dict[str, Any]], requested_top_k: int) -> bool:
        """Require multi-page, substantive evidence before overriding raw-logit confidence."""
        minimum_evidence = min(3, requested_top_k)
        distinct_pages = {(item["source"], item["page"]) for item in evidence}
        return len(evidence) >= minimum_evidence and len(distinct_pages) >= minimum_evidence

    def _document_purpose_candidate_pool(self, source_names: set[str]) -> List[DocumentChunk]:
        """Retrieve an explicit introduction/scope statement before generic purpose matches.

        Front matter often includes the word ``purpose`` in a disclaimer. A
        scoped document-purpose query instead needs the first substantive
        statement explaining what the report is and what it is for.
        """
        if len(source_names) != 1:
            return []
        purpose_pattern = re.compile(
            r"\b(?:this (?:document|report|paper|guide) is|purpose of (?:this|the)|companion resource|"
            r"intended (?:for|to)|provides (?:a|an)|scope of (?:this|the))\b",
            re.IGNORECASE,
        )
        candidates: list[tuple[int, int, DocumentChunk]] = []
        for chunk in self.retriever.chunks:
            if chunk.source not in source_names or chunk.type != "text":
                continue
            quality_flags = set(chunk.metadata.get("quality_flags", []))
            if quality_flags.intersection({"no_extractable_text", "limited_extractable_text"}):
                continue
            content = " ".join((chunk.text, chunk.table, chunk.figure_caption))
            normalized = re.sub(r"\s+", " ", content).casefold()
            if any(
                marker in normalized
                for marker in (
                    "table of contents",
                    "contents",
                    "bibliography",
                    "references",
                    "commercial entities",
                    "not intended to imply",
                    "recommendation or endorsement",
                    "publication history",
                )
            ):
                continue
            if not purpose_pattern.search(content):
                continue
            score = 0
            if re.search(r"\bthis (?:document|report|paper|guide) is\b", content, re.IGNORECASE):
                score += 8
            if re.search(r"\b(?:purpose|scope|aim|objective|intended)\b", content, re.IGNORECASE):
                score += 5
            if re.search(r"\b(?:cross-sectoral|profile|companion resource|framework)\b", content, re.IGNORECASE):
                score += 3
            candidates.append((score, -int(chunk.metadata.get("page", 0)), chunk))
        return [chunk for _, _, chunk in sorted(candidates, key=lambda candidate: candidate[:2], reverse=True)[:16]]

    @staticmethod
    def _structured_identifier_pattern(query: str) -> str | None:
        """Return a tolerant pattern for a named report table or control ID."""
        match = re.search(
            r"\b(govern|map|measure|manage)\s+(\d+)\s*[.]\s*(\d+)\b",
            query,
            re.IGNORECASE,
        )
        if not match:
            return None
        family, major, minor = match.groups()
        return rf"\b{re.escape(family)}\s+{major}\s*[.]\s*{minor}\b"

    @staticmethod
    def _query_named_source_names(query: str, source_names: set[str]) -> set[str]:
        """Infer an explicit document target from a filename mentioned in a query."""
        query_terms = set(re.findall(r"[A-Za-z]{4,}", query.casefold()))
        if not query_terms:
            return set()
        return {
            source
            for source in source_names
            if query_terms.intersection(re.findall(r"[A-Za-z]{4,}", Path(source).stem.casefold()))
        }

    @classmethod
    def _lexical_retrieval_query(cls, query: str, intent: str) -> str:
        """Add intent terms that improve BM25 recall without changing user intent.

        Dense retrieval already captures semantic similarity, while BM25 needs
        the vocabulary likely to occur in an answer-bearing passage. These are
        general question-shape terms, not document- or dataset-specific
        answers. The original query is always retained verbatim.
        """
        normalized = re.sub(r"\s+", " ", query.casefold())
        additions: list[str] = []
        if intent == "comparison":
            additions.extend(["comparison differences evidence", "rates values measures findings"])
        elif intent == "definition":
            additions.append("defined definition means refers to")
        elif intent == "procedure":
            additions.append("method steps process how to")
        elif intent == "summary":
            additions.append("purpose findings risks recommendations implications")
        elif intent == "visual":
            additions.append("table figure chart caption")

        if re.search(r"\b(?:cause|caused|why)\b", normalized):
            additions.append("cause causes attribution evidence due to")
        if re.search(r"\b(?:impact|impacts|harm|loss|damage|vulnerable)\b", normalized):
            additions.append("impacts risks losses damages affected")
        if re.search(r"\b(?:which|where).*(?:area|areas|region|regions|location|locations|studied)\b", normalized):
            additions.append("study area regions locations districts")
        if re.search(r"\b(?:how much|rate|rates|percent|percentage|quantif)\b", normalized):
            additions.append("data estimate rate value percentage")
        if cls._asks_for_recommendations(query):
            additions.append("suggested actions recommendations guidance controls")
        return " ".join([query, *additions])

    def _exact_identifier_candidates(
        self,
        query: str,
        source_names: set[str] | None,
    ) -> List[RetrievalResult]:
        """Find exact report tables before approximate hybrid retrieval.

        Tables such as ``GOVERN 1.2`` contain a precise identifier that should
        outrank adjacent controls even when all of them share similar language.
        """
        pattern = self._structured_identifier_pattern(query)
        if pattern is None:
            return []

        candidates = []
        for chunk in self.retriever.chunks:
            if source_names is not None and chunk.source not in source_names:
                continue
            if chunk.type != "table":
                continue
            content = f"{chunk.text}\n{chunk.table}"
            if re.search(pattern, content, re.IGNORECASE):
                candidates.append(
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        score=1.0,
                        text=chunk.text,
                        table=chunk.table,
                        figure_caption=chunk.figure_caption,
                        source=chunk.source,
                        section=chunk.section,
                        metadata=chunk.metadata,
                        equations=chunk.equations,
                        type=chunk.type,
                    )
                )
        return candidates

    @staticmethod
    def _merge_unique_candidates(
        primary: List[RetrievalResult],
        secondary: List[RetrievalResult],
        limit: int,
    ) -> List[RetrievalResult]:
        """Combine candidate lists while preserving the primary ordering."""
        merged = []
        seen_ids = set()
        for candidate in [*primary, *secondary]:
            if candidate.chunk_id in seen_ids:
                continue
            merged.append(candidate)
            seen_ids.add(candidate.chunk_id)
            if len(merged) >= limit:
                break
        return merged

    @staticmethod
    def _support_terms(query: str) -> set[str]:
        """Extract meaningful terms used to validate retrieved evidence.

        These terms are deliberately separate from the embedding query. They
        provide a small, deterministic guardrail against answering from a
        merely adjacent passage after the reranker has ordered the candidates.
        """
        stop_terms = {
            "about",
            "an",
            "and",
            "answer",
            "appear",
            "are",
            "across",
            "both",
            "can",
            "compare",
            "common",
            "could",
            "describe",
            "did",
            "do",
            "does",
            "document",
            "each",
            "explain",
            "for",
            "from",
            "give",
            "how",
            "in",
            "is",
            "it",
            "me",
            "of",
            "on",
            "or",
            "own",
            "paper",
            "please",
            "report",
            "reports",
            "separate",
            "stand",
            "summarize",
            "table",
            "tell",
            "the",
            "themes",
            "this",
            "to",
            "use",
            "used",
            "what",
            "when",
            "which",
            "why",
            "will",
            "with",
            "words",
            "you",
            "your",
        }
        def normalize(term: str) -> str:
            aliases = {
                "actions": "action",
                "assessed": "assess",
                "assesses": "assess",
                "assessing": "assess",
                "assessment": "assess",
                "assessments": "assess",
                "governance": "govern",
                "governing": "govern",
                "hazards": "hazard",
                "impacts": "impact",
                "managed": "manage",
                "management": "manage",
                "managing": "manage",
                "measurement": "measure",
                "measurements": "measure",
                "monitoring": "monitor",
                "monitored": "monitor",
                "organizations": "organization",
                "organizational": "organization",
                "options": "option",
                "recommendations": "recommend",
                "recommended": "recommend",
                "recommends": "recommend",
                "responses": "response",
                "risks": "risk",
                "strategies": "strategy",
                "vulnerabilities": "vulnerability",
            }
            return aliases.get(term, term)

        return {
            normalize(term)
            for term in re.findall(r"[A-Za-z][A-Za-z0-9]*", query.casefold())
            if term not in stop_terms
        }

    @classmethod
    def _evidence_supports_query(
        cls,
        query: str,
        evidence: List[Dict[str, Any]],
        intent: str,
    ) -> bool:
        """Decide whether retrieved text supports an answer without logit thresholds.

        Cross-encoder outputs rank candidates but are not calibrated
        probabilities. This gate instead requires question terminology in the
        retrieved evidence, and requires evidence from every requested side of
        a comparison. It intentionally remains conservative for questions that
        are absent from the local corpus, such as tomorrow's weather.
        """
        if not evidence:
            return False

        evidence_text = " ".join(
            " ".join(
                str(item.get(field, ""))
                for field in ("text", "table", "figure_caption")
            )
            for item in evidence
        ).casefold()
        acronym_match = re.search(
            r"\bwhat\s+does\s+([A-Za-z][A-Za-z0-9.-]{1,})\s+stand\s+for\b",
            query,
            re.IGNORECASE,
        )
        if acronym_match:
            acronym = re.escape(acronym_match.group(1))
            return bool(re.search(rf"\(\s*{acronym}\s*\)", evidence_text, re.IGNORECASE))

        identifier_pattern = cls._structured_identifier_pattern(query)
        if identifier_pattern and re.search(identifier_pattern, evidence_text, re.IGNORECASE):
            return True
        if re.search(r"\b(?:today|tomorrow|yesterday|latest|current|live)\b|\bstock\s+price\b", query, re.IGNORECASE):
            # The assistant is intentionally local-document-grounded and does
            # not have a live market or weather feed.
            return False

        if cls._is_document_purpose_question(query):
            return bool(
                re.search(
                    r"\b(?:this (?:document|report|paper|guide) is|purpose|scope|aim|objective|companion resource|intended (?:for|to))\b",
                    evidence_text,
                    re.IGNORECASE,
                )
            )

        query_terms = cls._support_terms(query)
        if not query_terms:
            return False
        evidence_terms = cls._support_terms(evidence_text)
        matching_terms = query_terms.intersection(evidence_terms)
        coverage = len(matching_terms) / len(query_terms)

        if intent == "comparison":
            # A comparison is not grounded if all evidence came from only one
            # document, even when that document happens to mention both terms.
            return (
                len({item["source"] for item in evidence}) >= 2
                and len(matching_terms) >= 2
                and coverage >= 0.35
            )

        aspects = cls._compound_aspects(query)
        if aspects:
            planned_term_sets = [cls._support_terms(aspect) for aspect in aspects]
            shared_terms = set.intersection(*planned_term_sets) if planned_term_sets else set()
            aspect_term_sets = [terms - shared_terms for terms in planned_term_sets]
            aspect_term_sets = [terms for terms in aspect_term_sets if terms]
            aspects_supported = all(terms.intersection(evidence_terms) for terms in aspect_term_sets)
            return aspects_supported and len(matching_terms) >= 2 and coverage >= 0.35

        required_matches = 1 if len(query_terms) <= 2 else 2
        return len(matching_terms) >= required_matches and coverage >= 0.5

    def _summary_source_names(self, request: QueryRequest) -> set[str]:
        """Resolve an explicit browser scope, or the sole indexed document."""
        indexed_names = {
            document["filename"]
            for document in self.documents
            if document.get("status") == "indexed"
        }
        requested_names = {Path(name).name for name in request.document_names}
        if requested_names:
            return indexed_names.intersection(requested_names)
        return indexed_names if len(indexed_names) == 1 else set()

    def _requested_source_names(self, request: QueryRequest) -> set[str] | None:
        """Resolve an explicit document scope without broadening invalid requests.

        ``None`` means the caller did not request a scope.  An empty set means
        it did request one, but none of those documents are currently indexed;
        callers must return no evidence rather than silently searching every
        document in the corpus.
        """
        requested_names = {Path(name).name for name in request.document_names}
        if not requested_names:
            return None
        indexed_names = {
            document["filename"]
            for document in self.documents
            if document.get("status") == "indexed"
        }
        return indexed_names.intersection(requested_names)

    @staticmethod
    def _diversify_comparison_candidates(
        candidates: List[RetrievalResult],
        source_names: set[str],
        limit: int,
    ) -> List[RetrievalResult]:
        """Preserve evidence from each requested source before reranking."""
        by_source: Dict[str, List[RetrievalResult]] = defaultdict(list)
        for candidate in candidates:
            if candidate.source in source_names:
                by_source[candidate.source].append(candidate)

        selected: List[RetrievalResult] = []
        selected_ids = set()
        ordered_sources = sorted(source_names)
        maximum_rank = max((len(items) for items in by_source.values()), default=0)
        for rank in range(maximum_rank):
            for source in ordered_sources:
                source_candidates = by_source.get(source, [])
                if rank >= len(source_candidates):
                    continue
                candidate = source_candidates[rank]
                if candidate.chunk_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.chunk_id)
                if len(selected) >= limit:
                    return selected
        return selected

    def _hybrid_candidate_pool(
        self,
        query_embedding: Any,
        lexical_query: str,
        source_names: set[str] | None,
        intent: str,
        requested_top_k: int,
        aspect_queries: List[str] | None = None,
        aspect_query_embeddings: List[Any] | None = None,
    ) -> tuple[List[RetrievalResult], List[RetrievalResult]]:
        """Fuse dense and BM25 candidates and retain dense backfill evidence."""
        dense_candidate_k = max(requested_top_k, settings.retrieval_candidate_k)
        sparse_candidate_k = max(requested_top_k, settings.sparse_candidate_k)
        is_multi_document_comparison = intent == "comparison" and source_names is not None and len(source_names) > 1
        sparse_retriever = getattr(self, "sparse_retriever", None)

        sparse_ranked_lists: List[List[RetrievalResult]] = []
        aspect_ranked_lists: List[List[RetrievalResult]] = []
        if is_multi_document_comparison:
            per_source_dense_k = math.ceil(dense_candidate_k / len(source_names))
            per_source_sparse_k = math.ceil(sparse_candidate_k / len(source_names))
            dense_candidates = self.retriever.retrieve_diversified(
                query_embedding,
                sources=source_names,
                per_source_k=per_source_dense_k,
            )
            for aspect_embedding in aspect_query_embeddings or []:
                aspect_candidates = self.retriever.retrieve_diversified(
                    aspect_embedding,
                    sources=source_names,
                    per_source_k=per_source_dense_k,
                )
                if aspect_candidates:
                    aspect_ranked_lists.append(aspect_candidates)
            sparse_candidates = (
                sparse_retriever.retrieve_diversified(
                    lexical_query,
                    sources=source_names,
                    per_source_k=per_source_sparse_k,
                )
                if sparse_retriever is not None
                else []
            )
            if sparse_candidates:
                sparse_ranked_lists.append(sparse_candidates)
            for aspect_query in aspect_queries or []:
                aspect_candidates = (
                    sparse_retriever.retrieve_diversified(
                        aspect_query,
                        sources=source_names,
                        per_source_k=per_source_sparse_k,
                    )
                    if sparse_retriever is not None
                    else []
                )
                if aspect_candidates:
                    aspect_ranked_lists.append(aspect_candidates)
        else:
            dense_candidates = self.retriever.retrieve(
                query_embedding,
                top_k=dense_candidate_k,
                sources=source_names,
            )
            for aspect_embedding in aspect_query_embeddings or []:
                aspect_candidates = self.retriever.retrieve(
                    aspect_embedding,
                    top_k=dense_candidate_k,
                    sources=source_names,
                )
                if aspect_candidates:
                    aspect_ranked_lists.append(aspect_candidates)
            sparse_candidates = (
                sparse_retriever.retrieve(
                    lexical_query,
                    top_k=sparse_candidate_k,
                    sources=source_names,
                )
                if sparse_retriever is not None
                else []
            )
            if sparse_candidates:
                sparse_ranked_lists.append(sparse_candidates)
            for aspect_query in aspect_queries or []:
                aspect_candidates = (
                    sparse_retriever.retrieve(
                        aspect_query,
                        top_k=sparse_candidate_k,
                        sources=source_names,
                    )
                    if sparse_retriever is not None
                    else []
                )
                if aspect_candidates:
                    aspect_ranked_lists.append(aspect_candidates)

        fused = reciprocal_rank_fusion(
            [dense_candidates, *aspect_ranked_lists, *sparse_ranked_lists],
            top_k=dense_candidate_k,
            rank_constant=settings.reciprocal_rank_fusion_constant,
        )
        if aspect_ranked_lists:
            # RRF favors passages that recur across every aspect. Preserve a
            # bounded round-robin head from each individual aspect so a unique
            # response/measurement passage still reaches the cross-encoder.
            aspect_backfill: List[RetrievalResult] = []
            maximum_rank = max(len(candidates) for candidates in aspect_ranked_lists)
            for rank in range(maximum_rank):
                for candidates in aspect_ranked_lists:
                    if rank < len(candidates):
                        aspect_backfill.append(candidates[rank])
                        if len(aspect_backfill) >= dense_candidate_k:
                            break
                if len(aspect_backfill) >= dense_candidate_k:
                    break
            fused = self._merge_unique_candidates(aspect_backfill, fused, dense_candidate_k)
        exact_identifier_candidates = self._exact_identifier_candidates(lexical_query, source_names)
        fused = self._merge_unique_candidates(exact_identifier_candidates, fused, dense_candidate_k)
        if is_multi_document_comparison:
            fused = self._diversify_comparison_candidates(fused, source_names, dense_candidate_k)
        return fused, dense_candidates

    @staticmethod
    def _hybrid_rerank_candidates(
        fused_candidates: List[RetrievalResult],
        dense_candidates: List[RetrievalResult],
        fused_limit: int,
        dense_backfill_limit: int,
    ) -> List[RetrievalResult]:
        """Keep dense recall coverage while introducing lexical hybrid evidence."""
        selected: List[RetrievalResult] = []
        selected_ids = set()
        for candidates, limit in (
            (fused_candidates, fused_limit),
            (dense_candidates, dense_backfill_limit),
        ):
            for candidate in candidates[:limit]:
                if candidate.chunk_id not in selected_ids:
                    selected.append(candidate)
                    selected_ids.add(candidate.chunk_id)
        return selected

    def _summary_candidate_pool(self, source_names: set[str]) -> List[DocumentChunk]:
        """Sample substantive chunks across sections before reranking a summary."""
        eligible = [
            chunk
            for chunk in self.retriever.chunks
            if chunk.source in source_names and self._is_summary_candidate(chunk)
        ]
        if len(eligible) <= settings.summary_candidate_k:
            return eligible

        selected: list[DocumentChunk] = []
        selected_ids = set()
        section_representatives: Dict[str, DocumentChunk] = {}
        for chunk in eligible:
            section_key = chunk.section.strip().lower() or f"page-{chunk.metadata.get('page', 0)}"
            existing = section_representatives.get(section_key)
            if existing is None or len(chunk.text) > len(existing.text):
                section_representatives[section_key] = chunk
        for chunk in section_representatives.values():
            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)
            if len(selected) >= settings.summary_candidate_k:
                return selected

        remaining_slots = settings.summary_candidate_k - len(selected)
        stride = max(1, math.ceil(len(eligible) / remaining_slots))
        for index in range(0, len(eligible), stride):
            chunk = eligible[index]
            if chunk.chunk_id in selected_ids:
                continue
            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)
            if len(selected) >= settings.summary_candidate_k:
                break
        return selected

    @staticmethod
    def _diversify_summary_reranking(reranked: List[tuple[tuple[str, Any], float]]) -> List[tuple[tuple[str, Any], float]]:
        """Prefer distinct sections and pages over near-duplicate summary evidence."""
        diversified = []
        deferred = []
        seen_sections = set()
        seen_pages = set()
        for result in reranked:
            _, chunk = result[0]
            page_key = (chunk.source, chunk.metadata.get("page", 1))
            section_key = (chunk.source, chunk.section.strip().lower() or f"page-{page_key[1]}")
            if section_key not in seen_sections and page_key not in seen_pages:
                diversified.append(result)
                seen_sections.add(section_key)
                seen_pages.add(page_key)
            else:
                deferred.append(result)
        return diversified + deferred

    @staticmethod
    def _diversify_comparison_reranking(
        reranked: List[tuple[tuple[str, Any], float]],
        source_names: set[str],
        limit: int,
    ) -> List[tuple[tuple[str, Any], float]]:
        """Keep the strongest evidence from each comparison source near the top."""
        by_source: Dict[str, List[tuple[tuple[str, Any], float]]] = defaultdict(list)
        for result in reranked:
            _, chunk = result[0]
            if chunk.source in source_names:
                by_source[chunk.source].append(result)

        selected: List[tuple[tuple[str, Any], float]] = []
        selected_ids = set()
        ordered_sources = sorted(source_names)
        maximum_rank = max((len(items) for items in by_source.values()), default=0)
        for rank in range(maximum_rank):
            ranked_round = [
                by_source[source][rank]
                for source in ordered_sources
                if rank < len(by_source.get(source, []))
            ]
            for result in sorted(ranked_round, key=lambda item: item[1], reverse=True):
                _, chunk = result[0]
                if chunk.chunk_id in selected_ids:
                    continue
                selected.append(result)
                selected_ids.add(chunk.chunk_id)
                if len(selected) >= limit:
                    return selected
        return selected

    @classmethod
    def _prioritize_comparison_aspects(
        cls,
        reranked: List[tuple[tuple[str, Any], float]],
        source_names: set[str],
        query: str,
        limit: int,
    ) -> List[tuple[tuple[str, Any], float]]:
        """Prefer evidence for each requested operation from each comparison source."""
        aspects = cls._compound_aspects(query)
        if len(aspects) < 2:
            return reranked[:limit]
        original_aspect_terms = [cls._support_terms(aspect) for aspect in aspects]
        shared_terms = set.intersection(*original_aspect_terms) if original_aspect_terms else set()
        aspect_term_sets = [
            (cls._expanded_aspect_terms(aspect) - shared_terms) or cls._expanded_aspect_terms(aspect)
            for aspect in aspects
        ]

        selected: List[tuple[tuple[str, Any], float]] = []
        selected_ids = set()
        for terms in aspect_term_sets:
            for source in sorted(source_names):
                matches = []
                for result in reranked:
                    _, chunk = result[0]
                    if chunk.source != source or chunk.chunk_id in selected_ids:
                        continue
                    content = " ".join((chunk.text, chunk.table, chunk.figure_caption))
                    content_terms = cls._support_terms(content)
                    overlap = len(terms.intersection(content_terms))
                    has_shared_topic = not shared_terms or bool(shared_terms.intersection(content_terms))
                    if overlap and has_shared_topic:
                        matches.append((overlap, result[1], result))
                if not matches:
                    continue
                best = max(matches, key=lambda match: (match[0], match[1]))[2]
                selected.append(best)
                selected_ids.add(best[0][1].chunk_id)
                if len(selected) >= limit:
                    return selected

        selected.extend(
            result
            for result in reranked
            if result[0][1].chunk_id not in selected_ids
        )
        return selected[:limit]

    @staticmethod
    def _is_comparison_candidate(chunk: Any) -> bool:
        """Exclude navigation/reference fragments from comparative synthesis."""
        section = str(getattr(chunk, "section", "")).casefold()
        if re.search(r"\b(?:acknowledg|bibliograph|references?|works cited)\b", section):
            return False
        content = " ".join((chunk.text, chunk.table, chunk.figure_caption)).strip()
        normalized = re.sub(r"\s+", " ", content).casefold()
        if len(re.findall(r"[a-z]{2,}", normalized)) < 8:
            return False
        if normalized.startswith(("http://", "https://", "www.")):
            return False
        return not re.search(r"\b(?:isbn|doi:)\b", normalized)

    @staticmethod
    def _query_requests_math_evidence(query: str) -> bool:
        """Only expose equation crops when the user's information need is mathematical."""
        return bool(
            re.search(
                r"\b(?:algebra|calculate|calculus|derivative|differential|divergence|equation|formula|"
                r"gradient|integral|laplacian|matrix|mathematical|notation|proof|solve|theorem|vector)\b|"
                r"[=+±√∫∑∂∇]",
                query,
                re.IGNORECASE,
            )
        )

    @classmethod
    def _prioritize_aspect_evidence(
        cls,
        query: str,
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Place evidence covering each coordinated aspect before general context."""
        aspects = cls._compound_aspects(query)
        if len(aspects) < 2:
            return evidence
        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        for aspect in aspects:
            terms = cls._support_terms(aspect)
            ranked = sorted(
                evidence,
                key=lambda item: (
                    len(
                        terms.intersection(
                            cls._support_terms(
                                " ".join(
                                    str(item.get(field, ""))
                                    for field in ("text", "table", "figure_caption")
                                )
                            )
                        )
                    ),
                    float(item.get("score", 0.0)),
                ),
                reverse=True,
            )
            if not ranked:
                continue
            best = ranked[0]
            best_content = " ".join(str(best.get(field, "")) for field in ("text", "table", "figure_caption"))
            if not terms.intersection(cls._support_terms(best_content)):
                continue
            chunk_id = best["chunk_id"]
            if chunk_id not in selected_ids:
                selected.append(best)
                selected_ids.add(chunk_id)
        selected.extend(item for item in evidence if item["chunk_id"] not in selected_ids)
        return selected

    @classmethod
    def _extract_claims(cls, text: str) -> List[str]:
        """Return readable sentences from an extracted passage.

        PDF chunks often begin with a heading or transition, while the useful
        explanation appears in a later sentence.  Treating every sentence as a
        candidate keeps synthesis grounded without losing that explanation.
        """
        normalized = re.sub(r"\s+", " ", text).strip()
        math_start = re.search(r"[âˆ«âˆ‘âˆšâ‰¤â‰¥â‰ˆâ‰ Â±Ã—Ã·âŽ›âŽâŽžâŽ âŒ©âŒªâ€–]", normalized)
        if math_start:
            normalized = normalized[: math_start.start()].strip()

        claims = []
        for sentence in re.split(r"(?<=[.!?])\s+", normalized):
            claim = cls._clean_claim(sentence)
            if claim:
                claims.append(claim)
        return claims

    @staticmethod
    def _extract_table_claims(table: str) -> List[str]:
        """Turn labeled table rows into short, source-backed claims.

        This is intentionally limited to labels that appear in the source; it
        does not infer a solution or alter mathematical notation.
        """
        normalized_table = re.sub(r"\s+", " ", table).casefold()
        if "r(x)" not in normalized_table or "initial guess" not in normalized_table:
            return []

        claims = []
        for row in table.splitlines():
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            if len(cells) < 2 or not cells[0] or set(cells[0]) == {"-"}:
                continue
            if cells[0].lower() == "r(x)" or cells[1].lower().startswith("initial guess"):
                continue
            claims.append(f"For r(x) = {cells[0]}, the initial guess for y_p(x) is {cells[1]}.")

        pattern = re.compile(
            r"r\s*\(\s*x\s*\)\s*:\s*([^;|\n]+).*?initial guess for\s*y\s*p\s*\(\s*x\s*\)\s*:\s*([^;|\n]+)",
            re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(table):
            forcing_term = re.sub(r"\s+", " ", match.group(1)).strip(" .")
            initial_guess = re.sub(r"\s+", " ", match.group(2)).strip(" .")
            if forcing_term and initial_guess:
                claims.append(f"For r(x) = {forcing_term}, the initial guess for y_p(x) is {initial_guess}.")
        return list(dict.fromkeys(claims))

    @staticmethod
    def _table_rows(table: str) -> List[List[str]]:
        """Parse the lightweight Markdown tables produced by local extraction."""
        rows = []
        for row in table.splitlines():
            cells = [re.sub(r"\s+", " ", cell.replace("\\|", "|")).strip() for cell in row.strip().strip("|").split("|")]
            if not cells or all(not cell or set(cell) == {"-"} for cell in cells):
                continue
            rows.append(cells)
        return rows

    @staticmethod
    def _table_label(cell: str) -> str:
        """Normalize imperfect OCR headers without changing extracted values."""
        return re.sub(r"[^a-z0-9]", "", cell.casefold())

    @classmethod
    def _suggested_action_rows(cls, table: str) -> List[tuple[str, str, str]]:
        """Extract action ID, recommendation, and risk columns from a report table."""
        rows = cls._table_rows(table)
        for header_index, header in enumerate(rows):
            labels = [cls._table_label(cell) for cell in header]
            action_id_index = next(
                (index for index, label in enumerate(labels) if ("action" in label or "actoin" in label) and "id" in label),
                None,
            )
            suggested_action_index = next(
                (index for index, label in enumerate(labels) if "suggested" in label and ("action" in label or "actoin" in label)),
                None,
            )
            risk_index = next((index for index, label in enumerate(labels) if "risk" in label), None)
            if action_id_index is None or suggested_action_index is None:
                continue

            extracted_rows = []
            for row in rows[header_index + 1 :]:
                padded = row + [""] * (max(action_id_index, suggested_action_index, risk_index or 0) + 1 - len(row))
                action_id = padded[action_id_index].strip()
                action = padded[suggested_action_index].strip()
                risk = padded[risk_index].strip() if risk_index is not None else ""
                if not action_id or not action or action_id.casefold().startswith("ai actor"):
                    continue
                if re.fullmatch(r"(?:GV|MP|MS|MG)-?\d+(?:\.\d+)?-\d+", action_id, re.IGNORECASE):
                    extracted_rows.append((action_id, action, risk))
            return extracted_rows
        return []

    @classmethod
    def _build_table_answer(cls, query: str, evidence: List[Dict[str, Any]]) -> str | None:
        """Render report recommendations from structured table rows, not flattened prose."""
        asks_for_actions = bool(re.search(r"\b(?:recommend|suggested action|actions?|table)\b", query, re.IGNORECASE))
        if not asks_for_actions:
            return None

        identifier_pattern = cls._structured_identifier_pattern(query)
        table_evidence = [item for item in evidence if item.get("type") == "table" and item.get("table")]
        if identifier_pattern:
            table_evidence = [
                item
                for item in table_evidence
                if re.search(identifier_pattern, f"{item['text']}\n{item['table']}", re.IGNORECASE)
            ]
        if not table_evidence:
            return None

        # Retrieval can surface a narrative or metadata table before the
        # action table. Choose the first table that actually exposes action
        # IDs and suggested-action rows rather than flattening an unrelated
        # table into a generic response.
        item = None
        rows = []
        for candidate in table_evidence:
            candidate_rows = cls._suggested_action_rows(candidate["table"])
            if candidate_rows:
                item = candidate
                rows = candidate_rows
                break
        if item is None:
            return None

        title_match = re.search(
            r"((?:govern|map|measure|manage)\s+\d+\s*[.]\s*\d+)\s*:\s*(.*?)(?=\.{1,2}\s*:?\s*act(?:io|oi)\s*n\s+id|$)",
            item["text"],
            re.IGNORECASE,
        )
        if title_match:
            table_title = re.sub(r"\s+", " ", title_match.group(1)).upper()
            table_purpose = re.sub(r"\s+", " ", title_match.group(2)).strip(" .")
        else:
            table_title = "Table recommendations"
            table_purpose = ""

        citation = cls._citation_label(item)
        answer_lines = [f"**{table_title} recommendations**", ""]
        if table_purpose:
            answer_lines.extend([f"{table_purpose}. {citation}", ""])
        answer_lines.append("**Recommended actions**")
        for index, (action_id, action, risk) in enumerate(rows[:5], start=1):
            answer_lines.append(f"{index}. **{action_id}** — {action} {citation}")
            if risk:
                answer_lines.append(f"   - Related risks: {risk}")
        answer_lines.extend(["", "**Source**", cls._format_citations([item])])
        return "\n".join(answer_lines)

    @classmethod
    def _build_acronym_answer(cls, query: str, evidence: List[Dict[str, Any]]) -> str | None:
        """Answer an acronym-expansion question from an explicit source expansion."""
        acronym_match = re.search(
            r"\bwhat\s+does\s+([A-Za-z][A-Za-z0-9.-]{1,})\s+stand\s+for\b",
            query,
            re.IGNORECASE,
        )
        if not acronym_match:
            return None

        acronym = acronym_match.group(1)
        expansion_pattern = re.compile(
            rf"(?<![A-Za-z])((?:The\s+)?[A-Z][A-Za-z]+(?:\s+(?:[A-Z][A-Za-z]+|of|and|the)){{1,8}})\s+\(\s*{re.escape(acronym)}\s*\)",
        )
        for item in evidence:
            match = expansion_pattern.search(item["text"])
            if not match:
                continue
            expansion = re.sub(r"\s+", " ", match.group(1)).strip()
            expansion = re.sub(r"^The\s+", "", expansion)
            return f"**Answer**\n\n**{acronym.upper()}** stands for **{expansion}**. {cls._citation_label(item)}"
        return None

    @classmethod
    def _build_document_purpose_answer(cls, query: str, evidence: List[Dict[str, Any]]) -> str | None:
        """Use introductory scope statements for document-purpose questions."""
        if not cls._is_document_purpose_question(query):
            return None

        purpose_pattern = re.compile(
            r"\b(?:this (?:document|report|paper|guide) is|purpose of (?:this|the)|"
            r"scope of (?:this|the)|aim of (?:this|the)|objective of (?:this|the)|"
            r"companion resource|intended for (?:voluntary use|organizations|use))\b",
            re.IGNORECASE,
        )
        candidates = []
        for evidence_index, item in enumerate(evidence):
            for sentence_index, claim in enumerate(cls._extract_claims(item["text"])):
                if not cls._is_useful_claim(claim) or not purpose_pattern.search(claim):
                    continue
                score = 0
                if re.search(r"\bthis (?:document|report|paper|guide) is\b", claim, re.IGNORECASE):
                    score += 8
                if re.search(r"\b(?:purpose|scope|aim|objective|intended)\b", claim, re.IGNORECASE):
                    score += 5
                if re.search(r"\b(?:cross-sectoral|profile|companion resource|framework)\b", claim, re.IGNORECASE):
                    score += 3
                if re.search(r"\b(?:impersonat|fraudulent|malicious actor)\b", claim, re.IGNORECASE):
                    # An incidental statement about the purpose of malicious
                    # content is not the purpose of the source document.
                    score -= 10
                candidates.append((score, -evidence_index, -sentence_index, claim, item))
        if not candidates:
            return None

        _, _, _, purpose, item = max(candidates, key=lambda candidate: candidate[:3])
        return "\n".join(["**Purpose**", "", f"{purpose} {cls._citation_label(item)}"])

    @classmethod
    def _rank_claims(
        cls,
        query: str,
        evidence: List[Dict[str, Any]],
        intent: str,
    ) -> List[tuple[int, int, int, str, Dict[str, Any]]]:
        """Rank citation-ready claims while rejecting headings and extraction noise."""
        query_terms = cls._support_terms(query)
        has_math_evidence = any(item.get("equations") for item in evidence)
        candidates: list[tuple[int, int, int, str, Dict[str, Any]]] = []
        seen_claims = set()

        for evidence_index, item in enumerate(evidence):
            passage = " ".join(
                content for content in (item["text"], item.get("figure_caption", "")) if content
            )
            claims = cls._extract_claims(passage)
            claims.extend(cls._extract_table_claims(item.get("table", "")))
            claims.extend(
                f"{action_id}: {action}"
                for action_id, action, _ in cls._suggested_action_rows(item.get("table", ""))
            )
            for sentence_index, claim in enumerate(claims):
                normalized_claim = re.sub(r"\W+", " ", claim.lower()).strip()
                if not cls._is_useful_claim(claim) or normalized_claim in seen_claims:
                    continue
                seen_claims.add(normalized_claim)
                claim_terms = cls._support_terms(claim)
                relevance = len(query_terms.intersection(claim_terms))
                if has_math_evidence and re.search(r"\b(?:arc length|equation|formula|integral|curve)\b", claim, re.IGNORECASE):
                    relevance += 2
                if has_math_evidence and "formula" in claim.lower() and "curve" in claim.lower():
                    relevance += 1
                if intent in {"definition", "explanation", "summary"}:
                    if re.search(r"\b(?:measures|is defined|represents)\b", claim, re.IGNORECASE):
                        relevance += 4
                    elif re.search(r"\bcalculated\b", claim, re.IGNORECASE):
                        relevance += 3
                    elif "formula" in claim.lower() or "is given by" in claim.lower():
                        relevance += 1
                    if re.search(r"\b(?:what|which)\s+(?:data\s+)?source\b", query, re.IGNORECASE) and re.search(
                        r"\bsource\s+for\s+(?:the\s+)?(?:estimate|data)|(?:data|estimate)\s+(?:come|comes)\s+from\b",
                        claim,
                        re.IGNORECASE,
                    ):
                        relevance += 6
                    if re.search(r"\b(?:represent|meaning|mean)\b", query, re.IGNORECASE):
                        if re.search(r"\b(?:distance|travels?)\b", claim, re.IGNORECASE):
                            relevance += 8
                        elif re.search(r"\brepresent(?:s|ed)?\b", claim, re.IGNORECASE):
                            relevance += 6
                        elif re.search(r"\bline segments?\b", claim, re.IGNORECASE):
                            relevance += 2
                    if "unrelated individual" in claim.casefold() and "unrelated" not in query.casefold():
                        # Avoid leading with a subgroup statistic when the
                        # question asks for an overall population value.
                        relevance -= 5
                    if re.search(r"\b(?:find|example|exercise|problem)\b", claim, re.IGNORECASE):
                        relevance -= 2
                    if re.search(r"\btheorem\b", query, re.IGNORECASE) and re.search(
                        r"\b(?:theorem|states?|relates?|translates?|flux|boundary|surface|volume)\b",
                        claim,
                        re.IGNORECASE,
                    ):
                        relevance += 6
                if intent == "summary":
                    if re.search(r"\b(?:purpose|aim|objective|scope|intended|this (?:document|report|guide)|provides)\b", claim, re.IGNORECASE):
                        relevance += 4
                    if re.search(r"\b(?:recommend|should|must|guidance|need to|call for|encourage|suggested actions?)\b", claim, re.IGNORECASE):
                        relevance += 3
                    if re.search(r"\b(?:ai|gai|risk|govern|manage|model|system)\b", claim, re.IGNORECASE):
                        relevance += 2
                if intent == "procedure" and re.search(
                    r"\b(?:method|step|first|then|use|assume|choose|guess|initial guess|particular solution|substitut|solv|deriv)\w*\b",
                    claim,
                    re.IGNORECASE,
                ):
                    relevance += 3
                if intent == "procedure" and "method of" in claim.lower() and "involves" in claim.lower():
                    relevance += 4
                if intent == "procedure" and "initial guess" in claim.lower():
                    relevance += 3
                if intent == "procedure" and "exponential" in query_terms:
                    if re.search(r"(?:exponential|e\^|e\u03bb|e\u03b1)", claim, re.IGNORECASE):
                        relevance += 3
                if intent == "procedure" and claim.lower().startswith("when we take derivatives"):
                    relevance -= 2
                candidates.append((relevance, -evidence_index, -sentence_index, claim, item))

        return sorted(candidates, key=lambda candidate: (candidate[0], candidate[1], candidate[2]), reverse=True)

    @classmethod
    def _build_summary_answer(cls, query: str, evidence: List[Dict[str, Any]]) -> str:
        """Present a document summary in a consistent, citation-backed structure."""
        candidates = cls._rank_claims(query, evidence, "summary")
        if not candidates:
            citations = cls._format_citations(evidence)
            return (
                "Relevant summary evidence was found, but it could not be summarized safely. "
                "Review the cited source regions."
                f"\n\nSources: {citations}"
            )

        purpose_pattern = re.compile(r"\b(?:purpose|aim|objective|scope|intended|this (?:document|report|guide)|provides)\b", re.IGNORECASE)
        recommendation_pattern = re.compile(r"\b(?:recommend|should|must|guidance|need to|call for|encourage|suggested actions?)\b", re.IGNORECASE)
        purpose = next((candidate for candidate in candidates if purpose_pattern.search(candidate[3])), candidates[0])
        used_claims = {re.sub(r"\W+", " ", purpose[3].lower()).strip()}
        used_pages = {(purpose[4]["source"], purpose[4]["page"])}

        findings = []
        recommendations = []
        deferred_recommendations = []
        for _, _, _, claim, item in candidates:
            normalized_claim = re.sub(r"\W+", " ", claim.lower()).strip()
            page_key = (item["source"], item["page"])
            if normalized_claim in used_claims or page_key in used_pages:
                continue
            if recommendation_pattern.search(claim):
                recommendation = f"- {claim} {cls._citation_label(item)}"
                if not re.search(r"\b(?:ai|gai|risk|govern|manage|model|system)\b", claim, re.IGNORECASE):
                    deferred_recommendations.append(recommendation)
                    continue
                if len(recommendations) < 2:
                    recommendations.append(recommendation)
                    used_claims.add(normalized_claim)
                    used_pages.add(page_key)
                continue
            if len(findings) < 3:
                findings.append(f"- {claim} {cls._citation_label(item)}")
                used_claims.add(normalized_claim)
                used_pages.add(page_key)

        if not findings:
            findings.append("- The selected evidence is concentrated in the document's purpose statement.")
        if not recommendations:
            recommendations.extend(deferred_recommendations[:1] or ["- No explicit recommendation was found in the selected summary evidence."])

        return "\n".join(
            [
                "**Purpose**",
                "",
                f"{purpose[3]} {cls._citation_label(purpose[4])}",
                "",
                "**Key findings**",
                *findings,
                "",
                "**Recommendations / implications**",
                *recommendations,
                "",
                "**Sources**",
                cls._format_citations(evidence),
            ]
        )

    @classmethod
    def _build_comparison_answer(cls, query: str, evidence: List[Dict[str, Any]]) -> str:
        """Render balanced source-specific evidence when local synthesis is unavailable."""
        source_order = list(dict.fromkeys(item["source"] for item in evidence))
        if len(source_order) < 2:
            return (
                "A supported comparison requires relevant evidence from at least two documents.\n\n"
                f"**Sources**\n{cls._format_citations(evidence)}"
            )

        answer_lines = ["**Source-by-source analysis**"]
        leading_claims: list[tuple[str, str, Dict[str, Any]]] = []
        aspects = cls._compound_aspects(query)
        original_aspect_terms = [cls._support_terms(aspect) for aspect in aspects]
        shared_aspect_terms = set.intersection(*original_aspect_terms) if original_aspect_terms else set()
        distinct_aspect_terms = [
            (cls._expanded_aspect_terms(aspect) - shared_aspect_terms) or cls._expanded_aspect_terms(aspect)
            for aspect in aspects
        ]
        for source in source_order:
            source_evidence = [item for item in evidence if item["source"] == source]
            candidates = cls._rank_claims(query, source_evidence, "comparison")
            if not candidates:
                continue
            label = Path(source).stem.replace("_", " ")
            answer_lines.extend(["", f"**{label} approach**"])
            used_claims = set()
            source_claims: list[tuple[str, Dict[str, Any]]] = []
            if distinct_aspect_terms:
                for aspect, terms in zip(aspects, distinct_aspect_terms):
                    matches = []
                    for relevance, evidence_rank, sentence_rank, claim, item in candidates:
                        normalized_claim = re.sub(r"\W+", " ", claim.casefold()).strip()
                        claim_terms = cls._support_terms(claim)
                        overlap = len(terms.intersection(claim_terms))
                        has_shared_topic = not shared_aspect_terms or bool(
                            shared_aspect_terms.intersection(claim_terms)
                        )
                        if overlap and has_shared_topic:
                            matches.append(
                                (
                                    normalized_claim not in used_claims,
                                    overlap,
                                    relevance,
                                    evidence_rank,
                                    sentence_rank,
                                    normalized_claim,
                                    claim,
                                    item,
                                )
                            )
                    if not matches:
                        continue
                    best = max(matches, key=lambda candidate: candidate[:5])
                    used_claims.add(best[5])
                    source_claims.append((best[6], best[7]))
                    answer_lines.append(
                        f"- **{aspect.title()}:** {best[6]} {cls._citation_label(best[7])}"
                    )
            if not source_claims:
                used_pages = set()
                for _, _, _, claim, item in candidates:
                    if item["page"] in used_pages:
                        continue
                    answer_lines.append(f"- {claim} {cls._citation_label(item)}")
                    used_pages.add(item["page"])
                    source_claims.append((claim, item))
                    if len(source_claims) == 2:
                        break
            if source_claims:
                leading_claims.append((label, source_claims[0][0], source_claims[0][1]))

        if len(leading_claims) >= 2:
            first_label, first_claim, first_item = leading_claims[0]
            second_label, second_claim, second_item = leading_claims[1]
            answer_lines.extend(
                [
                    "",
                    "**Evidence-based contrast**",
                    f"- **{first_label}:** {first_claim} {cls._citation_label(first_item)}",
                    f"- **{second_label}:** {second_claim} {cls._citation_label(second_item)}",
                ]
            )

        answer_lines.extend(["", "**Sources**", cls._format_citations(evidence)])
        return "\n".join(answer_lines)

    @classmethod
    def _build_compound_answer(cls, query: str, evidence: List[Dict[str, Any]]) -> str | None:
        """Answer every explicitly requested aspect with separately cited evidence."""
        aspects = cls._compound_aspects(query)
        if len(aspects) < 2:
            return None
        candidates = cls._rank_claims(query, evidence, "explanation")
        if not candidates:
            return None

        aspect_term_sets = [cls._support_terms(aspect) for aspect in aspects]
        shared_aspect_terms = set.intersection(*aspect_term_sets) if aspect_term_sets else set()
        distinct_aspect_terms = [(terms - shared_aspect_terms) or terms for terms in aspect_term_sets]
        selected: list[tuple[str, str, Dict[str, Any]]] = []
        used_claims = set()
        for aspect, aspect_terms in zip(aspects, distinct_aspect_terms):
            ranked_matches = []
            for relevance, evidence_rank, sentence_rank, claim, item in candidates:
                claim_terms = cls._support_terms(claim)
                overlap = len(aspect_terms.intersection(claim_terms))
                if overlap:
                    normalized_claim = re.sub(r"\W+", " ", claim.casefold()).strip()
                    ranked_matches.append(
                        (
                            normalized_claim not in used_claims,
                            overlap,
                            relevance,
                            evidence_rank,
                            sentence_rank,
                            normalized_claim,
                            claim,
                            item,
                        )
                    )
            if not ranked_matches:
                continue
            best = max(ranked_matches, key=lambda candidate: candidate[:5])
            used_claims.add(best[5])
            selected.append((aspect, best[6], best[7]))

        if len(selected) < 2:
            return None
        answer_lines = ["**Answer**"]
        for aspect, claim, item in selected:
            answer_lines.extend(["", f"**{aspect.title()}**", f"{claim} {cls._citation_label(item)}"])
        answer_lines.extend(["", "**Sources**", cls._format_citations([item for _, _, item in selected])])
        return "\n".join(answer_lines)

    @classmethod
    def _build_grounded_answer(cls, query: str, evidence: List[Dict[str, Any]]) -> str:
        """Create an intent-specific, evidence-only answer with claim-level citations."""
        intent = cls._question_intent(query)
        if intent == "summary":
            return cls._build_summary_answer(query, evidence)
        if intent == "comparison":
            return cls._build_comparison_answer(query, evidence)
        acronym_answer = cls._build_acronym_answer(query, evidence)
        if acronym_answer:
            return acronym_answer
        purpose_answer = cls._build_document_purpose_answer(query, evidence)
        if purpose_answer:
            return purpose_answer
        compound_answer = cls._build_compound_answer(query, evidence)
        if compound_answer:
            return compound_answer
        table_answer = cls._build_table_answer(query, evidence)
        if table_answer:
            return table_answer
        candidates = cls._rank_claims(query, evidence, intent)

        if not candidates:
            citations = cls._format_citations(evidence)
            return (
                "Relevant evidence was found, but its text cannot be summarized safely. "
                "Review the cited source regions and original notation below."
                f"\n\nSources: {citations}"
            )

        direct_claim = candidates[0]
        has_math_evidence = any(item.get("equations") for item in evidence)

        if intent == "procedure":
            answer_lines = ["**Answer**", "", f"{direct_claim[3]} {cls._citation_label(direct_claim[4])}", "", "**Evidence-based steps**"]
            step_claims = []
            used_claims = set()
            for _, _, _, claim, item in candidates:
                normalized_claim = re.sub(r"\W+", " ", claim.lower()).strip()
                if normalized_claim in used_claims:
                    continue
                step_claims.append(f"{len(step_claims) + 1}. {claim} {cls._citation_label(item)}")
                used_claims.add(normalized_claim)
                if len(step_claims) == 3:
                    break
            answer_lines.extend(step_claims)
        else:
            answer_lines = ["**Answer**", "", f"{direct_claim[3]} {cls._citation_label(direct_claim[4])}"]

        supporting_claims = []
        used_pages = {(direct_claim[4]["source"], direct_claim[4]["page"])}
        for _, _, _, claim, item in candidates[1:]:
            page_key = (item["source"], item["page"])
            if page_key in used_pages:
                continue
            supporting_claims.append(f"- {claim} {cls._citation_label(item)}")
            used_pages.add(page_key)
            if len(supporting_claims) == 2:
                break
        if supporting_claims and intent not in {"procedure", "summary"}:
            answer_lines.extend(["", "**Supporting context**", *supporting_claims])
        if has_math_evidence:
            answer_lines.extend(
                [
                    "",
                    "**Mathematical notation**",
                    "The original notation is rendered below from the cited PDF for verification.",
                ]
            )
        return "\n".join(answer_lines)

    def query(self, request: QueryRequest) -> Dict[str, Any]:
        start = time.perf_counter()
        intent = self._question_intent(request.query)
        aspect_queries = self._compound_aspect_queries(request.query)
        math_evidence_requested = self._query_requests_math_evidence(request.query)
        synthesis_mode = "deterministic"
        with self.storage_lock:
            saved_history = self.session_history.get(request.session_id, [])
            conversation_history = request.history or saved_history
            search_query = self._prepare_query(request.query, conversation_history)
            search_query = self._intent_search_query(request.query, search_query)
            requested_source_names = self._requested_source_names(request)
            indexed_source_names = {
                document["filename"]
                for document in getattr(self, "documents", [])
                if document.get("status") == "indexed"
            }
            searchable_source_names = requested_source_names if requested_source_names is not None else indexed_source_names
            named_source_names = self._query_named_source_names(request.query, searchable_source_names)
            if named_source_names and (intent != "comparison" or len(named_source_names) >= 2):
                # A filename-level match is stronger intent than an "all
                # documents" browser selection. This prevents an IPCC summary
                # from mixing unrelated corpus documents. A comparison narrows
                # only when at least two sources are named; one named source
                # may still be compared with another source referenced by
                # title rather than filename.
                requested_source_names = named_source_names
            comparison_source_names = (
                requested_source_names
                if intent == "comparison" and requested_source_names is not None
                else set()
            )
            summary_source_names = (
                requested_source_names
                if intent == "summary" and requested_source_names is not None
                else self._summary_source_names(request)
                if intent == "summary"
                else set()
            )
            initial_hits = self._summary_candidate_pool(summary_source_names) if summary_source_names else []
            purpose_source_names = (
                requested_source_names
                if self._is_document_purpose_question(request.query) and requested_source_names is not None
                else set()
            )
            if not initial_hits and purpose_source_names:
                initial_hits = self._document_purpose_candidate_pool(purpose_source_names)
            dense_backfill_candidates: List[RetrievalResult] = []
        if not initial_hits:
            gate = getattr(self, "inference_gate", nullcontext())
            with gate:
                if aspect_queries and hasattr(self.embedding_store, "encode"):
                    planned_embeddings = self.embedding_store.encode([search_query, *aspect_queries])
                    query_embedding = planned_embeddings[0]
                    aspect_query_embeddings = list(planned_embeddings[1:])
                else:
                    query_embedding = self.embedding_store.encode_single(search_query)
                    aspect_query_embeddings = []
            lexical_query = self._lexical_retrieval_query(request.query, intent)
            with self.storage_lock:
                if requested_source_names == set():
                    initial_hits = []
                else:
                    initial_hits, dense_backfill_candidates = self._hybrid_candidate_pool(
                        query_embedding,
                        lexical_query,
                        requested_source_names,
                        intent,
                        request.top_k,
                        aspect_queries,
                        aspect_query_embeddings,
                    )
        rerank_limit = (
            max(request.top_k, settings.summary_rerank_candidate_k)
            if summary_source_names
            else max(request.top_k, settings.rerank_candidate_k)
        )
        is_multi_document_comparison = (
            intent == "comparison" and requested_source_names is not None and len(requested_source_names) > 1
        )
        if is_multi_document_comparison and aspect_queries:
            # Compound comparisons need room for source x aspect candidates;
            # the normal 30-passage window can truncate a lower-ranked but
            # substantive response passage from a long report.
            rerank_limit = max(rerank_limit, min(settings.retrieval_candidate_k, 48))
        rerank_candidates = (
            self._hybrid_rerank_candidates(
                initial_hits,
                dense_backfill_candidates,
                rerank_limit,
                settings.hybrid_dense_backfill_k,
            )
            if dense_backfill_candidates
            else initial_hits
        )
        pairs = [(item.text + "\n" + item.table + "\n" + item.figure_caption, item) for item in rerank_candidates]
        gate = getattr(self, "inference_gate", nullcontext())
        with gate:
            reranked = self.reranker.rerank(
                # The retrieval query can include presentation hints for the
                # embedding model. A cross-encoder should score the user's
                # actual information need, rather than those instructions.
                request.query,
                pairs,
                top_k=len(pairs) if is_multi_document_comparison else rerank_limit,
            )
        if summary_source_names:
            reranked = self._diversify_summary_reranking(reranked)
        elif is_multi_document_comparison:
            reranked = [
                result
                for result in reranked
                if self._is_comparison_candidate(result[0][1])
            ]
            reranked = self._diversify_comparison_reranking(
                reranked,
                requested_source_names,
                rerank_limit,
            )
            reranked = self._prioritize_comparison_aspects(
                reranked,
                requested_source_names,
                request.query,
                rerank_limit,
            )
            # Aspect prioritization can find more matches in one source than
            # another. Re-apply source round-robin ordering so the final top-k
            # cannot silently lose one side of the comparison.
            reranked = self._diversify_comparison_reranking(
                reranked,
                requested_source_names,
                rerank_limit,
            )
        identifier_pattern = self._structured_identifier_pattern(request.query)
        if identifier_pattern and intent != "comparison":
            reranked = sorted(
                reranked,
                key=lambda result: (
                    bool(re.search(identifier_pattern, result[0][1].text + "\n" + result[0][1].table, re.IGNORECASE)),
                    result[1],
                ),
                reverse=True,
            )

        if not math_evidence_requested:
            # Equation-only chunks and equation attachments are valuable for
            # mathematical questions, but can create convincing-looking,
            # irrelevant evidence for prose research questions.
            reranked = [result for result in reranked if result[0][1].type != "equation"]

        if not reranked:
            evidence = []
            answer = "Unsupported answer: I could not find enough relevant evidence in the uploaded documents."
            unsupported = True
            confidence = 0.0
        else:
            evidence = []
            # Reranker.rerank returns ``((passage, RetrievalResult), score)``.
            # Unpack the candidate before reading its associated result.
            evidence_by_page_and_type: Dict[tuple[str, int, str], Dict[str, Any]] = {}
            for candidate, rerank_score in reranked:
                _, item = candidate
                evidence_key = (item.source, item.metadata.get("page", 1), item.type)
                existing_evidence = evidence_by_page_and_type.get(evidence_key)
                if existing_evidence is not None:
                    # A page can contain a heading in one chunk and its actual
                    # explanation in the next. Preserve both for synthesis,
                    # while still exposing one source card per page and type.
                    for field in ("text", "table", "figure_caption"):
                        content = getattr(item, field)
                        if content and content not in existing_evidence[field]:
                            existing_evidence[field] = f"{existing_evidence[field]} {content}".strip()
                    if math_evidence_requested:
                        for equation in item.equations:
                            if equation not in existing_evidence["equations"]:
                                existing_evidence["equations"].append(equation)
                    existing_evidence["score"] = max(existing_evidence["score"], round(float(rerank_score), 4))
                    continue
                if len(evidence) >= request.top_k:
                    continue
                evidence.append(
                    {
                        "chunk_id": item.chunk_id,
                        "score": round(float(rerank_score), 4),
                        "source": item.source,
                        "page": item.metadata.get("page", 1),
                        "text": item.text,
                        "table": item.table,
                        "figure_caption": item.figure_caption,
                        "section": item.section,
                        "equations": (
                            item.equations
                            if math_evidence_requested and not (intent == "summary" and not self._is_summary_candidate(item))
                            else []
                        ),
                        "type": item.type,
                        "bounding_box": item.metadata.get("bounding_box"),
                        "quality_flags": item.metadata.get("quality_flags", []),
                    }
                )
                evidence_by_page_and_type[evidence_key] = evidence[-1]
            best_score = float(reranked[0][1])
            # Reranker scores rank candidates but are not calibrated
            # probabilities. Do not reject a directly supported passage just
            # because a particular model emits negative logits.
            rank_confidence = min(1.0, max(0.0, (best_score + 0.5) / 1.5))
            has_document_summary_coverage = bool(summary_source_names) and self._has_summary_coverage(evidence, request.top_k)
            has_direct_evidence = self._evidence_supports_query(request.query, evidence, intent)
            has_comparison_coverage = (
                not comparison_source_names
                or comparison_source_names.issubset({item["source"] for item in evidence})
            )
            supported = has_document_summary_coverage or (has_direct_evidence and has_comparison_coverage)
            confidence = max(rank_confidence, 0.5) if supported else 0.0
            unsupported = not supported
            if unsupported:
                answer = "Unsupported answer: the retrieved evidence is too weak to support a confident response."
            else:
                answer_evidence = self._prioritize_aspect_evidence(request.query, evidence)[
                    : settings.max_answer_sentences
                ]
                synthesis = getattr(self, "synthesizer", None)
                if synthesis is not None:
                    gate = getattr(self, "inference_gate", nullcontext())
                    with gate:
                        synthesized = synthesis.synthesize(
                            query=request.query,
                            intent=intent,
                            evidence=answer_evidence,
                            required_aspects=self._compound_aspects(request.query),
                        )
                else:
                    synthesized = None
                if synthesized is not None:
                    answer = synthesized.answer
                    synthesis_mode = synthesized.provider
                else:
                    answer = self._build_grounded_answer(request.query, answer_evidence)

        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        with self.storage_lock:
            history = self.session_history.setdefault(request.session_id, [])
            history.append(request.query)
            del history[:-settings.max_session_history]
            while len(self.session_history) > settings.max_session_count:
                self.session_history.pop(next(iter(self.session_history)))
        return {
            "answer": answer,
            "unsupported": unsupported,
            "confidence": round(confidence, 4),
            "sources": evidence,
            "retrieval_scores": [item["score"] for item in evidence],
            "latency_ms": latency_ms,
            "session_id": request.session_id,
            "history": history,
            "answer_intent": intent,
            "synthesis_mode": synthesis_mode,
        }

    @staticmethod
    def _read_evaluation_artifact(path: Path) -> Dict[str, Any]:
        """Read a generated evaluation artifact without making metrics up."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def summary_metrics() -> Dict[str, Any]:
        """Expose measured benchmark values, or ``None`` when not evaluated.

        The dashboard must never imply that a hand-entered placeholder is a
        measured retrieval, citation, or faithfulness result.  Retrieval and
        automatic provenance values are read from their reproducible benchmark
        artifacts.  Faithfulness remains unavailable until the separate
        semantic review artifact has been generated.
        """
        evaluation_dir = settings.project_root / "evaluation" / "predictions"
        retrieval = RAGService._read_evaluation_artifact(evaluation_dir / "metrics.json")
        answer = RAGService._read_evaluation_artifact(evaluation_dir / "answer_metrics.json")
        semantic = RAGService._read_evaluation_artifact(evaluation_dir / "semantic_review.json")

        retrieval_rows = retrieval.get("hybrid_reranked", {}).get("5", {})
        answer_rows = answer.get("automatic_metrics", {}).get("deterministic", {})
        semantic_rows = semantic.get("modes", {}).get("deterministic", {})
        configurations = {
            "baseline_dense": retrieval.get("baseline", {}).get("5", {}),
            "dense_plus_reranker": retrieval.get("reranked", {}).get("5", {}),
            "hybrid_bm25_faiss": retrieval.get("hybrid", {}).get("5", {}),
            "hybrid_plus_reranker": retrieval_rows,
        }
        comparison = {
            name: {
                "recall@5": float(values["recall@5"]),
                "mrr": float(values["mrr"]),
            }
            for name, values in configurations.items()
            if isinstance(values, dict) and "recall@5" in values and "mrr" in values
        }
        return {
            "recall_at_5": retrieval_rows.get("recall@5"),
            "mrr": retrieval_rows.get("mrr"),
            "citation_accuracy": answer_rows.get("citation_reference_validity"),
            "answer_faithfulness": semantic_rows.get("criterion_rates", {}).get("faithfulness"),
            "latency_ms": answer_rows.get("latency_ms", {}).get("p95"),
            "comparison": comparison,
        }


service: RAGService | None = None
service_lock = Lock()


def get_service() -> RAGService:
    """Initialize models only when an endpoint needs retrieval."""
    global service
    if service is None:
        with service_lock:
            if service is None:
                service = RAGService()
    return service


@asynccontextmanager
async def application_lifespan(_app: FastAPI):
    yield
    global service
    with service_lock:
        current_service = service
        service = None
    if current_service is not None:
        current_service.close()


app = FastAPI(
    title="Multimodal RAG Research Assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=application_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

_rate_limit_lock = Lock()
_rate_limit_windows: Dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def security_controls(request: Request, call_next):
    """Apply optional authentication, bounded abuse controls, and headers."""
    public_paths = {"/health", "/ready"}
    if settings.api_key and request.url.path not in public_paths and request.method != "OPTIONS":
        supplied_key = request.headers.get("x-api-key", "")
        authorization = request.headers.get("authorization", "")
        if authorization.casefold().startswith("bearer "):
            supplied_key = authorization[7:].strip()
        if not supplied_key or not hmac.compare_digest(supplied_key, settings.api_key):
            return JSONResponse(status_code=401, content={"detail": "Authentication is required."})

    client_host = request.client.host if request.client else "unknown"
    is_expensive = (
        request.url.path in {"/upload", "/query"}
        or request.url.path.endswith("/query")
        or request.url.path.endswith("/page-preview")
    )
    route_group = "expensive" if is_expensive else "general"
    maximum_requests = 60 if is_expensive else 240
    now = time.monotonic()
    rate_key = f"{client_host}:{route_group}"
    with _rate_limit_lock:
        window = _rate_limit_windows[rate_key]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= maximum_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Request rate limit exceeded. Retry shortly."},
                headers={"Retry-After": "60"},
            )
        window.append(now)
        if len(_rate_limit_windows) > 1_000:
            for key in [key for key, values in _rate_limit_windows.items() if not values or now - values[-1] >= 60]:
                _rate_limit_windows.pop(key, None)
            while len(_rate_limit_windows) > 1_000:
                _rate_limit_windows.pop(next(iter(_rate_limit_windows)))

    response = await call_next(request)
    is_embeddable_source_file = request.url.path.startswith("/documents/") and request.url.path.endswith("/file")
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith(("/documents", "/jobs", "/session")) or request.url.path in {"/upload", "/query"}:
        response.headers["Cache-Control"] = "private, no-store"
    if is_embeddable_source_file:
        # The frontend embeds original PDFs for citation verification. X-Frame-
        # Options cannot express an allowlist across the separate frontend and
        # API origins, so use CSP's frame-ancestors directive for this narrow
        # route instead. All other responses remain unframeable.
        frame_ancestors = " ".join(settings.cors_origins) or "'none'"
        response.headers["Content-Security-Policy"] = f"frame-ancestors {frame_ancestors}"
    else:
        response.headers["X-Frame-Options"] = "DENY"
    if request.url.path not in {"/docs", "/redoc", "/openapi.json"} and not is_embeddable_source_file:
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'; sandbox"
    return response


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def readiness() -> Dict[str, Any]:
    """Verify that models and persisted retrieval state can serve queries."""
    try:
        current_service = get_service()
        if current_service.retriever.index.ntotal != len(current_service.retriever.chunks):
            raise RuntimeError("The dense index and chunk metadata are inconsistent.")
    except Exception as exc:
        logger.exception("Readiness check failed")
        raise HTTPException(status_code=503, detail="The retrieval service is not ready.") from exc
    return {
        "status": "ready",
        "indexed_chunks": int(current_service.retriever.index.ntotal),
        "indexed_documents": sum(document.get("status") == "indexed" for document in current_service.documents),
    }


@app.get("/math/status", response_model=MathOcrStatus)
def math_ocr_status() -> Dict[str, Any]:
    checkpoint_available = settings.math_ocr_checkpoint.is_file()
    return {
        "enabled": settings.math_ocr_enabled,
        "checkpoint_available": checkpoint_available,
        "checkpoint_path": settings.math_ocr_checkpoint.name if checkpoint_available else "",
        "mode": "local_ocr" if settings.math_ocr_enabled and checkpoint_available else "source_verification_only",
    }


@app.post("/upload", response_model=UploadResponse)
def upload_documents(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")
    if len(files) > settings.max_upload_files:
        raise HTTPException(status_code=413, detail=f"Upload at most {settings.max_upload_files} PDF files at a time.")
    return get_service().upload_documents(files)


@app.get("/documents", response_model=List[DocumentInfo], response_model_exclude_none=True)
def list_documents() -> List[Dict[str, Any]]:
    return RAGService.load_persisted_documents()


@app.get("/documents/{filename}/file")
def get_document_file(filename: str) -> FileResponse:
    """Serve an uploaded PDF for the source viewer."""
    safe_filename = RAGService._validated_upload_filename(filename)

    document_path = settings.uploads_dir / safe_filename
    if not document_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="The original PDF is not available. Upload the document again to view it.",
        )

    return FileResponse(
        document_path,
        media_type="application/pdf",
        filename=safe_filename,
        content_disposition_type="inline",
    )


@app.get("/documents/{filename}/page-preview")
def get_document_region_preview(
    filename: str,
    page: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> Response:
    """Render the cited page region as an image for in-app source verification."""
    safe_filename = RAGService._validated_upload_filename(filename)
    bounding_box = [x0, y0, x1, y1]
    if not all(math.isfinite(value) for value in bounding_box) or x1 <= x0 or y1 <= y0:
        raise HTTPException(status_code=400, detail="Invalid cited region.")

    document_path = settings.uploads_dir / safe_filename
    if not document_path.is_file():
        raise HTTPException(status_code=404, detail="The original PDF is not available.")
    try:
        image = render_pdf_region(
            document_path,
            page,
            bounding_box,
            max_pixels=settings.max_preview_pixels,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - malformed external PDF
        logger.exception("Could not render source preview for %s", safe_filename)
        raise HTTPException(status_code=422, detail="Could not render the cited PDF region.") from exc
    return Response(content=image, media_type="image/png")


@app.delete("/documents/{filename}", response_model=DeleteDocumentResponse)
def delete_document(filename: str) -> Dict[str, Any]:
    return get_service().delete_document(filename)


@app.post("/documents/{filename}/retry", response_model=IngestionJobResponse)
def retry_document(filename: str) -> Dict[str, Any]:
    return get_service().retry_document(filename)


@app.get("/jobs/{job_id}", response_model=IngestionJobResponse)
def get_ingestion_job(job_id: str) -> Dict[str, Any]:
    return get_service().get_job(job_id)


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> Dict[str, Any]:
    try:
        return get_service().query(request)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail="Query processing failed. Check the backend logs for details.") from exc


@app.get("/metrics", response_model=EvaluationSummary)
def metrics() -> Dict[str, Any]:
    return RAGService.summary_metrics()


@app.get("/compare", response_model=EvaluationSummary)
def compare() -> Dict[str, Any]:
    return RAGService.summary_metrics()


@app.post("/session/{session_id}/query", response_model=QueryResponse)
def query_session(session_id: str, request: QueryRequest) -> Dict[str, Any]:
    if not session_id or len(session_id) > 100 or any(ord(character) < 32 for character in session_id):
        raise HTTPException(status_code=422, detail="Invalid session identifier.")
    request.session_id = session_id
    return get_service().query(request)
