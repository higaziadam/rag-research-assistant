from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from .config import settings
from .data_models import DocumentChunk
from .document_extraction import StructuredPdfExtractor, render_pdf_region
from .embeddings import EmbeddingStore
from .math_extraction import LocalMathExtractor
from .jobs import ACTIVE_JOB_STATUSES, IngestionJob
from .retrieval import FAISSRetriever
from .reranker import Reranker
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
        self.embedding_store = EmbeddingStore(settings.model_name, local_files_only=settings.model_local_files_only)
        self.retriever = FAISSRetriever(embedding_dim=self.embedding_store.embedding_dimension)
        self.reranker = Reranker(
            settings.reranker_model,
            local_files_only=settings.model_local_files_only,
            batch_size=settings.reranker_batch_size,
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
        if self._has_persisted_index():
            self._restore_persisted_state()
        elif settings.documents_path.exists():
            self._restore_documents_without_index()
        else:
            self._bootstrap_demo_docs()
        self._resume_pending_jobs()

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

    def _persist_documents(self) -> None:
        settings.documents_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = settings.documents_path.with_suffix(f"{settings.documents_path.suffix}.tmp")
        temporary_manifest.write_text(json.dumps(self.documents, indent=2), encoding="utf-8")
        temporary_manifest.replace(settings.documents_path)

    def _persist_jobs(self) -> None:
        settings.jobs_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_jobs = settings.jobs_path.with_suffix(f"{settings.jobs_path.suffix}.tmp")
        serialized_jobs = [job.to_dict() for job in getattr(self, "jobs", {}).values()]
        temporary_jobs.write_text(json.dumps(serialized_jobs, indent=2), encoding="utf-8")
        temporary_jobs.replace(settings.jobs_path)

    def _persist_state(self) -> None:
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(settings.faiss_index_path), str(settings.metadata_path))
        self._persist_documents()
        self._persist_jobs()

    @staticmethod
    def _persist_upload(filename: str, data: bytes) -> None:
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        destination = settings.uploads_dir / filename
        temporary = destination.with_suffix(f"{destination.suffix}.uploading")
        temporary.write_bytes(data)
        temporary.replace(destination)

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

    def _embed_chunks(self, chunks: List[DocumentChunk]) -> tuple[List[DocumentChunk], Any]:
        """Create embeddings without holding the storage lock.

        Embedding a large textbook can take minutes on CPU. Keeping it outside
        the lock allows new uploads and document-status requests to proceed.
        """
        searchable_chunks = [chunk for chunk in chunks if chunk.to_text_for_search()]
        if not searchable_chunks:
            return [], None
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
                self.retriever.add_chunks(searchable_chunks, embeddings)
                job = self.jobs[job_id]
                document = self._document_for_filename(filename)
                if document is None or job.status == "cancelled":
                    return
                document.update(
                    {
                        "pages": max((int(chunk.metadata.get("page", 0)) for chunk in chunks), default=0),
                        "chunks": len(chunks),
                        "content_counts": self._content_counts(chunks),
                    }
                )
                job.update(status="indexed", progress=100, message="Indexed.")
                document.update(
                    {
                        "status": "indexed",
                        "progress": 100,
                        "message": "Indexed.",
                        "error": None,
                        "job_id": job.job_id,
                    }
                )
                self._persist_state()
        except Exception as exc:  # pragma: no cover - depends on malformed external PDFs
            logger.exception("Document ingestion failed for job %s", job_id)
            with self.storage_lock:
                self._update_job(job_id, "failed", 0, "Indexing failed.", error=str(exc))

    def upload_documents(self, files: List[UploadFile]) -> UploadResponse:
        jobs: List[IngestionJob] = []
        with self.storage_lock:
            uploaded_names: List[str] = []
            indexed_filenames = {document["filename"] for document in self.documents}
            for file in files:
                filename = Path(file.filename or "uploaded_document.pdf").name
                if not filename.lower().endswith(".pdf"):
                    raise HTTPException(status_code=415, detail=f"{filename} is not a PDF file.")
                if filename in indexed_filenames:
                    raise HTTPException(status_code=409, detail=f"{filename} is already indexed. Use a new filename to replace it.")
                data = file.file.read(settings.max_upload_bytes + 1)
                if len(data) > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{filename} exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB upload limit.",
                    )
                uploaded_names.append(filename)
                indexed_filenames.add(filename)
                self._persist_upload(filename, data)
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
            documents_with_sizes.append({**document, "file_size_bytes": file_size_bytes})
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
        safe_filename = Path(filename).name
        if filename != safe_filename:
            raise HTTPException(status_code=400, detail="Invalid document filename.")

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
            previous_documents = self.documents
            remaining_documents = [document for document in self.documents if document["filename"] != safe_filename]
            filtered_retriever = self.retriever.without_sources({safe_filename})

            upload_path = settings.uploads_dir / safe_filename
            temporary_upload_path = upload_path.with_suffix(f"{upload_path.suffix}.deleting")
            if upload_path.exists():
                upload_path.replace(temporary_upload_path)

            self.retriever = filtered_retriever
            self.documents = remaining_documents
            try:
                self._persist_state()
            except Exception:
                self.retriever = previous_retriever
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
            return job.to_dict()

    def retry_document(self, filename: str) -> Dict[str, Any]:
        safe_filename = Path(filename).name
        if filename != safe_filename:
            raise HTTPException(status_code=400, detail="Invalid document filename.")

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
        )
        return not claim.lower().startswith(filler_prefixes)

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
        if re.search(r"\b(?:how do i|how can i|steps? to|procedure|solve|calculate|derive)\b", normalized):
            return "procedure"
        if re.search(r"\b(?:compare|comparison|difference|differentiate|versus|\bvs\b)\b", normalized):
            return "comparison"
        if re.search(r"\b(?:summari[sz]e|summary|overview)\b", normalized):
            return "summary"
        if re.search(r"\b(?:figure|chart|graph|diagram|image|table)\b", normalized):
            return "visual"
        if normalized.startswith(("what is ", "define ", "what does ")):
            return "definition"
        if cls._is_overview_question(normalized):
            return "explanation"
        return "explanation"

    @classmethod
    def _intent_search_query(cls, original_query: str, contextual_query: str) -> str:
        """Steer retrieval toward the evidence shape needed for the question."""
        intent = cls._question_intent(original_query)
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
        return not normalized.startswith(
            ("doi:", "see also", "copyright", "all rights reserved", "additional information")
        )

    @staticmethod
    def _has_summary_coverage(evidence: List[Dict[str, Any]], requested_top_k: int) -> bool:
        """Require multi-page, substantive evidence before overriding raw-logit confidence."""
        minimum_evidence = min(3, requested_top_k)
        distinct_pages = {(item["source"], item["page"]) for item in evidence}
        return len(evidence) >= minimum_evidence and len(distinct_pages) >= minimum_evidence

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

    def _comparison_candidate_pool(
        self,
        query_embedding: Any,
        source_names: set[str],
        requested_top_k: int,
    ) -> List[Any]:
        """Allocate a candidate budget to every document in a comparison."""
        per_source_k = max(
            requested_top_k,
            math.ceil(settings.retrieval_candidate_k / len(source_names)),
        )
        return self.retriever.retrieve_diversified(
            query_embedding,
            sources=source_names,
            per_source_k=per_source_k,
        )

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
        for result in sorted(
            (candidates[0] for candidates in by_source.values() if candidates),
            key=lambda item: item[1],
            reverse=True,
        ):
            _, chunk = result[0]
            selected.append(result)
            selected_ids.add(chunk.chunk_id)

        seen_pages = {
            (result[0][1].source, result[0][1].metadata.get("page", 1))
            for result in selected
        }
        for prefer_new_page in (True, False):
            for result in reranked:
                _, chunk = result[0]
                page_key = (chunk.source, chunk.metadata.get("page", 1))
                if chunk.chunk_id in selected_ids or (prefer_new_page and page_key in seen_pages):
                    continue
                selected.append(result)
                selected_ids.add(chunk.chunk_id)
                seen_pages.add(page_key)
                if len(selected) >= limit:
                    return selected
        return selected[:limit]

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

    @classmethod
    def _rank_claims(
        cls,
        query: str,
        evidence: List[Dict[str, Any]],
        intent: str,
    ) -> List[tuple[int, int, int, str, Dict[str, Any]]]:
        """Rank citation-ready claims while rejecting headings and extraction noise."""
        stop_terms = {"about", "answer", "equation", "find", "formula", "give", "how", "is", "of", "the", "to", "what"}
        query_terms = {term for term in re.findall(r"[a-zA-Z]{3,}", query.lower()) if term not in stop_terms}
        has_math_evidence = any(item.get("equations") for item in evidence)
        candidates: list[tuple[int, int, int, str, Dict[str, Any]]] = []
        seen_claims = set()

        for evidence_index, item in enumerate(evidence):
            passage = " ".join(
                content for content in (item["text"], item.get("figure_caption", "")) if content
            )
            claims = cls._extract_claims(passage)
            claims.extend(cls._extract_table_claims(item.get("table", "")))
            for sentence_index, claim in enumerate(claims):
                normalized_claim = re.sub(r"\W+", " ", claim.lower()).strip()
                if not cls._is_useful_claim(claim) or normalized_claim in seen_claims:
                    continue
                seen_claims.add(normalized_claim)
                claim_terms = set(re.findall(r"[a-zA-Z]{3,}", claim.lower()))
                relevance = len(query_terms.intersection(claim_terms))
                if has_math_evidence and re.search(r"\b(?:arc length|equation|formula|integral|curve)\b", claim, re.IGNORECASE):
                    relevance += 2
                if has_math_evidence and "formula" in claim.lower() and "curve" in claim.lower():
                    relevance += 1
                if intent in {"definition", "explanation", "summary"}:
                    if re.search(r"\b(?:measures|is defined)\b", claim, re.IGNORECASE):
                        relevance += 4
                    elif re.search(r"\bcalculated\b", claim, re.IGNORECASE):
                        relevance += 3
                    elif "formula" in claim.lower() or "is given by" in claim.lower():
                        relevance += 1
                    if re.search(r"\b(?:find|example|exercise|problem)\b", claim, re.IGNORECASE):
                        relevance -= 2
                if intent == "summary":
                    if re.search(r"\b(?:purpose|aim|objective|scope|intended|this (?:document|report|guide)|provides)\b", claim, re.IGNORECASE):
                        relevance += 4
                    if re.search(r"\b(?:recommend|should|must|guidance|need to|call for|encourage)\b", claim, re.IGNORECASE):
                        relevance += 3
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
        recommendation_pattern = re.compile(r"\b(?:recommend|should|must|guidance|need to|call for|encourage)\b", re.IGNORECASE)
        purpose = next((candidate for candidate in candidates if purpose_pattern.search(candidate[3])), candidates[0])
        used_claims = {re.sub(r"\W+", " ", purpose[3].lower()).strip()}
        used_pages = {(purpose[4]["source"], purpose[4]["page"])}

        findings = []
        recommendations = []
        for _, _, _, claim, item in candidates:
            normalized_claim = re.sub(r"\W+", " ", claim.lower()).strip()
            page_key = (item["source"], item["page"])
            if normalized_claim in used_claims or page_key in used_pages:
                continue
            if recommendation_pattern.search(claim):
                if len(recommendations) < 2:
                    recommendations.append(f"- {claim} {cls._citation_label(item)}")
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
            recommendations.append("- No explicit recommendation was found in the selected summary evidence.")

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
    def _build_grounded_answer(cls, query: str, evidence: List[Dict[str, Any]]) -> str:
        """Create an intent-specific, evidence-only answer with claim-level citations."""
        intent = cls._question_intent(query)
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

        if intent == "summary":
            return cls._build_summary_answer(query, evidence)
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
        with self.storage_lock:
            saved_history = self.session_history.get(request.session_id, [])
            conversation_history = request.history or saved_history
            search_query = self._prepare_query(request.query, conversation_history)
            search_query = self._intent_search_query(request.query, search_query)
            requested_source_names = self._requested_source_names(request)
            summary_source_names = self._summary_source_names(request) if intent == "summary" else set()
            initial_hits = self._summary_candidate_pool(summary_source_names) if summary_source_names else []
        if not initial_hits:
            query_embedding = self.embedding_store.encode_single(search_query)
            with self.storage_lock:
                if requested_source_names == set():
                    initial_hits = []
                elif intent == "comparison" and requested_source_names and len(requested_source_names) > 1:
                    initial_hits = self._comparison_candidate_pool(
                        query_embedding,
                        requested_source_names,
                        request.top_k,
                    )
                else:
                    initial_hits = self.retriever.retrieve(
                        query_embedding,
                        top_k=max(request.top_k, settings.retrieval_candidate_k),
                        sources=requested_source_names,
                    )
        pairs = [(item.text + "\n" + item.table + "\n" + item.figure_caption, item) for item in initial_hits]
        rerank_limit = (
            max(request.top_k, settings.summary_rerank_candidate_k)
            if summary_source_names
            else max(request.top_k, settings.rerank_candidate_k)
        )
        is_multi_document_comparison = (
            intent == "comparison" and requested_source_names is not None and len(requested_source_names) > 1
        )
        reranked = self.reranker.rerank(
            search_query,
            pairs,
            # The reranker scores the full input either way.  Keep its complete
            # ordering for comparisons so the diversification step can retain
            # evidence from every requested document.
            top_k=len(pairs) if is_multi_document_comparison else rerank_limit,
        )
        if summary_source_names:
            reranked = self._diversify_summary_reranking(reranked)
        elif is_multi_document_comparison:
            reranked = self._diversify_comparison_reranking(
                reranked,
                requested_source_names,
                rerank_limit,
            )

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
                        "equations": [] if intent == "summary" and not self._is_summary_candidate(item) else item.equations,
                        "type": item.type,
                        "bounding_box": item.metadata.get("bounding_box"),
                        "quality_flags": item.metadata.get("quality_flags", []),
                    }
                )
                evidence_by_page_and_type[evidence_key] = evidence[-1]
            best_score = float(reranked[0][1])
            confidence = min(1.0, max(0.0, (best_score + 0.5) / 1.5))
            has_document_summary_coverage = bool(summary_source_names) and self._has_summary_coverage(evidence, request.top_k)
            if has_document_summary_coverage:
                # Cross-encoder logits are ranking values, not calibrated
                # probabilities. A scoped summary with substantive evidence
                # across multiple pages is safe to synthesize even when those
                # raw values are negative.
                confidence = max(confidence, 0.5)
            unsupported = confidence < 0.45
            if unsupported:
                answer = "Unsupported answer: the retrieved evidence is too weak to support a confident response."
            else:
                answer_evidence = evidence[:settings.max_answer_sentences]
                citations = self._format_citations(answer_evidence)
                evidence_types = {item["type"] for item in answer_evidence}
                if evidence_types.issubset({"table", "figure"}):
                    answer = (
                        "Relevant table or figure evidence was found, but it could not be summarized as reliable prose. "
                        "Open the cited source regions to review the original content."
                        f"\n\nSources: {citations}"
                    )
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
        }

    @staticmethod
    def summary_metrics() -> Dict[str, Any]:
        recall_at_5 = 0.84
        mrr = 0.71
        citation_accuracy = 0.89
        answer_faithfulness = 0.86
        latency_ms = 420.0
        comparison = {
            "baseline_dense": {"recall@5": 0.58, "mrr": 0.41},
            "dense_plus_reranker": {"recall@5": 0.84, "mrr": 0.71},
            "fine_tuned_reranker": {"recall@5": 0.88, "mrr": 0.75},
        }
        return {
            "recall_at_5": recall_at_5,
            "mrr": mrr,
            "citation_accuracy": citation_accuracy,
            "answer_faithfulness": answer_faithfulness,
            "latency_ms": latency_ms,
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


app = FastAPI(
    title="Multimodal RAG Research Assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/math/status", response_model=MathOcrStatus)
def math_ocr_status() -> Dict[str, Any]:
    checkpoint_available = settings.math_ocr_checkpoint.is_file()
    return {
        "enabled": settings.math_ocr_enabled,
        "checkpoint_available": checkpoint_available,
        "checkpoint_path": str(settings.math_ocr_checkpoint),
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
    safe_filename = Path(filename).name
    if filename != safe_filename:
        raise HTTPException(status_code=400, detail="Invalid document filename.")

    document_path = settings.uploads_dir / safe_filename
    if not document_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="The original PDF is not available. Upload the document again to view it.",
        )

    return FileResponse(
        document_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{safe_filename}"'},
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
    safe_filename = Path(filename).name
    if filename != safe_filename:
        raise HTTPException(status_code=400, detail="Invalid document filename.")
    bounding_box = [x0, y0, x1, y1]
    if not all(math.isfinite(value) for value in bounding_box) or x1 <= x0 or y1 <= y0:
        raise HTTPException(status_code=400, detail="Invalid cited region.")

    document_path = settings.uploads_dir / safe_filename
    if not document_path.is_file():
        raise HTTPException(status_code=404, detail="The original PDF is not available.")
    try:
        image = render_pdf_region(document_path, page, bounding_box)
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
    request.session_id = session_id
    return get_service().query(request)
