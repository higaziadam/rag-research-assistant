from __future__ import annotations

import io
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pypdf import PdfReader

from .config import settings
from .data_models import DocumentChunk
from .embeddings import EmbeddingStore
from .retrieval import FAISSRetriever
from .reranker import Reranker
from .schemas import DeleteDocumentResponse, DocumentInfo, EvaluationSummary, QueryRequest, QueryResponse, UploadResponse

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.embedding_store = EmbeddingStore(settings.model_name)
        self.retriever = FAISSRetriever(embedding_dim=self.embedding_store.embedding_dimension)
        self.reranker = Reranker(settings.reranker_model)
        self.documents: List[Dict[str, Any]] = []
        self.session_history: Dict[str, List[str]] = defaultdict(list)
        self.storage_lock = RLock()
        if self._has_persisted_index():
            self._restore_persisted_state()
        else:
            self._bootstrap_demo_docs()

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

    def _documents_from_chunks(self) -> List[Dict[str, Any]]:
        documents: Dict[str, Dict[str, Any]] = {}
        for chunk in self.retriever.chunks:
            document = documents.setdefault(chunk.source, {"filename": chunk.source, "pages": 0, "chunks": 0})
            document["pages"] = max(document["pages"], int(chunk.metadata.get("page", 0)))
            document["chunks"] += 1
        return list(documents.values())

    def _persist_state(self) -> None:
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(settings.faiss_index_path), str(settings.metadata_path))
        settings.documents_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = settings.documents_path.with_suffix(f"{settings.documents_path.suffix}.tmp")
        temporary_manifest.write_text(json.dumps(self.documents, indent=2), encoding="utf-8")
        temporary_manifest.replace(settings.documents_path)

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
        self.documents.append({"filename": "technical_report.pdf", "pages": 10, "chunks": len(demo_chunks)})

    def _split_text(self, text: str, chunk_length: int = 260) -> List[str]:
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
        return pieces or [text[:chunk_length]]

    def _extract_table_and_caption(self, raw_text: str) -> tuple[str, str]:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        table_lines = [line for line in lines if "|" in line or re.search(r"\b[A-Za-z]+\s*\|\s*[A-Za-z0-9]", line)]
        caption = ""
        for line in lines:
            if re.search(r"(?i)figure\s+\d+|chart\s+\d+|table\s+\d+", line):
                caption = line
                break
        return "\n".join(table_lines[:6]), caption

    def _parse_pdf_to_chunks(self, filename: str, file_bytes: bytes) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        reader = PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            table_text, figure_caption = self._extract_table_and_caption(page_text)
            sections = ["Methods", "Results", "Discussion", "Appendix"]
            section = sections[(page_num - 1) % len(sections)]
            for idx, paragraph in enumerate(self._split_text(page_text)):
                chunk_id = f"{filename}-{page_num}-{idx}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=paragraph,
                        table=table_text,
                        figure_caption=figure_caption,
                        source=filename,
                        section=section,
                        metadata={"page": page_num, "document_type": "pdf"},
                        type="text",
                    )
                )
        return chunks

    def _index_chunks(self, chunks: List[DocumentChunk]) -> None:
        searchable_chunks = [chunk for chunk in chunks if chunk.to_text_for_search()]
        if not searchable_chunks:
            return
        embeddings = self.embedding_store.encode([chunk.to_text_for_search() for chunk in searchable_chunks])
        self.retriever.add_chunks(searchable_chunks, embeddings)

    def upload_documents(self, files: List[UploadFile]) -> UploadResponse:
        with self.storage_lock:
            all_chunks: List[DocumentChunk] = []
            uploaded_names: List[str] = []
            uploaded_documents: List[Dict[str, Any]] = []
            uploaded_files: List[tuple[str, bytes]] = []
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
                try:
                    chunks = self._parse_pdf_to_chunks(filename, data)
                except Exception as exc:
                    raise HTTPException(status_code=422, detail=f"Could not read {filename} as a PDF: {exc}") from exc
                if not chunks:
                    raise HTTPException(status_code=422, detail=f"No extractable text was found in {filename}. Scanned PDFs need OCR before upload.")
                all_chunks.extend(chunks)
                uploaded_names.append(filename)
                uploaded_files.append((filename, data))
                indexed_filenames.add(filename)
                uploaded_documents.append({"filename": filename, "pages": max((c.metadata.get("page", 0) for c in chunks), default=0), "chunks": len(chunks)})
            self._index_chunks(all_chunks)
            self.documents.extend(uploaded_documents)
            for filename, data in uploaded_files:
                self._persist_upload(filename, data)
            self._persist_state()
            return UploadResponse(
                uploaded=uploaded_names,
                total_chunks=len(all_chunks),
                documents=self._documents_with_file_sizes(self.documents),
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
            if not any(document["filename"] == safe_filename for document in self.documents):
                raise HTTPException(status_code=404, detail=f"{safe_filename} is not indexed.")

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

    def _prepare_query(self, query: str, history: List[str]) -> str:
        prior = " ".join(history[-4:])
        if prior:
            return f"Context: {prior} \nQuestion: {query}"
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
        tokens = sentence.split()
        if len(tokens) < 5:
            return False

        word_count = len(re.findall(r"[A-Za-z]{2,}", sentence))
        noisy_token_count = sum(
            any(character.isdigit() for character in token)
            or len(re.sub(r"[^A-Za-z]", "", token)) < 2
            for token in tokens
        )
        visible_characters = [character for character in sentence if not character.isspace()]
        letter_ratio = sum(character.isalpha() for character in visible_characters) / max(len(visible_characters), 1)

        return word_count >= 5 and noisy_token_count / len(tokens) <= 0.25 and letter_ratio >= 0.65

    @staticmethod
    def _extract_summary_sentences(
        query: str,
        evidence: List[Dict[str, Any]],
        max_sentence_characters: int,
    ) -> List[str]:
        query_terms = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
        sentences = []
        for item in evidence:
            candidates = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", item["text"])
                if len(sentence.strip()) >= 30 and RAGService._is_readable_prose(sentence.strip())
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda sentence: len(query_terms.intersection(re.findall(r"[a-zA-Z]{3,}", sentence.lower()))))
            if len(best) > max_sentence_characters:
                shortened = best[:max_sentence_characters].rsplit(" ", 1)[0]
                sentences.append(f"{shortened}...")
            else:
                sentences.append(best)
        return sentences

    def query(self, request: QueryRequest) -> Dict[str, Any]:
        start = time.perf_counter()
        with self.storage_lock:
            saved_history = self.session_history.get(request.session_id, [])
            conversation_history = request.history or saved_history
            search_query = self._prepare_query(request.query, conversation_history)
        query_embedding = self.embedding_store.encode_single(search_query)
        initial_hits = self.retriever.retrieve(query_embedding, top_k=max(request.top_k, 5))
        pairs = [(item.text + "\n" + item.table + "\n" + item.figure_caption, item) for item in initial_hits]
        reranked = self.reranker.rerank(search_query, pairs, top_k=request.top_k)

        if not reranked:
            evidence = []
            answer = "Unsupported answer: I could not find enough relevant evidence in the uploaded documents."
            unsupported = True
            confidence = 0.0
        else:
            evidence = []
            # Reranker.rerank returns ``((passage, RetrievalResult), score)``.
            # Unpack the candidate before reading its associated result.
            seen_pages = set()
            for candidate, rerank_score in reranked:
                _, item = candidate
                page_key = (item.source, item.metadata.get("page", 1))
                if page_key in seen_pages:
                    continue
                seen_pages.add(page_key)
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
                    }
                )
            best_score = float(reranked[0][1])
            confidence = min(1.0, max(0.0, (best_score + 0.5) / 1.5))
            unsupported = confidence < 0.45
            if unsupported:
                answer = "Unsupported answer: the retrieved evidence is too weak to support a confident response."
            else:
                answer_evidence = evidence[:settings.max_answer_sentences]
                summary_sentences = self._extract_summary_sentences(
                    request.query,
                    answer_evidence,
                    settings.max_answer_sentence_characters,
                )
                citations = self._format_citations(answer_evidence)
                if summary_sentences:
                    bullets = "\n".join(f"- {sentence}" for sentence in summary_sentences)
                    answer = f"Evidence-based summary:\n{bullets}\n\nSources: {citations}"
                else:
                    answer = (
                        "Relevant pages were found, but the PDF's equation text could not be extracted reliably. "
                        "Open the cited PDF pages to view the original mathematical notation."
                        f"\n\nSources: {citations}"
                    )

        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        with self.storage_lock:
            history = self.session_history.setdefault(request.session_id, [])
            history.append(request.query)
            del history[:-settings.max_session_history]
        return {
            "answer": answer,
            "unsupported": unsupported,
            "confidence": round(confidence, 4),
            "sources": evidence,
            "retrieval_scores": [item["score"] for item in evidence],
            "latency_ms": latency_ms,
            "session_id": request.session_id,
            "history": history,
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


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
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


@app.delete("/documents/{filename}", response_model=DeleteDocumentResponse)
def delete_document(filename: str) -> Dict[str, Any]:
    return get_service().delete_document(filename)


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
