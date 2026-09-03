from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from .data_models import DocumentChunk, RetrievalResult


class FAISSRetriever:
    """Dense retriever over document chunks."""

    def __init__(self, embedding_dim: int, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[DocumentChunk] = []
        self.chunk_lookup: Dict[str, DocumentChunk] = {}

        if index_path:
            self.load(index_path, metadata_path)

    def add_chunks(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have matching lengths")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embeddings with shape (n, {self.embedding_dim}).")
        if not chunks:
            return

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids) or any(chunk_id in self.chunk_lookup for chunk_id in chunk_ids):
            raise ValueError("Chunk IDs must be unique within and across indexed documents.")

        self.chunks.extend(chunks)
        for chunk in chunks:
            self.chunk_lookup[chunk.chunk_id] = chunk
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected a query embedding with {self.embedding_dim} dimensions.")
        scores, indices = self.index.search(query, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        hits = self.search(query_embedding, top_k=top_k)
        results: List[RetrievalResult] = []
        for idx, score in hits:
            chunk = self.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    score=score,
                    text=chunk.text,
                    table=chunk.table,
                    figure_caption=chunk.figure_caption,
                    source=chunk.source,
                    section=chunk.section,
                    metadata=chunk.metadata,
                    equations=chunk.equations,
                )
            )
        return results

    def without_sources(self, sources: set[str]) -> "FAISSRetriever":
        """Return a new index that excludes every chunk from the given sources."""
        retained_indices = [index for index, chunk in enumerate(self.chunks) if chunk.source not in sources]
        filtered = FAISSRetriever(embedding_dim=self.embedding_dim)
        if not retained_indices:
            return filtered

        retained_chunks = [self.chunks[index] for index in retained_indices]
        retained_embeddings = np.vstack([self.index.reconstruct(index) for index in retained_indices]).astype(np.float32)
        filtered.add_chunks(retained_chunks, retained_embeddings)
        return filtered

    def save(self, index_path: str, metadata_path: str) -> None:
        index_dir = Path(index_path).parent
        index_dir.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "section": chunk.section,
                    "type": chunk.type,
                    "text": chunk.text,
                    "table": chunk.table,
                    "figure_caption": chunk.figure_caption,
                    "metadata": chunk.metadata,
                    "equations": chunk.equations,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def load(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        self.index = faiss.read_index(index_path)
        self.embedding_dim = self.index.d

        resolved_metadata_path = metadata_path or str(Path(index_path).with_suffix(".jsonl"))
        if not Path(resolved_metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {resolved_metadata_path}")

        self.chunks = []
        with open(resolved_metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                self.chunks.append(
                    DocumentChunk(
                        chunk_id=payload["chunk_id"],
                        text=payload.get("text", ""),
                        table=payload.get("table", ""),
                        figure_caption=payload.get("figure_caption", ""),
                        source=payload.get("source", ""),
                        section=payload.get("section", ""),
                        metadata=payload.get("metadata", {}),
                        equations=payload.get("equations", []),
                        type=payload.get("type", "text"),
                    )
                )
        if len(self.chunks) != self.index.ntotal:
            raise ValueError("FAISS index and metadata contain different numbers of chunks.")
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in self.chunks}
