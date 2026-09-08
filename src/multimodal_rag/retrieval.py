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
        self.source_indices: Dict[str, List[int]] = {}

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

        start_index = len(self.chunks)
        self.chunks.extend(chunks)
        for offset, chunk in enumerate(chunks):
            self.chunk_lookup[chunk.chunk_id] = chunk
            self.source_indices.setdefault(chunk.source, []).append(start_index + offset)
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

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        sources: Optional[set[str]] = None,
    ) -> List[RetrievalResult]:
        """Return the highest-scoring chunks, optionally restricted to sources.

        ``IndexFlatIP`` does not provide payload filtering.  Scoped retrieval
        therefore ranks only the indexed vectors belonging to the requested
        documents.  Filtering happens before reranking, which prevents an
        unrelated, large document from consuming the reranker's candidate
        budget.
        """
        if sources is not None and not sources:
            return []

        requested_sources = set(sources) if sources is not None else None
        hits = self.search(query_embedding, top_k=top_k) if requested_sources is None else self._search_sources(
            query_embedding,
            requested_sources,
            top_k,
        )

        results: List[RetrievalResult] = []
        for idx, score in hits[:top_k]:
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
                    type=chunk.type,
                )
            )
        return results

    def _search_sources(
        self,
        query_embedding: np.ndarray,
        sources: set[str],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """Search source-local vector rows without duplicating the FAISS index."""
        if self.index.ntotal == 0:
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected a query embedding with {self.embedding_dim} dimensions.")

        source_rows = [self.source_indices.get(source, []) for source in sources]
        indices = np.asarray([index for rows in source_rows for index in rows], dtype=np.int64)
        if len(indices) == 0:
            return []

        # ``get_xb`` exposes the IndexFlatIP storage as a NumPy view, so this
        # computes only the selected-document scores and does not retain a
        # second full embedding matrix in memory.
        flat_index = faiss.downcast_index(self.index)
        if not hasattr(flat_index, "get_xb"):
            # All supported runtime indexes are IndexFlatIP. Keep a safe
            # fallback for a future compatible FAISS replacement.
            vectors = np.vstack([self.index.reconstruct(int(index)) for index in indices])
        else:
            vectors = faiss.rev_swig_ptr(
                flat_index.get_xb(),
                self.index.ntotal * self.embedding_dim,
            ).reshape(self.index.ntotal, self.embedding_dim)[indices]

        scores = vectors @ query
        result_count = min(top_k, len(indices))
        top_positions = np.argpartition(-scores, result_count - 1)[:result_count]
        ordered_positions = top_positions[np.argsort(-scores[top_positions])]
        return [(int(indices[position]), float(scores[position])) for position in ordered_positions]

    def retrieve_diversified(
        self,
        query_embedding: np.ndarray,
        sources: set[str],
        per_source_k: int,
    ) -> List[RetrievalResult]:
        """Return an interleaved candidate pool with evidence from each source.

        This is reserved for explicit comparison questions.  It deliberately
        retrieves each requested document independently so a high-volume
        document cannot eliminate the other side of a comparison before the
        cross-encoder evaluates it.
        """
        if per_source_k < 1:
            raise ValueError("per_source_k must be at least 1")
        if not sources:
            return []

        source_results = {
            source: self.retrieve(query_embedding, top_k=per_source_k, sources={source})
            for source in sorted(sources)
        }
        diversified: List[RetrievalResult] = []
        for rank in range(per_source_k):
            for source in sorted(source_results):
                candidates = source_results[source]
                if rank < len(candidates):
                    diversified.append(candidates[rank])
        return diversified

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
        self.source_indices = {}
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
        for index, chunk in enumerate(self.chunks):
            self.source_indices.setdefault(chunk.source, []).append(index)
