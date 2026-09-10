from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

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
        self._source_filter_cache: Dict[frozenset[str], np.ndarray] = {}

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
        self._source_filter_cache.clear()
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

        if sources.issuperset(self.source_indices):
            return self.search(query, top_k=top_k)

        source_key = frozenset(sources)
        allowed_indices = self._source_filter_cache.get(source_key)
        if allowed_indices is None:
            allowed_indices = np.asarray(
                sorted(
                    index
                    for source in source_key
                    for index in self.source_indices.get(source, ())
                ),
                dtype=np.int64,
            )
            if len(self._source_filter_cache) >= 128:
                self._source_filter_cache.clear()
            self._source_filter_cache[source_key] = allowed_indices
        if len(allowed_indices) == 0:
            return []

        result_count = min(top_k, len(allowed_indices))
        try:
            parameters = faiss.SearchParameters()
            parameters.sel = faiss.IDSelectorBatch(allowed_indices)
            scores, indices = self.index.search(
                query.reshape(1, -1),
                result_count,
                params=parameters,
            )
            return [
                (int(index), float(score))
                for index, score in zip(indices[0], scores[0])
                if index != -1
            ]
        except (AttributeError, RuntimeError, TypeError):
            # Compatibility fallback for older FAISS builds. It retains only
            # O(N) scores/ids rather than copying O(N * embedding_dim) vectors.
            allowed_set = set(allowed_indices.tolist())
            scores, indices = self.index.search(query.reshape(1, -1), self.index.ntotal)
            hits: List[Tuple[int, float]] = []
            for index, score in zip(indices[0], scores[0]):
                resolved_index = int(index)
                if resolved_index in allowed_set:
                    hits.append((resolved_index, float(score)))
                    if len(hits) == result_count:
                        break
            return hits

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

        removed_indices = np.asarray(
            [index for index, chunk in enumerate(self.chunks) if chunk.source in sources],
            dtype=np.int64,
        )
        filtered.index = faiss.clone_index(self.index)
        if len(removed_indices):
            filtered.index.remove_ids(removed_indices)
        filtered.chunks = [self.chunks[index] for index in retained_indices]
        filtered._rebuild_lookups()
        return filtered

    def clone(self) -> "FAISSRetriever":
        """Clone index state without reconstructing every vector in Python."""
        cloned = FAISSRetriever(self.embedding_dim)
        cloned.index = faiss.clone_index(self.index)
        cloned.chunks = list(self.chunks)
        cloned._rebuild_lookups()
        return cloned

    def _rebuild_lookups(self) -> None:
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.source_indices = {}
        self._source_filter_cache = {}
        for index, chunk in enumerate(self.chunks):
            self.source_indices.setdefault(chunk.source, []).append(index)

    def save(self, index_path: str, metadata_path: str) -> None:
        index_dir = Path(index_path).parent
        index_dir.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        resolved_index = Path(index_path)
        resolved_metadata = Path(metadata_path)
        transaction_id = uuid4().hex
        temporary_index = resolved_index.with_name(f".{resolved_index.name}.{transaction_id}.tmp")
        temporary_metadata = resolved_metadata.with_name(f".{resolved_metadata.name}.{transaction_id}.tmp")
        try:
            faiss.write_index(self.index, str(temporary_index))
            with temporary_metadata.open("w", encoding="utf-8", newline="\n") as file:
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
                    file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())

            # Validate the complete pair before either public path changes.
            validation = FAISSRetriever(self.embedding_dim)
            validation.load(str(temporary_index), str(temporary_metadata))
            os.replace(temporary_metadata, resolved_metadata)
            os.replace(temporary_index, resolved_index)
        finally:
            temporary_index.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)

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
        self._rebuild_lookups()
