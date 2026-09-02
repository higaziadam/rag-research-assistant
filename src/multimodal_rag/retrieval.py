from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from .data_models import DocumentChunk, RetrievalResult


class FAISSRetriever:
    """Dense retriever over document chunks."""

    def __init__(self, embedding_dim: int, index_path: Optional[str] = None):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[DocumentChunk] = []
        self.chunk_lookup: Dict[str, DocumentChunk] = {}

        if index_path:
            self.load(index_path)

    def add_chunks(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have matching lengths")

        self.chunks.extend(chunks)
        for chunk in chunks:
            self.chunk_lookup[chunk.chunk_id] = chunk
        self.index.add(embeddings.astype(np.float32))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1), top_k)
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
                )
            )
        return results

    def save(self, index_path: str, metadata_path: str):
        index_dir = Path(index_path).parent
        index_dir.mkdir(parents=True, exist_ok=True)

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
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def load(self, index_path: str):
        self.index = faiss.read_index(index_path)

        metadata_path = str(Path(index_path).with_suffix(".jsonl"))
        if Path(metadata_path).exists():
            self.chunks = []
            with open(metadata_path, "r", encoding="utf-8") as f:
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
                            type=payload.get("type", "text"),
                        )
                    )
            self.chunk_lookup = {chunk.chunk_id: chunk for chunk in self.chunks}
