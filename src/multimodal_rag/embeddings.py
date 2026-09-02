from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingStore:
    """Lightweight wrapper around sentence-transformers for dense retrieval."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @property
    def embedding_dimension(self) -> int:
        get_dimension = getattr(self.model, "get_embedding_dimension", None)
        if get_dimension is not None:
            return int(get_dimension())
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        clean = [str(text).strip() for text in texts]
        if not clean:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)
        if any(not text for text in clean):
            raise ValueError("Cannot embed empty document chunks.")
        embeddings = self.model.encode(clean, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
