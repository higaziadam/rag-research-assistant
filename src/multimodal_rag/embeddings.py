from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingStore:
    """Lightweight wrapper around sentence-transformers for dense retrieval."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        clean = [str(t).strip() for t in texts if str(t).strip()]
        if not clean:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))
        embeddings = self.model.encode(clean, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
