from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Reranker:
    """Cross-encoder reranker for improving top-k retrieval quality."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        features = self.tokenizer(
            [(query, passage) for passage in passages],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**features).logits
            scores = logits[:, 1].cpu().tolist()
        return scores

    def rerank(self, query: str, candidates: Sequence[Tuple[str, dict]], top_k: int = 5):
        if not candidates:
            return []
        passages = [candidate[0] for candidate in candidates]
        scores = self.score_pairs(query, passages)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]
