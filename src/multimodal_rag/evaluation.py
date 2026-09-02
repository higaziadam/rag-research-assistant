from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.metrics import ndcg_score


def compute_recall_at_k(relevance: Iterable[Iterable[int]], k: int = 5) -> float:
    """Compute Recall@k for binary relevance lists."""
    scores = []
    for rankings in relevance:
        hits = 0
        total = sum(1 for x in rankings if x > 0)
        if total == 0:
            scores.append(1.0)
            continue
        for idx, item in enumerate(rankings[:k], start=1):
            if item > 0:
                hits += 1
        scores.append(hits / total)
    return float(np.mean(scores))


def compute_ndcg_at_k(relevance: Iterable[Iterable[int]], k: int = 5) -> float:
    scores = []
    for rankings in relevance:
        true_relevance = np.asarray(rankings, dtype=float)
        if true_relevance.size == 0:
            scores.append(0.0)
            continue
        pred = np.asarray([true_relevance], dtype=float)
        scores.append(ndcg_score(pred, true_relevance[:k]))
    return float(np.mean(scores))


def evaluate_ranking_predictions(predictions_path: str, ground_truth_path: str, k: int = 5) -> Dict[str, float]:
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    relevance_list = []
    for item in predictions:
        qid = item["query_id"]
        gt = next((entry["relevance"] for entry in ground_truth if entry["query_id"] == qid), [])
        ranking = [1 if chunk_id in gt else 0 for chunk_id in item["ranked_chunk_ids"]]
        relevance_list.append(ranking)

    return {
        "recall@5": compute_recall_at_k(relevance_list, k=k),
        "ndcg@5": compute_ndcg_at_k(relevance_list, k=k),
    }


def save_results(metrics: Dict[str, Any], output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
