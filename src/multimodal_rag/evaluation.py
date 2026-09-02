from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def compute_recall_at_k(relevance: Iterable[Iterable[int]], k: int = 5) -> float:
    """Compute Recall@k for binary relevance lists."""
    scores = []
    for rankings in relevance:
        rankings = list(rankings)
        hits = 0
        total = sum(1 for x in rankings if x > 0)
        if total == 0:
            scores.append(1.0)
            continue
        for idx, item in enumerate(rankings[:k], start=1):
            if item > 0:
                hits += 1
        scores.append(hits / total)
    return float(np.mean(scores)) if scores else 0.0


def compute_ndcg_at_k(relevance: Iterable[Iterable[int]], k: int = 5) -> float:
    scores = []
    for rankings in relevance:
        values = np.asarray(list(rankings), dtype=float)
        if values.size == 0:
            scores.append(0.0)
            continue
        observed = values[:k]
        discounts = np.log2(np.arange(2, observed.size + 2))
        dcg = float(np.sum((2**observed - 1) / discounts))
        ideal = np.sort(values)[::-1][:k]
        ideal_discounts = np.log2(np.arange(2, ideal.size + 2))
        ideal_dcg = float(np.sum((2**ideal - 1) / ideal_discounts))
        scores.append(dcg / ideal_dcg if ideal_dcg else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def evaluate_ranking_predictions(predictions_path: str, ground_truth_path: str, k: int = 5) -> Dict[str, float]:
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    relevance_by_query = {entry["query_id"]: set(entry.get("relevance", [])) for entry in ground_truth}
    relevance_list = []
    for item in predictions:
        qid = item["query_id"]
        gt = relevance_by_query.get(qid, set())
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
