from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def compute_recall_at_k(relevance: Iterable[Iterable[int]], k: int = 5) -> float:
    """Compute Recall@k for binary relevance lists."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    scores = []
    for rankings in relevance:
        rankings = list(rankings)
        hits = 0
        total = sum(1 for x in rankings if x > 0)
        if total == 0:
            scores.append(1.0)
            continue
        for item in rankings[:k]:
            if item > 0:
                hits += 1
        scores.append(hits / total)
    return float(np.mean(scores)) if scores else 0.0


def compute_ndcg_at_k(relevance: Iterable[Iterable[int]], k: int = 5) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")
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


def compute_mrr(relevance: Iterable[Iterable[int]]) -> float:
    """Compute mean reciprocal rank for ranked binary relevance labels."""
    scores = []
    for rankings in relevance:
        reciprocal_rank = 0.0
        for rank, item in enumerate(rankings, start=1):
            if item > 0:
                reciprocal_rank = 1.0 / rank
                break
        scores.append(reciprocal_rank)
    return float(np.mean(scores)) if scores else 0.0


def _source_page_from_chunk_id(chunk_id: str) -> tuple[str, int] | None:
    """Recover source/page from this project's stable extracted-chunk IDs.

    Older prediction artifacts only contain chunk IDs. New artifacts also
    store explicit source/page metadata, but this parser keeps those earlier
    benchmark runs comparable.
    """
    match = re.match(r"^(.*)-(\d+)-(?:text|table|figure|equation)-", chunk_id)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _prediction_source_pages(prediction: Dict[str, Any]) -> list[tuple[str, int]]:
    explicit = prediction.get("ranked_results", [])
    if isinstance(explicit, list):
        source_pages = [
            (str(item["source"]), int(item["page"]))
            for item in explicit
            if isinstance(item, dict) and item.get("source") and item.get("page") is not None
        ]
        if source_pages:
            return source_pages
    return [
        source_page
        for chunk_id in prediction.get("ranked_chunk_ids", [])
        if isinstance(chunk_id, str) and (source_page := _source_page_from_chunk_id(chunk_id)) is not None
    ]


def _ground_truth_source_pages(entry: Dict[str, Any]) -> set[tuple[str, int]]:
    return {
        source_page
        for chunk_id in entry.get("relevance", [])
        if isinstance(chunk_id, str) and (source_page := _source_page_from_chunk_id(chunk_id)) is not None
    }


def evaluate_ranking_predictions(
    predictions_path: str,
    ground_truth_path: str,
    k: int = 5,
    relevance_level: str = "chunk",
) -> Dict[str, float]:
    """Evaluate ranked results at strict-chunk or source-page granularity.

    Source-page relevance is the primary RAG retrieval measure: several
    chunks can represent the same answer-bearing PDF page, and any one of
    them enables the same cited-page verification experience. Strict chunk
    relevance remains available as a diagnostic for chunking sensitivity.
    """
    if relevance_level not in {"chunk", "source_page"}:
        raise ValueError("relevance_level must be 'chunk' or 'source_page'.")
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    relevance_by_query = {
        entry["query_id"]: (
            set(entry.get("relevance", []))
            if relevance_level == "chunk"
            else _ground_truth_source_pages(entry)
        )
        for entry in ground_truth
    }
    relevance_list = []
    for item in predictions:
        qid = item["query_id"]
        gt = relevance_by_query.get(qid, set())
        ranked_items = item.get("ranked_chunk_ids", []) if relevance_level == "chunk" else _prediction_source_pages(item)
        ranking = [1 if ranked_item in gt else 0 for ranked_item in ranked_items]
        missing_relevant = len(gt.difference(ranked_items))
        ranking.extend([1] * missing_relevant)
        relevance_list.append(ranking)

    return {
        f"recall@{k}": compute_recall_at_k(relevance_list, k=k),
        f"ndcg@{k}": compute_ndcg_at_k(relevance_list, k=k),
        "mrr": compute_mrr(relevance_list),
    }


def save_results(metrics: Dict[str, Any], output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
