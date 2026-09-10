"""Run the local retrieval benchmark with and without cross-encoder reranking."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from multimodal_rag.config import settings
from multimodal_rag.embeddings import EmbeddingStore
from multimodal_rag.evaluation import evaluate_ranking_predictions
from multimodal_rag.api import RAGService
from multimodal_rag.reranker import Reranker
from multimodal_rag.retrieval import FAISSRetriever
from multimodal_rag.sparse_retrieval import BM25Retriever, reciprocal_rank_fusion


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array and fail early with a useful dataset error."""
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}.")
    return payload


def scoped_candidates(retriever: FAISSRetriever, query_embedding: Any, sources: set[str], candidate_k: int):
    """Use the production source filter before candidates reach the reranker."""
    return retriever.retrieve(query_embedding, top_k=candidate_k, sources=sources or None)


def comparison_candidates(retriever: FAISSRetriever, query_embedding: Any, sources: set[str], candidate_k: int):
    """Mirror API candidate allocation for cross-document comparisons."""
    if len(sources) < 2:
        return scoped_candidates(retriever, query_embedding, sources, candidate_k)
    return retriever.retrieve_diversified(
        query_embedding,
        sources=sources,
        per_source_k=math.ceil(candidate_k / len(sources)),
    )


def sparse_scoped_candidates(retriever: BM25Retriever, query: str, sources: set[str], candidate_k: int):
    """Apply the same document scope to lexical BM25 retrieval."""
    return retriever.retrieve(query, top_k=candidate_k, sources=sources or None)


def sparse_comparison_candidates(retriever: BM25Retriever, query: str, sources: set[str], candidate_k: int):
    """Allocate lexical candidates to every document in a comparison."""
    if len(sources) < 2:
        return sparse_scoped_candidates(retriever, query, sources, candidate_k)
    return retriever.retrieve_diversified(
        query,
        sources=sources,
        per_source_k=math.ceil(candidate_k / len(sources)),
    )


def diversify_comparison_reranking(reranked: list[tuple[tuple[str, Any], float]], sources: set[str], limit: int):
    """Guarantee at least one top-ranked result from each comparison source."""
    by_source: dict[str, list[tuple[tuple[str, Any], float]]] = {}
    for result in reranked:
        source = result[0][1].source
        if source in sources:
            by_source.setdefault(source, []).append(result)

    selected = sorted((items[0] for items in by_source.values() if items), key=lambda item: item[1], reverse=True)
    selected_ids = {result[0][1].chunk_id for result in selected}
    for result in reranked:
        if result[0][1].chunk_id not in selected_ids:
            selected.append(result)
            selected_ids.add(result[0][1].chunk_id)
        if len(selected) >= limit:
            break
    return selected[:limit]


def reranked_ids(
    reranker: Reranker,
    query: str,
    candidates: list[Any],
    top_k: int,
    is_comparison: bool,
    source_scope: set[str],
) -> list[str]:
    """Rerank one candidate configuration using the API's comparison policy."""
    rerank_input = [(reranker_text(candidate), candidate) for candidate in candidates]
    reranked = reranker.rerank(query, rerank_input, top_k=len(rerank_input) if is_comparison else top_k)
    if is_comparison:
        reranked = diversify_comparison_reranking(reranked, source_scope, top_k)
    return [candidate.chunk_id for (_, candidate), _score in reranked]


def hybrid_rerank_candidates(
    fused_candidates: list[Any],
    dense_candidates: list[Any],
    fused_limit: int,
    dense_backfill_limit: int,
) -> list[Any]:
    """Retain dense-retrieval recall while adding BM25-discovered evidence."""
    selected = []
    selected_ids = set()
    for candidates, limit in ((fused_candidates, fused_limit), (dense_candidates, dense_backfill_limit)):
        for candidate in candidates[:limit]:
            if candidate.chunk_id not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate.chunk_id)
    return selected


def prediction_record(query_id: str, ranked_candidates: list[Any]) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "ranked_chunk_ids": [candidate.chunk_id for candidate in ranked_candidates],
        "ranked_results": [
            {
                "chunk_id": candidate.chunk_id,
                "source": candidate.source,
                "page": int(candidate.metadata.get("page", 1)),
            }
            for candidate in ranked_candidates
        ],
    }


def reranker_text(candidate: Any) -> str:
    """Build the same multimodal search text for a retrieved result."""
    type_label = f"{candidate.type.title()} evidence" if candidate.type != "text" else ""
    parts = [type_label, candidate.text.strip(), candidate.table.strip(), candidate.figure_caption.strip()]
    return "\n".join(part for part in parts if part)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=settings.retrieval_candidate_k)
    parser.add_argument("--sparse-candidate-k", type=int, default=settings.sparse_candidate_k)
    parser.add_argument("--rerank-candidate-k", type=int, default=settings.rerank_candidate_k)
    args = parser.parse_args()

    if (
        args.top_k < 1
        or args.candidate_k < args.top_k
        or args.sparse_candidate_k < args.top_k
        or args.rerank_candidate_k < args.top_k
    ):
        raise ValueError("Candidate counts must be at least top-k, and all values must be positive.")

    questions = load_json(args.dataset_dir / "questions.json")
    ground_truth = load_json(args.dataset_dir / "ground_truth.json")
    ground_truth_ids = {item["query_id"] for item in ground_truth}
    missing_labels = [question["query_id"] for question in questions if question["query_id"] not in ground_truth_ids]
    if missing_labels:
        raise ValueError(f"Ground-truth labels are missing for: {', '.join(missing_labels)}")
    if not settings.faiss_index_path.exists() or not settings.metadata_path.exists():
        raise FileNotFoundError("No persisted FAISS index was found. Upload and finish indexing the evaluation documents first.")

    embeddings = EmbeddingStore(settings.model_name, local_files_only=settings.model_local_files_only)
    retriever = FAISSRetriever(
        embedding_dim=embeddings.embedding_dimension,
        index_path=str(settings.faiss_index_path),
        metadata_path=str(settings.metadata_path),
    )
    reranker = Reranker(
        settings.reranker_model,
        local_files_only=settings.model_local_files_only,
        batch_size=settings.reranker_batch_size,
    )
    sparse_retriever = BM25Retriever(retriever.chunks)

    baseline_predictions = []
    reranked_predictions = []
    hybrid_predictions = []
    hybrid_reranked_predictions = []
    for question in questions:
        # Unsupported questions are exercised by answer-level review, not retrieval metrics.
        if not question.get("expected_supported", True):
            continue

        query = str(question["query"])
        source_scope = set(question.get("document_scope", []))
        intent = RAGService._question_intent(query)
        is_comparison = intent == "comparison" and len(source_scope) > 1
        dense_selector = comparison_candidates if is_comparison else scoped_candidates
        sparse_selector = sparse_comparison_candidates if is_comparison else sparse_scoped_candidates
        query_embedding = embeddings.encode_single(query)
        dense_candidates = dense_selector(
            retriever,
            query_embedding,
            source_scope,
            args.candidate_k,
        )
        sparse_candidates = sparse_selector(
            sparse_retriever,
            RAGService._lexical_retrieval_query(query, intent),
            source_scope,
            args.sparse_candidate_k,
        )
        hybrid_candidates = reciprocal_rank_fusion(
            [dense_candidates, sparse_candidates],
            top_k=args.candidate_k,
            rank_constant=settings.reciprocal_rank_fusion_constant,
        )
        baseline_predictions.append(
            prediction_record(question["query_id"], dense_candidates[: args.top_k])
        )
        hybrid_predictions.append(
            prediction_record(question["query_id"], hybrid_candidates[: args.top_k])
        )
        reranked_predictions.append(
            prediction_record(
                question["query_id"],
                [
                    next(candidate for candidate in dense_candidates if candidate.chunk_id == chunk_id)
                    for chunk_id in reranked_ids(
                    reranker,
                    query,
                    dense_candidates[: args.rerank_candidate_k],
                    args.top_k,
                    is_comparison,
                    source_scope,
                    )
                ],
            )
        )
        hybrid_reranked_predictions.append(
            prediction_record(
                question["query_id"],
                [
                    next(candidate for candidate in hybrid_rerank_candidates(
                        hybrid_candidates,
                        dense_candidates,
                        args.rerank_candidate_k,
                        settings.hybrid_dense_backfill_k,
                    ) if candidate.chunk_id == chunk_id)
                    for chunk_id in reranked_ids(
                    reranker,
                    query,
                    hybrid_rerank_candidates(
                        hybrid_candidates,
                        dense_candidates,
                        args.rerank_candidate_k,
                        settings.hybrid_dense_backfill_k,
                    ),
                    args.top_k,
                    is_comparison,
                    source_scope,
                    )
                ],
            )
        )

    predictions_dir = args.dataset_dir / "predictions"
    baseline_path = predictions_dir / "baseline.json"
    reranked_path = predictions_dir / "reranked.json"
    hybrid_path = predictions_dir / "hybrid.json"
    hybrid_reranked_path = predictions_dir / "hybrid_reranked.json"
    write_json(baseline_path, baseline_predictions)
    write_json(reranked_path, reranked_predictions)
    write_json(hybrid_path, hybrid_predictions)
    write_json(hybrid_reranked_path, hybrid_reranked_predictions)

    ground_truth_path = args.dataset_dir / "ground_truth.json"
    metrics = {
        "benchmark": args.dataset_dir.name,
        "evaluated_supported_queries": len(baseline_predictions),
        "excluded_unsupported_queries": len(questions) - len(baseline_predictions),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "sparse_candidate_k": args.sparse_candidate_k,
        "rerank_candidate_k": args.rerank_candidate_k,
        "primary_relevance_level": "source_page",
        "baseline": {
            str(k): evaluate_ranking_predictions(str(baseline_path), str(ground_truth_path), k=k, relevance_level="source_page")
            for k in (1, 3, args.top_k)
        },
        "reranked": {
            str(k): evaluate_ranking_predictions(str(reranked_path), str(ground_truth_path), k=k, relevance_level="source_page")
            for k in (1, 3, args.top_k)
        },
        "hybrid": {
            str(k): evaluate_ranking_predictions(str(hybrid_path), str(ground_truth_path), k=k, relevance_level="source_page")
            for k in (1, 3, args.top_k)
        },
        "hybrid_reranked": {
            str(k): evaluate_ranking_predictions(str(hybrid_reranked_path), str(ground_truth_path), k=k, relevance_level="source_page")
            for k in (1, 3, args.top_k)
        },
        "strict_chunk_diagnostics": {
            "baseline": {str(k): evaluate_ranking_predictions(str(baseline_path), str(ground_truth_path), k=k) for k in (1, 3, args.top_k)},
            "reranked": {str(k): evaluate_ranking_predictions(str(reranked_path), str(ground_truth_path), k=k) for k in (1, 3, args.top_k)},
            "hybrid": {str(k): evaluate_ranking_predictions(str(hybrid_path), str(ground_truth_path), k=k) for k in (1, 3, args.top_k)},
            "hybrid_reranked": {
                str(k): evaluate_ranking_predictions(str(hybrid_reranked_path), str(ground_truth_path), k=k)
                for k in (1, 3, args.top_k)
            },
        },
    }
    metrics_path = predictions_dir / "metrics.json"
    write_json(metrics_path, metrics)

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote predictions to {predictions_dir}")


if __name__ == "__main__":
    main()
