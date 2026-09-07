"""Run the local retrieval benchmark with and without cross-encoder reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from multimodal_rag.config import settings
from multimodal_rag.embeddings import EmbeddingStore
from multimodal_rag.evaluation import evaluate_ranking_predictions
from multimodal_rag.reranker import Reranker
from multimodal_rag.retrieval import FAISSRetriever


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array and fail early with a useful dataset error."""
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}.")
    return payload


def scoped_candidates(retriever: FAISSRetriever, query_embedding: Any, sources: set[str], candidate_k: int):
    """Retrieve dense candidates and retain only documents in the question scope."""
    candidates = retriever.retrieve(query_embedding, top_k=candidate_k)
    return [candidate for candidate in candidates if not sources or candidate.source in sources]


def prediction_record(query_id: str, ranked_chunk_ids: list[str]) -> dict[str, Any]:
    return {"query_id": query_id, "ranked_chunk_ids": ranked_chunk_ids}


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
    parser.add_argument("--rerank-candidate-k", type=int, default=settings.rerank_candidate_k)
    args = parser.parse_args()

    if args.top_k < 1 or args.candidate_k < args.top_k or args.rerank_candidate_k < args.top_k:
        raise ValueError("candidate-k and rerank-candidate-k must be at least top-k, and all values must be positive.")

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

    baseline_predictions = []
    reranked_predictions = []
    for question in questions:
        # Unsupported questions are exercised by answer-level review, not retrieval metrics.
        if not question.get("expected_supported", True):
            continue

        query = str(question["query"])
        source_scope = set(question.get("document_scope", []))
        dense_candidates = scoped_candidates(
            retriever,
            embeddings.encode_single(query),
            source_scope,
            args.candidate_k,
        )
        baseline_predictions.append(
            prediction_record(question["query_id"], [candidate.chunk_id for candidate in dense_candidates[: args.top_k]])
        )

        rerank_candidates = dense_candidates[: args.rerank_candidate_k]
        rerank_input = [(reranker_text(candidate), candidate) for candidate in rerank_candidates]
        reranked = reranker.rerank(query, rerank_input, top_k=args.top_k)
        reranked_predictions.append(
            prediction_record(question["query_id"], [candidate.chunk_id for (_, candidate), _score in reranked])
        )

    predictions_dir = args.dataset_dir / "predictions"
    baseline_path = predictions_dir / "baseline.json"
    reranked_path = predictions_dir / "reranked.json"
    write_json(baseline_path, baseline_predictions)
    write_json(reranked_path, reranked_predictions)

    ground_truth_path = args.dataset_dir / "ground_truth.json"
    metrics = {
        "benchmark": args.dataset_dir.name,
        "evaluated_supported_queries": len(baseline_predictions),
        "excluded_unsupported_queries": len(questions) - len(baseline_predictions),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "rerank_candidate_k": args.rerank_candidate_k,
        "baseline": {str(k): evaluate_ranking_predictions(str(baseline_path), str(ground_truth_path), k=k) for k in (1, 3, args.top_k)},
        "reranked": {str(k): evaluate_ranking_predictions(str(reranked_path), str(ground_truth_path), k=k) for k in (1, 3, args.top_k)},
    }
    metrics_path = predictions_dir / "metrics.json"
    write_json(metrics_path, metrics)

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote predictions to {predictions_dir}")


if __name__ == "__main__":
    main()
