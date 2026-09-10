"""Evaluate deterministic and local-Ollama answers against the labeled corpus.

This runner records automatic provenance and abstention checks. It also writes
a category-balanced CSV template for the rubric's human semantic review.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from multimodal_rag.answer_evaluation import (
    citation_coverage,
    extract_citations,
    ground_truth_citations,
    select_review_query_ids,
    summarize_answer_records,
)
from multimodal_rag.api import RAGService
from multimodal_rag.schemas import QueryRequest


REVIEW_COLUMNS = [
    "query_id",
    "mode",
    "category",
    "query",
    "expected_supported",
    "expected_claims",
    "answer",
    "returned_source_pages",
    "correctness",
    "completeness",
    "citation_correctness",
    "faithfulness",
    "clarity",
    "multimodal_accuracy",
    "unsupported_behavior",
    "reviewer_notes",
]


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}.")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def source_pages(response: dict[str, Any]) -> list[tuple[str, int]]:
    return list(
        dict.fromkeys(
            (str(source["source"]), int(source["page"]))
            for source in response.get("sources", [])
        )
    )


def answer_record(
    *,
    question: dict[str, Any],
    ground_truth: dict[str, Any],
    mode: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    answer = str(response["answer"])
    return {
        "query_id": question["query_id"],
        "query": question["query"],
        "category": question.get("category", "other"),
        "document_scope": question.get("document_scope", []),
        "expected_supported": bool(question.get("expected_supported", True)),
        "expected_claims": ground_truth.get("expected_claims", []),
        "mode": mode,
        "answer": answer,
        "answer_intent": response["answer_intent"],
        "unsupported": bool(response["unsupported"]),
        "confidence": float(response["confidence"]),
        "synthesis_mode": response.get("synthesis_mode", "deterministic"),
        "latency_ms": float(response["latency_ms"]),
        "citations": extract_citations(answer),
        "citation_coverage": citation_coverage(answer),
        "returned_source_pages": source_pages(response),
        "ground_truth_source_pages": sorted(ground_truth_citations(ground_truth)),
        "sources": [
            {
                "chunk_id": source["chunk_id"],
                "source": source["source"],
                "page": source["page"],
                "type": source["type"],
            }
            for source in response.get("sources", [])
        ],
    }


def write_review_template(path: Path, records: list[dict[str, Any]], query_ids: list[str]) -> None:
    selected_ids = set(query_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for record in records:
            if record["query_id"] not in selected_ids:
                continue
            writer.writerow(
                {
                    "query_id": record["query_id"],
                    "mode": record["mode"],
                    "category": record["category"],
                    "query": record["query"],
                    "expected_supported": record["expected_supported"],
                    "expected_claims": " | ".join(record["expected_claims"]),
                    "answer": record["answer"],
                    "returned_source_pages": " | ".join(
                        f"{source}, p. {page}" for source, page in record["returned_source_pages"]
                    ),
                    "correctness": "",
                    "completeness": "",
                    "citation_correctness": "",
                    "faithfulness": "",
                    "clarity": "",
                    "multimodal_accuracy": "",
                    "unsupported_behavior": "",
                    "reviewer_notes": "",
                }
            )


def validate_document_scopes(service: RAGService, questions: list[dict[str, Any]]) -> None:
    indexed_sources = {
        document["filename"]
        for document in service.documents
        if document.get("status") == "indexed"
    }
    required_sources = {source for question in questions for source in question.get("document_scope", [])}
    missing_sources = sorted(required_sources.difference(indexed_sources))
    if missing_sources:
        raise FileNotFoundError(
            "The answer evaluation corpus is incomplete. Index these files before running: "
            + ", ".join(missing_sources)
        )


def run_mode(
    service: RAGService,
    *,
    mode: str,
    questions: list[dict[str, Any]],
    ground_truth_by_id: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    service.synthesizer.enabled = mode == "ollama"
    service.synthesizer._unavailable_until = 0.0
    records = []
    for question in questions:
        query_id = question["query_id"]
        response = service.query(
            QueryRequest(
                query=question["query"],
                top_k=top_k,
                document_names=question.get("document_scope", []),
                # A unique session prevents previous benchmark questions from
                # becoming conversational context for the current answer.
                session_id=f"answer-evaluation-{mode}-{query_id}",
            )
        )
        records.append(
            answer_record(
                question=question,
                ground_truth=ground_truth_by_id[query_id],
                mode=mode,
                response=response,
            )
        )
        print(f"[{mode}] {query_id}: {response['synthesis_mode']} ({response['latency_ms']} ms)")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated answer predictions and review template.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--modes", nargs="+", choices=("deterministic", "ollama"), default=("deterministic", "ollama"))
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N labeled questions for a smoke test.")
    parser.add_argument("--review-sample-size", type=int, default=24)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("top-k must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be at least 1 when provided.")

    questions = load_json(args.dataset_dir / "questions.json")
    if args.limit is not None:
        questions = questions[: args.limit]
    ground_truth_by_id = {item["query_id"]: item for item in load_json(args.dataset_dir / "ground_truth.json")}
    missing_ground_truth = [question["query_id"] for question in questions if question["query_id"] not in ground_truth_by_id]
    if missing_ground_truth:
        raise ValueError(f"Ground-truth labels are missing for: {', '.join(missing_ground_truth)}")

    service = RAGService()
    try:
        validate_document_scopes(service, questions)
        all_records: list[dict[str, Any]] = []
        predictions_dir = args.output_dir or args.dataset_dir / "predictions"
        for mode in args.modes:
            mode_records = run_mode(
                service,
                mode=mode,
                questions=questions,
                ground_truth_by_id=ground_truth_by_id,
                top_k=args.top_k,
            )
            write_json(predictions_dir / f"answers_{mode}.json", mode_records)
            all_records.extend(mode_records)

        review_ids = select_review_query_ids(questions, args.review_sample_size)
        review_path = predictions_dir / "answer_review_template.csv"
        write_review_template(review_path, all_records, review_ids)
        metrics = {
            "benchmark": args.dataset_dir.name,
            "questions": len(questions),
            "modes": list(args.modes),
            "automatic_metrics": summarize_answer_records(all_records),
            "human_review": {
                "template": str(review_path),
                "sample_query_ids": review_ids,
                "instructions": "Score the CSV with evaluation/rubric.md. Automated provenance checks do not establish semantic correctness or faithfulness.",
            },
        }
        metrics_path = predictions_dir / "answer_metrics.json"
        write_json(metrics_path, metrics)
        print("\n" + json.dumps(metrics, indent=2))
        print(f"\nWrote answer predictions to {predictions_dir}")
        print(f"Wrote human-review template to {review_path}")
    finally:
        service.job_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
