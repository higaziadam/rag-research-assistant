"""Validate and summarize rubric-scored semantic answer reviews.

Semantic correctness, citation entailment, and clarity cannot be inferred from
retrieval ranks or citation provenance alone.  This module deliberately treats
them as reviewer-entered judgments and makes the release decision reproducible
once that review is complete.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


PRIMARY_CRITERIA = (
    "correctness",
    "completeness",
    "citation_correctness",
    "faithfulness",
    "clarity",
)
OPTIONAL_CRITERIA = ("multimodal_accuracy",)
UNSUPPORTED_CRITERION = "unsupported_behavior"


def _as_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _score(value: Any, *, field: str, query_id: str, required: bool) -> int | None:
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError(f"{query_id}: missing required {field} score.")
        return None
    try:
        score = int(str(value).strip())
    except ValueError as error:
        raise ValueError(f"{query_id}: {field} must be 0, 1, or 2.") from error
    if score not in {0, 1, 2}:
        raise ValueError(f"{query_id}: {field} must be 0, 1, or 2.")
    return score


def summarize_semantic_reviews(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return mode-level semantic metrics and a strict release decision.

    Supported rows require all five primary rubric scores.  Unsupported rows
    require only the abstention score.  Multimodal accuracy is reported when
    entered, but does not distort the ten-point general-answer score.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        query_id = str(row.get("query_id", "<unknown>"))
        mode = str(row.get("mode", "deterministic"))
        supported = _as_boolean(row.get("expected_supported", True))
        reviewed = {"query_id": query_id, "supported": supported}
        if supported:
            reviewed["primary"] = {
                field: _score(row.get(field), field=field, query_id=query_id, required=True)
                for field in PRIMARY_CRITERIA
            }
            reviewed["multimodal"] = _score(
                row.get("multimodal_accuracy"),
                field="multimodal_accuracy",
                query_id=query_id,
                required=False,
            )
        else:
            reviewed["unsupported"] = _score(
                row.get(UNSUPPORTED_CRITERION),
                field=UNSUPPORTED_CRITERION,
                query_id=query_id,
                required=True,
            )
        grouped[mode].append(reviewed)

    summaries: dict[str, Any] = {}
    for mode, reviewed_rows in grouped.items():
        supported_rows = [row for row in reviewed_rows if row["supported"]]
        unsupported_rows = [row for row in reviewed_rows if not row["supported"]]
        if not supported_rows:
            raise ValueError(f"{mode}: no supported answers were reviewed.")
        primary_totals = {
            field: sum(int(row["primary"][field]) for row in supported_rows)
            for field in PRIMARY_CRITERIA
        }
        multimodal_scores = [row["multimodal"] for row in supported_rows if row["multimodal"] is not None]
        mean_score = sum(sum(row["primary"].values()) for row in supported_rows) / len(supported_rows)
        criterion_rates = {
            field: primary_totals[field] / (2 * len(supported_rows))
            for field in PRIMARY_CRITERIA
        }
        unsupported_accuracy = (
            sum(row["unsupported"] == 2 for row in unsupported_rows) / len(unsupported_rows)
            if unsupported_rows
            else None
        )
        release_passed = (
            mean_score >= 8.0
            and criterion_rates["citation_correctness"] >= 0.9
            and criterion_rates["faithfulness"] >= 0.9
            and unsupported_accuracy == 1.0
        )
        summaries[mode] = {
            "supported_answers_reviewed": len(supported_rows),
            "unsupported_answers_reviewed": len(unsupported_rows),
            "mean_primary_score_out_of_10": round(mean_score, 3),
            "criterion_rates": {field: round(rate, 4) for field, rate in criterion_rates.items()},
            "multimodal_accuracy": {
                "reviewed_answers": len(multimodal_scores),
                "mean_score_out_of_2": round(sum(multimodal_scores) / len(multimodal_scores), 3)
                if multimodal_scores
                else None,
            },
            "unsupported_answer_accuracy": round(unsupported_accuracy, 4) if unsupported_accuracy is not None else None,
            "release_gate_passed": release_passed,
        }

    return {
        "rubric": {
            "primary_criteria": list(PRIMARY_CRITERIA),
            "pass_criteria": {
                "mean_primary_score_out_of_10": 8.0,
                "citation_correctness": 0.9,
                "faithfulness": 0.9,
                "unsupported_answer_accuracy": 1.0,
            },
        },
        "modes": summaries,
    }
