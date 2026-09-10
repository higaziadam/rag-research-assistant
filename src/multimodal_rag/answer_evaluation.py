"""Metrics and review sampling for answer-level RAG evaluation.

These helpers intentionally separate mechanically verifiable checks from
human judgment. Citation-reference validity is not the same thing as semantic
entailment, so faithfulness and completeness remain rubric-scored fields.
"""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import median
from typing import Any, Iterable


Citation = tuple[str, int]

_CITATION_PATTERN = re.compile(r"\[([^\]\n,]+),\s*p\.\s*(\d+(?:\s*,\s*\d+)*)\]", re.IGNORECASE)
_GROUND_TRUTH_CHUNK_PATTERN = re.compile(r"^(.*\.pdf)-(\d+)-")
_BULLET_PATTERN = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")
_SYSTEM_TEXT_PREFIXES = (
    "the original notation is rendered below",
    "relevant table or figure evidence was found",
)


def extract_citations(answer: str) -> list[Citation]:
    """Extract document/page citations in the public answer format."""
    citations: list[Citation] = []
    for source, page_values in _CITATION_PATTERN.findall(answer):
        for page in page_values.split(","):
            citations.append((source.strip(), int(page.strip())))
    return list(dict.fromkeys(citations))


def factual_blocks(answer: str) -> list[str]:
    """Return Markdown prose blocks that should carry a source citation."""
    without_sources = re.split(r"^\*\*Sources?\*\*\s*$", answer, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)[0]
    blocks: list[str] = []
    pending: list[str] = []
    for raw_line in without_sources.splitlines():
        line = raw_line.strip()
        if not line:
            if pending:
                blocks.append(" ".join(pending))
                pending = []
            continue
        if line.startswith("#") or re.fullmatch(r"\*\*[^*]+\*\*", line):
            if pending:
                blocks.append(" ".join(pending))
                pending = []
            continue
        if _BULLET_PATTERN.match(line):
            if pending:
                blocks.append(" ".join(pending))
                pending = []
            blocks.append(line)
            continue
        pending.append(line)
    if pending:
        blocks.append(" ".join(pending))

    return [
        block
        for block in blocks
        if not block.casefold().startswith(_SYSTEM_TEXT_PREFIXES)
        and not block.casefold().startswith("unsupported answer:")
    ]


def citation_coverage(answer: str) -> dict[str, int]:
    """Count factual Markdown blocks and the subset carrying a page citation."""
    blocks = factual_blocks(answer)
    cited_blocks = sum(bool(_CITATION_PATTERN.search(block)) for block in blocks)
    return {"factual_blocks": len(blocks), "cited_blocks": cited_blocks}


def ground_truth_citations(ground_truth: dict[str, Any]) -> set[Citation]:
    """Derive labeled source/page pairs from persisted relevance chunk IDs."""
    pairs: set[Citation] = set()
    for chunk_id in ground_truth.get("relevance", []):
        match = _GROUND_TRUTH_CHUNK_PATTERN.match(str(chunk_id))
        if match:
            pairs.add((match.group(1), int(match.group(2))))
    return pairs


def _normalized_citation_pairs(record: dict[str, Any], field: str) -> set[Citation]:
    """Read tuple pairs both before and after JSON serialization.

    Evaluation runs keep pairs as tuples in memory, while persisted JSON
    represents them as two-item lists. Normalizing at the reporting boundary
    makes stored answer artifacts reproducible and independently reviewable.
    """
    pairs: set[Citation] = set()
    for value in record.get(field, []):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            source, page = value
            try:
                pairs.add((str(source), int(page)))
            except (TypeError, ValueError):
                continue
    return pairs


def percentile(values: Iterable[float], percent: float) -> float | None:
    """Compute a deterministic nearest-rank percentile without extra dependencies."""
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percent))))
    return round(float(ordered[index]), 3)


def summarize_answer_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize automatic, non-semantic checks by answer-generation mode."""
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_mode[str(record["mode"])].append(record)

    summaries: dict[str, dict[str, Any]] = {}
    for mode, mode_records in by_mode.items():
        expected_supported = [record for record in mode_records if record["expected_supported"]]
        expected_unsupported = [record for record in mode_records if not record["expected_supported"]]
        support_correct = sum(record["unsupported"] is (not record["expected_supported"]) for record in mode_records)
        unsupported_correct = sum(record["unsupported"] for record in expected_unsupported)

        factual_block_count = sum(record["citation_coverage"]["factual_blocks"] for record in expected_supported)
        cited_block_count = sum(record["citation_coverage"]["cited_blocks"] for record in expected_supported)
        cited_pairs = [pair for record in expected_supported for pair in _normalized_citation_pairs(record, "citations")]
        valid_pair_count = sum(
            pair in _normalized_citation_pairs(record, "returned_source_pages")
            for record in expected_supported
            for pair in _normalized_citation_pairs(record, "citations")
        )
        labeled_pair_count = sum(
            pair in _normalized_citation_pairs(record, "ground_truth_source_pages")
            for record in expected_supported
            for pair in _normalized_citation_pairs(record, "citations")
        )
        supported_with_labeled_citation = sum(
            bool(
                _normalized_citation_pairs(record, "citations").intersection(
                    _normalized_citation_pairs(record, "ground_truth_source_pages")
                )
            )
            for record in expected_supported
        )
        latencies = [float(record["latency_ms"]) for record in mode_records]
        accepted_synthesis = sum(record["synthesis_mode"] == "ollama" for record in expected_supported)

        summaries[mode] = {
            "queries": len(mode_records),
            "expected_supported_queries": len(expected_supported),
            "expected_unsupported_queries": len(expected_unsupported),
            "support_classification_accuracy": round(support_correct / len(mode_records), 4) if mode_records else None,
            "unsupported_answer_accuracy": round(unsupported_correct / len(expected_unsupported), 4)
            if expected_unsupported
            else None,
            "citation_block_coverage": round(cited_block_count / factual_block_count, 4) if factual_block_count else None,
            "citation_reference_validity": round(valid_pair_count / len(cited_pairs), 4) if cited_pairs else None,
            "labeled_citation_pair_overlap": round(labeled_pair_count / len(cited_pairs), 4) if cited_pairs else None,
            "answers_with_labeled_citation": round(supported_with_labeled_citation / len(expected_supported), 4)
            if expected_supported
            else None,
            "synthesis_acceptance_rate": round(accepted_synthesis / len(expected_supported), 4)
            if expected_supported
            else None,
            "fallback_rate": round(1 - (accepted_synthesis / len(expected_supported)), 4)
            if mode == "ollama" and expected_supported
            else None,
            "latency_ms": {
                "median": round(float(median(latencies)), 3) if latencies else None,
                "p95": percentile(latencies, 0.95),
            },
            "metric_notes": {
                "citation_reference_validity": "Checks that every answer citation names a source/page returned by retrieval; it does not prove semantic entailment.",
                "labeled_citation_pair_overlap": "Lower-bound overlap with the benchmark's labeled relevance pages; citations to other valid supporting pages are not counted as matches.",
            },
        }
    return summaries


def select_review_query_ids(questions: list[dict[str, Any]], sample_size: int) -> list[str]:
    """Choose a deterministic, category-balanced question sample for human review."""
    if sample_size < 1:
        return []
    unsupported = sorted(
        question["query_id"] for question in questions if not question.get("expected_supported", True)
    )
    selected = unsupported[:sample_size]
    remaining = sample_size - len(selected)
    if remaining <= 0:
        return selected

    by_category: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        if question.get("expected_supported", True):
            by_category[str(question.get("category", "other"))].append(question["query_id"])
    for values in by_category.values():
        values.sort()

    while remaining and any(by_category.values()):
        for category in sorted(by_category):
            if not remaining:
                break
            if by_category[category]:
                selected.append(by_category[category].pop(0))
                remaining -= 1
    return selected
