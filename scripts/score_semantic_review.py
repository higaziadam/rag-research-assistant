"""Turn a completed answer-review CSV into a reproducible release summary.

Use the CSV emitted by ``run_answer_evaluation.py``.  Score it with
``evaluation/rubric.md`` first; this script validates that no required review
fields are silently omitted and writes a machine-readable release artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from multimodal_rag.semantic_evaluation import summarize_semantic_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_csv", type=Path, help="Completed CSV from run_answer_evaluation.py.")
    parser.add_argument("--output", type=Path, required=True, help="Path for the semantic-review JSON summary.")
    args = parser.parse_args()

    with args.review_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("The review CSV is empty.")

    summary = summarize_semantic_reviews(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote semantic review to {args.output}")


if __name__ == "__main__":
    main()
