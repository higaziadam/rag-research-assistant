# Evaluation dataset

This initial benchmark is intentionally limited to the currently persisted corpus, `NIST.AI.600-1.pdf`. It contains 30 questions: definitions, explanations, table questions, document summaries, and an unsupported question.

The ground truth was labelled from `artifacts/metadata.jsonl` on the indexed corpus currently in this workspace. Its chunk IDs are valid only while the document, extraction logic, and chunking settings remain unchanged. Re-indexing the document can change the IDs; revalidate the labels after an intentional re-ingest.

Run both dense-only and dense-plus-reranker retrieval locally:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

This writes `evaluation/predictions/baseline.json`, `evaluation/predictions/reranked.json`, and `evaluation/predictions/metrics.json`. These are retrieval-only metrics. Use `rubric.md` to score the resulting complete answers separately.
