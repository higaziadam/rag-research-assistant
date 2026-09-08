# Evaluation dataset

This benchmark covers the persisted evaluation corpus: NIST AI-risk guidance, the Transformer paper, the IPCC AR6 synthesis report, U.S. Census poverty statistics, a USGS remote-sensing report, and OpenStax Calculus Volume 3. It contains 66 questions: definitions, explanations, procedures, summaries, table questions, figure questions, mathematical concepts, multi-document comparisons, and unsupported questions.

The ground truth was labelled from `artifacts/metadata.jsonl` on the indexed corpus currently in this workspace. Its chunk IDs are valid only while the document, extraction logic, and chunking settings remain unchanged. Re-indexing the document can change the IDs; revalidate the labels after an intentional re-ingest.

Run both dense-only and dense-plus-reranker retrieval locally:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

This writes `evaluation/predictions/baseline.json`, `evaluation/predictions/reranked.json`, and `evaluation/predictions/metrics.json`. These are retrieval-only metrics. Use `rubric.md` to score the resulting complete answers separately.
