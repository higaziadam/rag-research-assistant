# Evaluation dataset

This benchmark covers the persisted evaluation corpus: NIST AI-risk guidance, the Transformer paper, the IPCC AR6 synthesis report, U.S. Census poverty statistics, a USGS remote-sensing report, and OpenStax Calculus Volume 3. It contains 66 questions: definitions, explanations, procedures, summaries, table questions, figure questions, mathematical concepts, multi-document comparisons, and unsupported questions.

The ground truth was labelled from `artifacts/metadata.jsonl` on the indexed corpus currently in this workspace. Its chunk IDs are valid only while the document, extraction logic, and chunking settings remain unchanged. Re-indexing the document can change the IDs; revalidate the labels after an intentional re-ingest.

Run dense-only, dense+reranker, hybrid BM25+FAISS, and hybrid+reranker retrieval locally:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

This writes `evaluation/predictions/baseline.json`, `evaluation/predictions/reranked.json`, `evaluation/predictions/hybrid.json`, `evaluation/predictions/hybrid_reranked.json`, and `evaluation/predictions/metrics.json`. These are retrieval-only metrics. Use `rubric.md` to score the resulting complete answers separately.

## Answer evaluation: deterministic versus local Ollama

Run the answer-level benchmark after the evaluation PDFs are indexed. The runner executes each labeled question with both the deterministic answer builder and the optional local Ollama synthesis provider:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe scripts\run_answer_evaluation.py
```

It writes the following under `evaluation/predictions/`:

- `answers_deterministic.json` and `answers_ollama.json`: raw answer text, source/page evidence, latency, citations, and synthesis/fallback mode;
- `answer_metrics.json`: automatic provenance, abstention, citation-coverage, labeled-page-overlap, acceptance-rate, and latency metrics;
- `answer_review_template.csv`: a deterministic, category-balanced sample for human review using `rubric.md`.

The automatic metrics intentionally do **not** claim semantic faithfulness or citation correctness. A citation can point to a returned page without entailing the written claim, and the benchmark labels do not list every potentially valid supporting page. Complete the rubric fields in the CSV before reporting correctness, completeness, faithfulness, or clarity.

Validate and aggregate the completed review before treating it as a release decision:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe scripts\score_semantic_review.py evaluation\predictions\answer_review_template.csv --output evaluation\predictions\semantic_review.json
```

The scorer rejects missing required rubric fields, keeps multimodal scoring separate from the ten-point primary score, and records the citation, faithfulness, abstention, and average-answer release gates. The API dashboard reads this artifact for **Faithfulness (review)**; it shows an unavailable value instead of a fabricated score until the review exists.

For a quick no-write smoke test, direct output to the ignored `results/` directory and run one deterministic query:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe scripts\run_answer_evaluation.py --modes deterministic --limit 1 --output-dir results\answer-evaluation-smoke
```
