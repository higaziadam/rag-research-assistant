# Evaluation Results

**Run date:** 2026-09-10
**Corpus:** 66 labelled questions across six indexed PDFs  
**Semantic-review sample:** 24 category-balanced questions in deterministic and local-Ollama modes (48 outputs total)

## Release-evaluation decision

**The automated retrieval and sampled rubric gates pass.** The project is a
release candidate for local, document-grounded research use. The local-Ollama
mode remains opt-in because it has materially higher latency, and the known
answer-quality limitations below should be considered before high-stakes use.

The rubric review was completed from the regenerated current-code outputs in
`evaluation/predictions/answer_review_template.csv`. It is reproducible through
`evaluation/predictions/semantic_review.json`. An independent human reviewer
should still sign off on this sample before representing the semantic scores as
an externally audited result.

| Release gate | Result | Status |
| --- | ---: | --- |
| Hybrid + reranked source-page Recall@5 | 0.8306 (target >= 0.800) | Pass |
| Hybrid + reranked source-page MRR | 0.7250 (target >= 0.700) | Pass |
| Deterministic mean rubric score | 8.30 / 10 (target >= 8.00) | Pass |
| Local-Ollama mean rubric score | 8.85 / 10 (target >= 8.00) | Pass |
| Citation correctness, reviewed sample | 100% (target >= 90%) | Pass |
| Faithfulness, reviewed sample | 100% (target >= 90%) | Pass |
| Unsupported-answer correctness | 100% (target 100%) | Pass |

The primary retrieval measure is source-page relevance: a retrieved chunk from
an answer-bearing labelled PDF page is valid evidence even if its chunk boundary
differs from the manually labelled chunk. The strict chunk-level hybrid+rereanked
Recall@5 diagnostic is 0.6788 and should not replace the primary metric.

## Answer benchmark results

| Metric | Deterministic | Local Ollama |
| --- | ---: | ---: |
| Questions | 66 | 66 |
| Support / refusal classification | 98.48% | 98.48% |
| Unsupported-answer correctness | 100% | 100% |
| Citation block coverage | 91.90% | 100% |
| Citation references returned evidence | 100% | 100% |
| Labeled citation-page overlap* | 32.56% | 50.00% |
| Accepted local synthesis | — | 95.16% |
| Median latency | 1.10 s | 4.95 s |
| p95 latency | 2.38 s | 10.68 s |

\*Labeled-page overlap is a lower-bound retrieval signal. A valid citation to
an unlabeled supporting page is not counted. Citation-reference validity proves
provenance only; it does not independently prove semantic entailment.

## Rubric review

Supported outputs were scored from 0–2 for correctness, completeness, citation
correctness, faithfulness, and clarity. Unsupported outputs were scored
separately for abstention behavior. Multimodal accuracy remains separate from
the ten-point primary score.

| Mode | Supported reviewed | Mean / 10 | Correctness | Completeness | Citation correctness | Faithfulness | Clarity | Unsupported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deterministic | 20 | 8.30 | 70.0% | 65.0% | 100% | 100% | 80.0% | 4 / 4 |
| Local Ollama | 20 | 8.85 | 77.5% | 72.5% | 100% | 100% | 92.5% | 4 / 4 |

The review confirms that the citation guardrails and abstention behavior are
reliable on this sample. It also shows that passing the aggregate gate does not
mean every question is fully answered.

## Known limitations and follow-up work

| Priority | Finding | Evidence from current sample | Follow-up |
| --- | --- | --- | --- |
| P0 | Some direct factual prompts retrieve related rather than answer-bearing passages. | Both modes miss the human greenhouse-gas-emissions cause in `ipcc-001`. | Add focused causal-query expansion and direct-answer selection tests. |
| P0 | Cross-document comparisons can lack one required side of the comparison. | `comparison-001` and `comparison-002` miss the requested causal or action evidence. | Require source-by-source answer coverage and abstain when balanced evidence is unavailable. |
| P1 | List coverage can be incomplete. | `usgs-002` omits the Antelope Range near Marysvale in Ollama mode and fails in deterministic mode. | Add list-item coverage checks before synthesis. |
| P1 | Deterministic answers sometimes lead with indirect context. | `transformer-004` and `calculus-002`. | Improve direct-claim ranking and answer-first selection. |
| P2 | Local synthesis remains slower. | Local-Ollama p95 latency is 10.68 seconds. | Stream output, reduce context, or evaluate a smaller quantized model. |

## Reproducibility artifacts

- `evaluation/predictions/metrics.json` — retrieval configurations and primary source-page metrics.
- `evaluation/predictions/answers_deterministic.json` — 66 deterministic outputs.
- `evaluation/predictions/answers_ollama.json` — 66 local-Ollama outputs.
- `evaluation/predictions/answer_metrics.json` — automated answer checks.
- `evaluation/predictions/answer_review_template.csv` — completed rubric review with per-output notes.
- `evaluation/predictions/semantic_review.json` — validated, machine-readable review summary.
- `evaluation/rubric.md` — scoring definitions and release thresholds.
