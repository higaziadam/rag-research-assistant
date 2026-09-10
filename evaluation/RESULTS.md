# Evaluation Results

**Run date:** 2026-09-09  
**Corpus:** 66 labelled questions across six indexed PDFs  
**Answer-review sample:** 24 representative questions evaluated in deterministic and local-Ollama modes (48 outputs total)

## Benchmark decision

**Current result: retrieval release gate passed; full release remains pending updated semantic review.**

The revised hybrid retrieval configuration meets the source-page Recall@5 and MRR targets. The answer-level automated checks also preserve perfect unsupported-answer behavior and citation provenance. Because retrieval behavior changed, the earlier manual semantic scores are historical rather than a valid release sign-off; the regenerated review CSV must be scored before claiming the full answer-quality release gate.

## Release re-evaluation — 2026-09-09

| Release gate | Result | Status |
| --- | ---: | --- |
| Hybrid + reranked source-page Recall@5 | 0.8306 (target >= 0.800) | Pass |
| Hybrid + reranked source-page MRR | 0.7250 (target >= 0.700) | Pass |
| Unsupported-answer correctness | 1.0000 (target 1.0000) | Pass |
| Citation provenance validity | 1.0000 (target >= 0.900) | Pass |
| Updated human semantic review | Not yet rescored | Pending |

The primary retrieval metric is source-page relevance. A retrieved chunk from an answer-bearing labeled PDF page is valid evidence even when its chunk boundary differs from the manually labeled chunk. Strict chunk-level metrics remain in `metrics.json` as a chunking-sensitivity diagnostic; the revised hybrid Recall@5 there is 0.6788.

The regenerated answer artifacts are under `results/answer-evaluation-release/`. The local-Ollama run accepted synthesis for 29.03% of supported answers and safely fell back to deterministic answers for the remainder; its p95 end-to-end latency was 21.84 seconds. This regression should be addressed before making local synthesis the default path.

## Historical retrieval benchmark

| Configuration | Recall@5 | MRR | nDCG |
| --- | ---: | ---: | ---: |
| Hybrid retrieval + reranking | 0.590 | 0.699 | 0.571 |
| Initial target | >= 0.800 | >= 0.700 | — |

These pre-change values document the baseline. The current release metrics are in `evaluation/predictions/metrics.json` and are summarized above.

## Automated answer checks

| Metric | Deterministic | Local Ollama |
| --- | ---: | ---: |
| Questions | 66 | 66 |
| Support / refusal classification | 96.97% | 96.97% |
| Unsupported-answer correctness | 100% | 100% |
| Citation block coverage | 97.14% | 100% |
| Citation references returned evidence | 100% | 100% |
| Labeled citation-page overlap* | 28.92% | 40.62% |
| Ollama synthesis accepted | — | 83.87% |
| Median latency | 507 ms | 5.61 s |
| p95 latency | 1.50 s | 13.53 s |

\*Page overlap is a lower-bound retrieval signal. A valid citation to an unlabeled supporting page is not counted. Citation-reference validity proves provenance only; it does not prove that a claim is semantically entailed by its citation.

## Completed semantic review

The reviewer scored each supported answer from 0–2 for correctness, completeness, citation correctness, faithfulness, and clarity, using the expected claims and retrieved source pages. Unsupported answers were scored separately under unsupported-answer behavior. The five primary scores total 10.

| Mode | Supported outputs reviewed | Mean score / 10 | Citation correctness | Faithfulness | Unsupported behavior |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deterministic | 20 | 7.30 | 92.5% | 95.0% | 8/8 correct |
| Local Ollama | 20 | 8.20 | 95.0% | 90.0% | 8/8 correct |
| Initial target | — | >= 8.00 | >= 90% | >= 90% | 100% |

The Ollama mode meets the sampled answer-quality threshold. It is not the default low-latency path: its p95 latency is 13.53 seconds, and 10 of 62 supported queries safely fell back to deterministic synthesis.

### Per-question primary scores

Scores are `correctness / completeness / citation correctness / faithfulness / clarity`.

| Query | Deterministic | Local Ollama | Review outcome |
| --- | --- | --- | --- |
| nist-001 | 2/2/2/2/2 | 2/2/2/2/2 | Correct acronym expansion. |
| nist-005 | 2/2/2/2/2 | 2/2/2/2/2 | Correct four considerations. |
| nist-016 | 1/1/2/2/2 | 2/2/2/2/2 | Deterministic answer is incomplete; Ollama synthesizes the requested actions. |
| nist-017 | 0/0/1/2/1 | 2/2/2/2/2 | Deterministic table fallback is safe but does not answer; Ollama interprets the table correctly. |
| nist-028 | 1/0/2/2/1 | 1/0/2/2/1 | Both return only a narrow purpose passage, omitting risks and actions. |
| transformer-004 | 1/1/2/2/1 | 2/2/2/2/2 | Ollama gives the complete scaled-attention calculation. |
| ipcc-001 | 0/0/2/2/1 | 0/0/2/2/1 | Both miss the causal claim about human greenhouse-gas emissions. |
| ipcc-002 | 2/2/2/2/1 | 2/2/2/2/2 | Correct 1.1°C result; deterministic wording is less clear. |
| ipcc-003 | 0/0/2/2/1 | 0/0/2/2/1 | Both describe the report rather than the impacts on vulnerable communities. |
| ipcc-004 | 1/1/2/2/2 | 2/2/2/2/2 | Ollama includes mitigation and adaptation actions explicitly. |
| ipcc-005 | 2/2/2/2/2 | 2/2/2/2/2 | Correct net-zero requirement. |
| ipcc-006 | 1/1/2/2/2 | 2/2/2/2/2 | Deterministic lead sentence is incomplete; Ollama states the definition. |
| poverty-001 | 1/1/2/2/1 | 2/2/2/2/2 | Deterministic lead gives the unrelated-individual rate before the correct national rate. |
| poverty-004 | 2/2/2/2/2 | 2/2/2/2/2 | Correct CPS ASEC source. |
| usgs-002 | 0/0/2/2/1 | 0/0/2/2/1 | Both fail to list the requested mining areas. |
| usgs-005 | 2/2/2/2/2 | 2/2/2/2/2 | Correct figure interpretation. |
| calculus-001 | 1/1/2/2/2 | 2/1/2/2/2 | Deterministic answer is formula-focused; Ollama explains distance along the curve but omits the line-segment approximation. |
| calculus-002 | 2/2/2/2/2 | 2/2/2/2/2 | Correct area, volume, and average-value uses. |
| comparison-001 | 0/0/1/1/1 | 0/0/1/0/1 | Neither source side answers the requested comparison; Ollama adds an unsupported AI/environment connection. |
| comparison-002 | 0/0/1/1/1 | 0/0/1/0/1 | Both miss the IPCC mitigation/adaptation actions and instead treat it as an AI report. |

## Unsupported-answer review

All eight reviewed unsupported outputs (four questions in two modes) received **2/2** for unsupported-answer behavior. They declined to answer without adding unsupported claims:

- `nist-030` — author's favorite programming language
- `transformer-007` — restaurant recommendation
- `ipcc-007` — tomorrow's hottest city
- `calculus-007` — tomorrow's Dow Jones close

## Failure ledger and next fixes

| Priority | Failure | Evidence | Recommended fix |
| --- | --- | --- | --- |
| P0 | Cross-document comparisons retrieve weak or one-sided evidence. | `comparison-001`, `comparison-002`, and the full-set `comparison-003` miss the required IPCC or Census claim. | Enforce per-document candidate quotas before reranking and require at least one answerable passage from every scoped document. Refuse the comparison when balanced support is absent. |
| P0 | Target passages can lose to lexical near-matches. | `ipcc-001`, `ipcc-003`, `usgs-002`, and `poverty-006` retrieve related pages but not the direct answer. | Add intent-aware query expansion and a focused evaluation set for causal, list, and numeric-comparison queries; tune BM25/dense fusion and candidate depth against Recall@5. |
| P1 | Document summaries may collapse to one introductory passage. | `nist-028` omits risks and actions in both modes. | Require summary evidence from distinct substantive sections and validate purpose, findings, and implications before synthesis. |
| P1 | Deterministic selector can lead with a less relevant passage despite better supporting context. | `poverty-001`, `ipcc-006`, and `transformer-004`. | Choose the highest entailment passage for the lead answer, rather than the first retrieved item; preserve secondary context separately. |
| P2 | Local synthesis has a high p95 latency. | Ollama p95 is 13.53 s; 10 supported answers fall back. | Stream responses in the UI, lower context size, use a smaller quantized model, and record validation-rejection reasons. |

## Reproducibility artifacts

- `evaluation/predictions/answer_metrics.json` — automatic answer checks.
- `evaluation/predictions/answers_deterministic.json` — 66 deterministic outputs.
- `evaluation/predictions/answers_ollama.json` — 66 local-Ollama outputs.
- `evaluation/predictions/answer_review_template.csv` — the reviewed 48-output sample and score-entry schema.
- `evaluation/rubric.md` — scoring definitions and benchmark targets.

No claims in this report treat provenance checks as a replacement for semantic review.
