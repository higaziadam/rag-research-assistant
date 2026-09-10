# Multimodal RAG Evaluation Rubric

Score each supported answer from 0 to 2 on the first five criteria (maximum 10). Score multimodal accuracy when the question relies on a table, figure, or mathematical notation. Score unsupported-answer behavior separately for every unsupported question.

| Criterion | 2 — Strong | 1 — Partial | 0 — Failed |
| --- | --- | --- | --- |
| Retrieval relevance | At least one top-k result directly supports the answer; summary evidence covers distinct substantive sections. | Results are related but incomplete or indirect. | No result supports the answer. |
| Citation correctness | Every substantive claim has a citation that directly supports it. | Most citations are correct, but one is weak, imprecise, or missing. | Citations are wrong, irrelevant, or absent. |
| Faithfulness | All claims are grounded in retrieved evidence; uncertainty is stated where needed. | A minor unsupported inference or overstatement appears. | The answer invents, contradicts, or confidently overstates information. |
| Completeness | It addresses every major part of the question with sufficient detail. | It answers the core question but misses a material condition, step, or implication. | It does not meaningfully answer the question. |
| Clarity and structure | It is concise, readable, and suited to the question type. | It is understandable but repetitive or poorly organized. | It is confusing, fragmented, or unusable. |
| Multimodal accuracy | A table, figure, or equation is interpreted accurately, or an original PDF crop is used safely for verification. | The relevant material is identified but has limited explanation or needs verification. | A table, figure, or equation is misread or unreliable extraction is presented as fact. |
| Unsupported-answer behavior | It explicitly states that evidence is insufficient and does not invent an answer. | It hedges but makes a weak unsupported claim. | It hallucinates an answer despite insufficient evidence. |

## Pass criteria for the first benchmark

- Retrieval Recall@5: at least 0.80, measured at **source-page** granularity. A different extracted chunk from the same labeled source page is relevant because it supports the same cited-page verification workflow. Retain strict chunk-level metrics as diagnostics; do not substitute them for the primary source-page gate.
- Retrieval MRR: at least 0.70
- Citation correctness: at least 0.90
- Faithfulness: at least 0.90
- Unsupported-answer correctness: 1.00 for every unsupported question in the benchmark
- Average supported-answer score: at least 8.0 / 10

Record the page-level evidence and reviewer notes for every score below 2. A failed retrieval, incorrect citation, and unsupported-answer failure should each result in a separate issue, because their fixes are different.
