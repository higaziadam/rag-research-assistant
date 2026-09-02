# Dataset notes

This folder is intentionally lightweight and acts as a sample dataset for a multimodal RAG project. Replace it with your own document corpus, annotation schema, and evaluation set when preparing the project for a portfolio or interview.

Recommended schema:

```json
{
  "id": "doc_001_chunk_01",
  "source": "technical_report.pdf",
  "section": "System design",
  "text": "The retrieval pipeline uses a dense vector index and a cross-encoder reranker.",
  "table": "| Metric | Score |\n| --- | --- |\n| Recall@5 | 0.82 |",
  "figure_caption": "Figure 3: latency by retrieval strategy",
  "type": "text",
  "relevant_queries": ["What is the retrieval pipeline?"],
  "metadata": {
    "page": 12,
    "document_type": "technical_report"
  }
}
```

For a stronger resume project, use a curated set of technical documents with domain-specific QA pairs and ground-truth relevance labels.
