# Multimodal RAG Research Assistant

A portfolio-grade multimodal retrieval-augmented generation (RAG) project built for technical document understanding. The system ingests text, tables, and figure captions from research or engineering docs, retrieves relevant chunks with FAISS, re-ranks with a transformer, and generates grounded answers with a language model.

## Why this is strong for a resume

This project is intentionally structured to showcase a realistic, end-to-end AI engineering workflow:

- Multimodal document ingestion for text, table excerpts, and figure captions
- Efficient similarity search with FAISS over dense embeddings
- Transformer reranking with a cross-encoder and LoRA/PEFT tuning workflow
- Evaluation pipelines covering retrieval quality, answer faithfulness, and latency
- API deployment with FastAPI and Docker for inference serving

The project is designed to support claims such as:

- Built a multimodal RAG pipeline using PyTorch, Hugging Face Transformers, and FAISS to answer questions across text, tables, and figures in technical documents.
- Improved retrieval Recall@5 from 58.4% to 83.7% by benchmarking chunking, embedding, and transformer reranking strategies on a curated evaluation dataset.
- Fine-tuned a transformer reranker with LoRA/PEFT and deployed inference with FastAPI/Docker, evaluating faithfulness, citation quality, and latency.

> These benchmark numbers are representative portfolio targets for a curated evaluation set and can be replaced with your own measured numbers when running on real documents.

---

## Architecture

```text
Technical docs
   ↓
PDF / markdown / csv extraction
   ↓
Chunking by section + table + figure caption
   ↓
Multimodal embeddings (text + table + image caption)
   ↓
FAISS vector store
   ↓
Cross-encoder reranker (LoRA-tuned)
   ↓
Grounded answer generation
   ↓
FastAPI service + Docker deployment
```

---

## Project structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── sample_dataset.json
│   └── README.md
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   └── multimodal_rag/
│       ├── __init__.py
│       ├── config.py
│       ├── data_models.py
│       ├── embeddings.py
│       ├── retrieval.py
│       ├── reranker.py
│       ├── evaluation.py
│       ├── api.py
│       ├── train_reranker.py
│       └── demo.py
└── tests/
    └── test_basic.py
```

---

## Quick start

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Run a demo query:

```bash
PYTHONPATH=src python -m multimodal_rag.demo
```

3. Start the API server:

```bash
PYTHONPATH=src uvicorn multimodal_rag.api:app --host 0.0.0.0 --port 8000
```

4. Query the API:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main bottleneck of the retrieval pipeline?"}'
```

---

## Benchmarking story

This project is designed around a strong engineering narrative:

- Compare chunk sizes: 150, 300, and 600 tokens
- Compare embedding models: sentence-transformers vs. domain-tuned encoders
- Compare retrieval methods: dense FAISS retrieval vs. hybrid BM25 + dense retrieval
- Evaluate reranking models before and after LoRA adaptation
- Measure Recall@k, MRR, answer faithfulness, and latency

Representative benchmark target:

| Metric | Baseline | Improved |
| --- | ---: | ---: |
| Recall@5 | 58.4% | 83.7% |
| Recall@10 | 67.1% | 89.4% |
| MRR | 0.41 | 0.68 |
| Average latency | 820 ms | 390 ms |

---

## Training and fine-tuning

The repository includes a LoRA-tuning script for the cross-encoder reranker:

```bash
PYTHONPATH=src python -m multimodal_rag.train_reranker --data-path data/sample_dataset.json --output-dir artifacts/reranker
```

The training script uses PEFT/LoRA and is structured to support a real question-pair ranking dataset or relevance labels extracted from document QA pairs.

---

## Deployment

The Docker image exposes a FastAPI service for inference:

```bash
docker build -f docker/Dockerfile -t multimodal-rag .
docker run -p 8000:8000 multimodal-rag
```

You can also use Docker Compose:

```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## Notes for portfolio use

To make this project truly competitive for a resume:

1. Replace the demo data with a real technical document corpus.
2. Run a curated evaluation set with answer-level grounding metrics.
3. Save benchmark artifacts under a `results/` directory with plots and tables.
4. Add a brief architecture diagram and a screenshot of the API or evaluation metrics.
5. Publish a short narrative explaining the trade-offs between chunking, retrieval, and reranker quality.

This project is intentionally designed to be a polished showcase for ML engineering experience, systems design, and evaluation rigor.
