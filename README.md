# Multimodal RAG Research Assistant

A full-stack research assistant for PDF-based document Q&A. The app uses a Python FastAPI backend with FAISS retrieval and reranking, plus a Next.js frontend for asking questions and viewing sources.

## What is included right now

- PDF upload and indexing directly from the frontend
- Text chunking and metadata extraction from uploaded documents
- Dense vector search using FAISS
- Cross-encoder reranking on retrieved chunks
- Query endpoint with answer generation flow and unsupported-answer handling
- Frontend dashboard for asking questions and showing sources
- API docs via FastAPI at /docs
- Docker setup and pytest smoke tests

## Tech stack

- Python
- FastAPI
- FAISS
- SentenceTransformers / embeddings
- PyPDF
- Next.js + TypeScript
- Docker

## Project structure

```text
.
├── README.md
├── requirements.txt
├── pytest.ini
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   └── multimodal_rag/
│       ├── api.py
│       ├── config.py
│       ├── data_models.py
│       ├── embeddings.py
│       ├── retrieval.py
│       ├── reranker.py
│       ├── evaluation.py
│       └── train_reranker.py
├── frontend/
│   └── src/
├── data/
├── tests/
└── .github/workflows/
```

## Run locally

### 1. Backend

```powershell
cd "C:\Users\Uploa\Documents\RAG"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
uvicorn multimodal_rag.api:app --host 0.0.0.0 --port 8000
```

### 2. Frontend

```powershell
cd "C:\Users\Uploa\Documents\RAG\frontend"
npm install
npm run dev
```

Then open the frontend in the browser, upload one or more text-based PDFs, and ask questions. The app expects the backend to be running on port 8000. To use a different API address, create `frontend/.env.local` with `NEXT_PUBLIC_API_BASE_URL=http://your-host:8000`.

### Run the full stack with Docker

```powershell
docker compose -f docker/docker-compose.yml up --build
```

Open `http://localhost:3000`. For a deployed environment, set `NEXT_PUBLIC_API_BASE_URL` to the browser-accessible backend URL and set `CORS_ORIGINS` to the frontend URL before building the images. The frontend API URL is embedded during its Docker build.

## Current features in the app

- Upload PDF documents
- Search over indexed document chunks
- Return ranked sources with page references
- Show evaluation metrics such as Recall@5, MRR, and faithfulness
- Support unsupported answers when evidence is weak
- Demo UI for research assistant workflows

## Important gaps / next work

This project is functional as a prototype, but it is not yet a production-grade system. Important next steps are:

- Add persistent vector storage instead of in-memory indexing
- Improve PDF parsing and chunk quality for real documents
- Add a real multimodal workflow for images/tables/figures, not only text extraction
- Connect the reranker to a real dataset and evaluate with stronger metrics
- Add user authentication, session management, and database storage
- Clean up deployment and production configuration for real-world use

## Notes

This README is intentionally focused on the project as it exists now. The project is a working prototype for a research assistant, and the goal is to continue improving the backend, retrieval quality, and deployment reliability.
