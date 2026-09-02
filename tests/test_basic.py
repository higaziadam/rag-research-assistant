import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import multimodal_rag.api as api
from multimodal_rag.api import QueryRequest, RAGService
from multimodal_rag.data_models import DocumentChunk, RetrievalResult
from multimodal_rag.evaluation import compute_ndcg_at_k, compute_recall_at_k
from multimodal_rag.retrieval import FAISSRetriever
from multimodal_rag.reranker import logits_to_scores


def test_retriever_returns_matching_chunk_for_query_embedding():
    # Use document-shaped input and deterministic embeddings so this remains an offline test.
    document_chunks = [
        DocumentChunk(chunk_id="first", text="first chunk"),
        DocumentChunk(chunk_id="second", text="second chunk"),
    ]
    retriever = FAISSRetriever(embedding_dim=2)
    retriever.add_chunks(document_chunks, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    results = retriever.retrieve(np.asarray([0.0, 1.0], dtype=np.float32), top_k=1)

    assert results[0].chunk_id == "second"


def test_retriever_persists_and_loads_with_an_explicit_metadata_path(tmp_path):
    chunks = [DocumentChunk(chunk_id="first", text="first chunk")]
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "chunks.jsonl"

    retriever = FAISSRetriever(embedding_dim=2)
    retriever.add_chunks(chunks, embeddings)
    retriever.save(str(index_path), str(metadata_path))

    loaded = FAISSRetriever(embedding_dim=999, index_path=str(index_path), metadata_path=str(metadata_path))

    assert loaded.embedding_dim == 2
    assert loaded.retrieve(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)[0].chunk_id == "first"


def test_empty_evaluation_inputs_return_zero_instead_of_nan():
    assert compute_recall_at_k([]) == 0.0
    assert compute_ndcg_at_k([]) == 0.0


def test_health_and_metrics_are_available_without_loading_models():
    api.service = None
    client = TestClient(api.app)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/metrics").json()["recall_at_5"] == 0.84
    assert api.service is None


def test_logits_to_scores_handles_single_logit_output():
    logits = torch.tensor([[0.2], [0.8]])
    scores = logits_to_scores(logits)
    assert scores == pytest.approx([0.2, 0.8])


def test_query_unpacks_reranked_candidate_before_building_evidence():
    result = RetrievalResult(
        chunk_id="chunk-1",
        score=0.4,
        text="Organic chemistry studies carbon-containing compounds.",
        source="chemistry.pdf",
        metadata={"page": 3},
    )

    class FakeEmbeddings:
        def encode_single(self, query):
            return query

    class FakeRetriever:
        def retrieve(self, query_embedding, top_k):
            return [result]

    class FakeReranker:
        def rerank(self, query, candidates, top_k):
            return [(candidates[0], 0.9)]

    service = RAGService.__new__(RAGService)
    service.embedding_store = FakeEmbeddings()
    service.retriever = FakeRetriever()
    service.reranker = FakeReranker()
    service.session_history = {"test": []}

    response = service.query(QueryRequest(query="What is organic chemistry?", session_id="test"))

    assert "Organic chemistry studies carbon-containing compounds." in response["answer"]
    assert response["retrieval_scores"] == [0.9]
    assert response["sources"] == [
        {
            "chunk_id": "chunk-1",
            "score": 0.9,
            "source": "chemistry.pdf",
            "page": 3,
            "text": "Organic chemistry studies carbon-containing compounds.",
            "table": "",
            "figure_caption": "",
            "section": "",
        }
    ]
