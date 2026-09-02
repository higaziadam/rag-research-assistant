import numpy as np
import pytest
import torch

from multimodal_rag.api import QueryRequest, RAGService
from multimodal_rag.data_models import DocumentChunk, RetrievalResult
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
