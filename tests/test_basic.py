import numpy as np
import pymupdf
import pytest
import torch
from threading import RLock
from fastapi.testclient import TestClient

import multimodal_rag.api as api
from multimodal_rag.api import QueryRequest, RAGService
from multimodal_rag.data_models import DocumentChunk, RetrievalResult
from multimodal_rag.evaluation import compute_ndcg_at_k, compute_recall_at_k
from multimodal_rag.math_extraction import LocalMathExtractor
from multimodal_rag.jobs import IngestionJob
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


def test_ingestion_job_persists_its_status_and_progress():
    job = IngestionJob.create("paper.pdf")
    job.update(status="embedding", progress=65, message="Creating embeddings.")

    restored = IngestionJob.from_dict(job.to_dict())

    assert restored.filename == "paper.pdf"
    assert restored.status == "embedding"
    assert restored.progress == 65


def test_summary_sentences_respect_the_configured_character_limit():
    sentence = "Organic chemistry " + "explains molecular structure " * 40 + "."
    evidence = [{"text": sentence}]

    summaries = RAGService._extract_summary_sentences("organic chemistry", evidence, max_sentence_characters=120)

    assert len(summaries) == 1
    assert len(summaries[0]) <= 123
    assert summaries[0].endswith("...")


def test_summary_sentences_skip_formula_corrupted_pdf_text():
    evidence = [
        {
            "text": (
                "The procedure for defining arc length is similar to the procedure used for defining area and volume. "
                "0d Thus the arc length function is given by ssxd – y x 1 s1 1 f f 9stdq2 dt – y x 1 S2t1 1 8t."
            )
        }
    ]

    summaries = RAGService._extract_summary_sentences("How is arc length defined?", evidence, max_sentence_characters=500)

    assert summaries == ["The procedure for defining arc length is similar to the procedure used for defining area and volume."]


def test_readable_prose_allows_ordinary_numbers():
    sentence = "For a radius of 3, the arc length is 6.28 units along the circle."

    assert RAGService._is_readable_prose(sentence)


def test_local_math_validation_requires_balanced_latex_delimiters():
    assert LocalMathExtractor._is_valid_latex(r"\int_0^1 x^2 \, dx")
    assert not LocalMathExtractor._is_valid_latex(r"\frac{a}{b")


def test_local_math_extractor_marks_equations_as_source_only_without_a_checkpoint(tmp_path):
    document = pymupdf.open()
    document.new_page().insert_text((200, 300), "x = 2")
    pdf_bytes = document.tobytes()
    document.close()
    extractor = LocalMathExtractor(enabled=True, checkpoint_path=tmp_path / "weights.pth")

    pages = extractor.extract_pages(pdf_bytes)

    assert pages[0].equations == [
        {
            "latex": "",
            "status": "source_only",
            "confidence": 0.0,
            "bounding_box": pages[0].equations[0]["bounding_box"],
        }
    ]


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
            self.last_query = query
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
    service.session_history = {"test": ["Earlier conversation context."]}
    service.storage_lock = RLock()

    response = service.query(QueryRequest(query="What is organic chemistry?", session_id="test"))

    assert "Organic chemistry studies carbon-containing compounds." in response["answer"]
    assert "Earlier conversation context." in service.embedding_store.last_query
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
            "equations": [],
        }
    ]
