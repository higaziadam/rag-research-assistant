import json
from threading import RLock

import numpy as np
import pytest

from multimodal_rag.data_models import DocumentChunk, RetrievalResult
from multimodal_rag.answer_evaluation import citation_coverage, extract_citations, select_review_query_ids, summarize_answer_records
from multimodal_rag.semantic_evaluation import summarize_semantic_reviews
from multimodal_rag.evaluation import evaluate_ranking_predictions
from multimodal_rag.jobs import IngestionJob
from multimodal_rag.retrieval import FAISSRetriever
from multimodal_rag.sparse_retrieval import BM25Retriever, reciprocal_rank_fusion
from multimodal_rag.train_reranker import prepare_dataset, train_reranker
import multimodal_rag.api as api


def test_retriever_rejects_invalid_embedding_and_query_dimensions():
    retriever = FAISSRetriever(embedding_dim=2)
    chunk = DocumentChunk(chunk_id="chunk", text="text")

    with pytest.raises(ValueError, match="shape"):
        retriever.add_chunks([chunk], np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))

    retriever.add_chunks([chunk], np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="dimensions"):
        retriever.retrieve(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def test_retriever_rejects_duplicate_chunk_ids():
    retriever = FAISSRetriever(embedding_dim=2)
    first_chunk = DocumentChunk(chunk_id="duplicate", text="first")
    second_chunk = DocumentChunk(chunk_id="duplicate", text="second")

    retriever.add_chunks([first_chunk], np.asarray([[1.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="unique"):
        retriever.add_chunks([second_chunk], np.asarray([[0.0, 1.0]], dtype=np.float32))


def test_retriever_can_remove_chunks_for_a_document_source():
    retriever = FAISSRetriever(embedding_dim=2)
    chunks = [
        DocumentChunk(chunk_id="first", text="first", source="first.pdf"),
        DocumentChunk(chunk_id="second", text="second", source="second.pdf"),
    ]
    retriever.add_chunks(chunks, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    filtered = retriever.without_sources({"first.pdf"})

    assert [chunk.source for chunk in filtered.chunks] == ["second.pdf"]
    assert filtered.retrieve(np.asarray([0.0, 1.0], dtype=np.float32), top_k=1)[0].chunk_id == "second"


def test_retriever_expands_dense_window_for_a_source_scoped_query():
    retriever = FAISSRetriever(embedding_dim=2)
    chunks = [
        DocumentChunk(chunk_id="dominant-1", text="first", source="dominant.pdf"),
        DocumentChunk(chunk_id="dominant-2", text="second", source="dominant.pdf"),
        DocumentChunk(chunk_id="scoped-1", text="target", source="scoped.pdf"),
    ]
    retriever.add_chunks(
        chunks,
        np.asarray([[1.0, 0.0], [0.9, 0.0], [0.8, 0.0]], dtype=np.float32),
    )

    scoped = retriever.retrieve(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1, sources={"scoped.pdf"})

    assert [result.chunk_id for result in scoped] == ["scoped-1"]


def test_retriever_diversifies_candidates_across_requested_sources():
    retriever = FAISSRetriever(embedding_dim=2)
    chunks = [
        DocumentChunk(chunk_id="first-1", text="first", source="first.pdf"),
        DocumentChunk(chunk_id="first-2", text="first", source="first.pdf"),
        DocumentChunk(chunk_id="second-1", text="second", source="second.pdf"),
        DocumentChunk(chunk_id="second-2", text="second", source="second.pdf"),
    ]
    retriever.add_chunks(
        chunks,
        np.asarray([[1.0, 0.0], [0.9, 0.0], [0.8, 0.0], [0.7, 0.0]], dtype=np.float32),
    )

    diversified = retriever.retrieve_diversified(
        np.asarray([1.0, 0.0], dtype=np.float32),
        sources={"first.pdf", "second.pdf"},
        per_source_k=2,
    )

    assert [result.source for result in diversified] == ["first.pdf", "second.pdf", "first.pdf", "second.pdf"]


def test_bm25_retrieves_exact_terms_and_honors_source_scope():
    chunks = [
        DocumentChunk(chunk_id="dense", text="Dense vector search uses embeddings.", source="dense.pdf"),
        DocumentChunk(chunk_id="sparse", text="BM25 ranks exact lexical terms.", source="sparse.pdf"),
    ]
    retriever = BM25Retriever(chunks)

    results = retriever.retrieve("How does BM25 rank lexical terms?", top_k=2, sources={"sparse.pdf"})

    assert [result.chunk_id for result in results] == ["sparse"]


def test_bm25_ignores_numeric_fragments_when_a_report_term_is_available():
    chunks = [
        DocumentChunk(chunk_id="numeric", text="Exercise 1.2 evaluates an integral.", source="math.pdf"),
        DocumentChunk(chunk_id="govern", text="GOVERN 1.2 recommends documented risk controls.", source="report.pdf"),
    ]
    retriever = BM25Retriever(chunks)

    results = retriever.retrieve("What does the GOVERN 1.2 table recommend?", top_k=2)

    assert results[0].chunk_id == "govern"


def test_reciprocal_rank_fusion_rewards_candidates_found_by_both_retrievers():
    first = RetrievalResult(chunk_id="first", score=0.9, text="first")
    shared = RetrievalResult(chunk_id="shared", score=0.8, text="shared")
    third = RetrievalResult(chunk_id="third", score=0.7, text="third")

    fused = reciprocal_rank_fusion([[first, shared], [shared, third]], top_k=3)

    assert [result.chunk_id for result in fused] == ["shared", "first", "third"]


def test_retriever_requires_metadata_when_loading_an_index(tmp_path):
    retriever = FAISSRetriever(embedding_dim=2)
    retriever.add_chunks([DocumentChunk(chunk_id="chunk", text="text")], np.asarray([[1.0, 0.0]], dtype=np.float32))
    index_path = tmp_path / "index.faiss"
    retriever.save(str(index_path), str(tmp_path / "metadata.jsonl"))

    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        FAISSRetriever(embedding_dim=2, index_path=str(index_path))


def test_evaluation_reads_prediction_and_ground_truth_files(tmp_path):
    predictions_path = tmp_path / "predictions.json"
    ground_truth_path = tmp_path / "ground_truth.json"
    predictions_path.write_text(json.dumps([{"query_id": "q1", "ranked_chunk_ids": ["a", "x", "b"]}]), encoding="utf-8")
    ground_truth_path.write_text(json.dumps([{"query_id": "q1", "relevance": ["a", "b"]}]), encoding="utf-8")

    metrics = evaluate_ranking_predictions(str(predictions_path), str(ground_truth_path), k=2)

    assert metrics["recall@2"] == 0.5
    assert 0.0 < metrics["ndcg@2"] <= 1.0


def test_evaluation_counts_relevant_chunks_missing_from_predictions(tmp_path):
    predictions_path = tmp_path / "predictions.json"
    ground_truth_path = tmp_path / "ground_truth.json"
    predictions_path.write_text('[{"query_id": "q1", "ranked_chunk_ids": ["a"]}]', encoding="utf-8")
    ground_truth_path.write_text('[{"query_id": "q1", "relevance": ["a", "b"]}]', encoding="utf-8")

    metrics = evaluate_ranking_predictions(str(predictions_path), str(ground_truth_path), k=1)

    assert metrics["recall@1"] == 0.5


def test_page_level_evaluation_counts_a_different_chunk_from_the_same_answer_page(tmp_path):
    predictions_path = tmp_path / "predictions.json"
    ground_truth_path = tmp_path / "ground_truth.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "ranked_chunk_ids": ["report.pdf-4-text-9-0"],
                    "ranked_results": [{"chunk_id": "report.pdf-4-text-9-0", "source": "report.pdf", "page": 4}],
                }
            ]
        ),
        encoding="utf-8",
    )
    ground_truth_path.write_text(
        json.dumps([{"query_id": "q1", "relevance": ["report.pdf-4-text-1-0"], "relevant_pages": [4]}]),
        encoding="utf-8",
    )

    metrics = evaluate_ranking_predictions(
        str(predictions_path),
        str(ground_truth_path),
        k=1,
        relevance_level="source_page",
    )

    assert metrics["recall@1"] == 1.0


def test_answer_evaluation_extracts_citations_and_excludes_source_footer_from_coverage():
    answer = (
        "**Answer**\n\n"
        "The report defines a risk-management process. [report.pdf, p. 4]\n\n"
        "- It requires periodic review. [report.pdf, p. 8]\n\n"
        "**Sources**\n"
        "report.pdf (pp. 4, 8)"
    )

    assert extract_citations(answer) == [("report.pdf", 4), ("report.pdf", 8)]
    assert citation_coverage(answer) == {"factual_blocks": 2, "cited_blocks": 2}


def test_answer_evaluation_metrics_keep_provenance_checks_separate_from_semantic_judgment():
    records = [
        {
            "mode": "ollama",
            "expected_supported": True,
            "unsupported": False,
            "citation_coverage": {"factual_blocks": 2, "cited_blocks": 2},
            "citations": [("report.pdf", 4)],
            "returned_source_pages": [("report.pdf", 4)],
            "ground_truth_source_pages": [("report.pdf", 4)],
            "synthesis_mode": "ollama",
            "latency_ms": 100.0,
        },
        {
            "mode": "ollama",
            "expected_supported": False,
            "unsupported": True,
            "citation_coverage": {"factual_blocks": 0, "cited_blocks": 0},
            "citations": [],
            "returned_source_pages": [],
            "ground_truth_source_pages": [],
            "synthesis_mode": "deterministic",
            "latency_ms": 20.0,
        },
    ]

    metrics = summarize_answer_records(records)["ollama"]

    assert metrics["support_classification_accuracy"] == 1.0
    assert metrics["citation_reference_validity"] == 1.0
    assert metrics["labeled_citation_pair_overlap"] == 1.0
    assert metrics["synthesis_acceptance_rate"] == 1.0
    assert "does not prove semantic entailment" in metrics["metric_notes"]["citation_reference_validity"]


def test_semantic_review_requires_all_scores_and_applies_release_thresholds():
    reviewed_rows = [
        {
            "query_id": "supported",
            "mode": "deterministic",
            "expected_supported": "True",
            "correctness": "2",
            "completeness": "2",
            "citation_correctness": "2",
            "faithfulness": "2",
            "clarity": "2",
            "multimodal_accuracy": "2",
            "unsupported_behavior": "",
        },
        {
            "query_id": "unsupported",
            "mode": "deterministic",
            "expected_supported": "False",
            "correctness": "",
            "completeness": "",
            "citation_correctness": "",
            "faithfulness": "",
            "clarity": "",
            "multimodal_accuracy": "",
            "unsupported_behavior": "2",
        },
    ]

    summary = summarize_semantic_reviews(reviewed_rows)["modes"]["deterministic"]

    assert summary["mean_primary_score_out_of_10"] == 10.0
    assert summary["criterion_rates"]["faithfulness"] == 1.0
    assert summary["unsupported_answer_accuracy"] == 1.0
    assert summary["release_gate_passed"] is True


def test_semantic_review_rejects_missing_supported_scores():
    with pytest.raises(ValueError, match="missing required completeness"):
        summarize_semantic_reviews(
            [
                {
                    "query_id": "incomplete",
                    "mode": "deterministic",
                    "expected_supported": "True",
                    "correctness": "2",
                    "completeness": "",
                    "citation_correctness": "2",
                    "faithfulness": "2",
                    "clarity": "2",
                }
            ]
        )


def test_answer_evaluation_review_sample_includes_unsupported_and_multiple_categories():
    questions = [
        {"query_id": "unsupported", "category": "unsupported", "expected_supported": False},
        {"query_id": "definition", "category": "definition", "expected_supported": True},
        {"query_id": "summary", "category": "summary", "expected_supported": True},
    ]

    assert select_review_query_ids(questions, sample_size=3) == ["unsupported", "definition", "summary"]


def test_training_dataset_accepts_jsonl(tmp_path):
    training_path = tmp_path / "training.jsonl"
    training_path.write_text(
        '{"query": "What is retrieval?", "passage": "Retrieval finds evidence.", "label": 1}\n',
        encoding="utf-8",
    )

    dataset = prepare_dataset(str(training_path))

    assert len(dataset) == 1
    assert dataset[0]["label"] == 1.0


def test_reranker_training_rejects_one_class_dataset_before_loading_a_model(tmp_path):
    training_path = tmp_path / "one-class.json"
    training_path.write_text(
        '[{"query": "What is retrieval?", "passage": "Retrieval finds evidence.", "label": 1}]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both relevant and non-relevant"):
        train_reranker(str(training_path), str(tmp_path / "checkpoint"))


def test_service_restores_persisted_index_and_document_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(api.settings, "artifacts_dir", tmp_path)
    monkeypatch.setattr(api.settings, "faiss_index_path", tmp_path / "faiss.index")
    monkeypatch.setattr(api.settings, "metadata_path", tmp_path / "metadata.jsonl")
    monkeypatch.setattr(api.settings, "documents_path", tmp_path / "documents.json")
    monkeypatch.setattr(api.settings, "jobs_path", tmp_path / "jobs.json")

    service = api.RAGService.__new__(api.RAGService)
    service.retriever = FAISSRetriever(embedding_dim=2)
    service.retriever.add_chunks([DocumentChunk(chunk_id="chunk", text="persistent text", source="paper.pdf", metadata={"page": 4})], np.asarray([[1.0, 0.0]], dtype=np.float32))
    service.documents = [{"filename": "paper.pdf", "pages": 4, "chunks": 1}]
    service._persist_state()

    restored = api.RAGService.__new__(api.RAGService)

    class FakeEmbeddings:
        embedding_dimension = 2

    restored.embedding_store = FakeEmbeddings()
    restored._restore_persisted_state()

    assert restored.documents == service.documents
    assert restored.retriever.retrieve(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)[0].source == "paper.pdf"


def test_background_ingestion_indexes_a_queued_document_and_persists_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(api.settings, "artifacts_dir", tmp_path)
    monkeypatch.setattr(api.settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(api.settings, "faiss_index_path", tmp_path / "faiss.index")
    monkeypatch.setattr(api.settings, "metadata_path", tmp_path / "metadata.jsonl")
    monkeypatch.setattr(api.settings, "documents_path", tmp_path / "documents.json")
    monkeypatch.setattr(api.settings, "jobs_path", tmp_path / "jobs.json")

    class FakeEmbeddings:
        def encode(self, texts):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    service = api.RAGService.__new__(api.RAGService)
    service.embedding_store = FakeEmbeddings()
    service.retriever = FAISSRetriever(embedding_dim=2)
    service.storage_lock = RLock()
    job = IngestionJob.create("queued.pdf")
    service.jobs = {job.job_id: job}
    service.documents = [
        {
            "filename": job.filename,
            "pages": 0,
            "chunks": 0,
            "status": "queued",
            "progress": 0,
            "message": "Queued for indexing.",
            "job_id": job.job_id,
        }
    ]
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / job.filename).write_bytes(b"placeholder")
    service._parse_pdf_to_chunks = lambda filename, data: [
        DocumentChunk(chunk_id="queued-1", text="Indexed background content.", source=filename, metadata={"page": 1})
    ]

    service._run_ingestion_job(job.job_id)

    assert service.jobs[job.job_id].status == "indexed"
    assert service.documents[0]["status"] == "indexed"
    assert service.documents[0]["chunks"] == 1
    assert (tmp_path / "jobs.json").is_file()


def test_failed_state_publication_does_not_mutate_the_live_index(tmp_path, monkeypatch):
    monkeypatch.setattr(api.settings, "artifacts_dir", tmp_path)
    monkeypatch.setattr(api.settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(api.settings, "documents_path", tmp_path / "documents.json")
    monkeypatch.setattr(api.settings, "jobs_path", tmp_path / "jobs.json")

    class FakeEmbeddings:
        def encode(self, texts):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    service = api.RAGService.__new__(api.RAGService)
    service.embedding_store = FakeEmbeddings()
    service.retriever = FAISSRetriever(embedding_dim=2)
    service.sparse_retriever = BM25Retriever([])
    service.storage_lock = RLock()
    job = IngestionJob.create("queued.pdf")
    service.jobs = {job.job_id: job}
    service.documents = [{"filename": job.filename, "pages": 0, "chunks": 0, "status": "queued", "progress": 0, "message": "Queued.", "job_id": job.job_id}]
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / job.filename).write_bytes(b"placeholder")
    service._parse_pdf_to_chunks = lambda filename, data: [
        DocumentChunk(chunk_id="queued-1", text="content", source=filename, metadata={"page": 1})
    ]
    service._persist_state = lambda **kwargs: (_ for _ in ()).throw(OSError("simulated disk failure"))

    service._run_ingestion_job(job.job_id)

    assert service.retriever.index.ntotal == 0
    assert service.documents[0]["status"] == "failed"
    assert service.jobs[job.job_id].status == "failed"


def test_terminal_job_retention_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(api.settings, "jobs_path", tmp_path / "jobs.json")
    monkeypatch.setattr(api.settings, "max_terminal_jobs", 2)
    service = api.RAGService.__new__(api.RAGService)
    service.documents = []
    service.jobs = {}
    for index in range(4):
        job = IngestionJob.create(f"{index}.pdf")
        job.update("failed", 0, "failed")
        service.jobs[job.job_id] = job

    service._persist_jobs()

    assert len(service.jobs) == 2
    assert len(json.loads((tmp_path / "jobs.json").read_text(encoding="utf-8"))) == 2


def test_interrupted_state_transaction_restores_the_previous_generation(tmp_path, monkeypatch):
    paths = {
        "artifacts_dir": tmp_path,
        "faiss_index_path": tmp_path / "faiss.index",
        "metadata_path": tmp_path / "metadata.jsonl",
        "documents_path": tmp_path / "documents.json",
        "jobs_path": tmp_path / "jobs.json",
    }
    for setting_name, path in paths.items():
        monkeypatch.setattr(api.settings, setting_name, path)

    state_paths = api.RAGService._state_paths()
    for path in state_paths:
        path.write_text("new", encoding="utf-8")
        path.with_name(f".{path.name}.rollback").write_text("previous", encoding="utf-8")
    (tmp_path / "state-transaction.json").write_text(
        json.dumps({"existed": {path.name: True for path in state_paths}}),
        encoding="utf-8",
    )

    api.RAGService._recover_interrupted_state()

    assert all(path.read_text(encoding="utf-8") == "previous" for path in state_paths)
    assert not (tmp_path / "state-transaction.json").exists()
