import json
from threading import RLock

import numpy as np
import pytest

from multimodal_rag.data_models import DocumentChunk
from multimodal_rag.evaluation import evaluate_ranking_predictions
from multimodal_rag.jobs import IngestionJob
from multimodal_rag.retrieval import FAISSRetriever
from multimodal_rag.train_reranker import prepare_dataset
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


def test_training_dataset_accepts_jsonl(tmp_path):
    training_path = tmp_path / "training.jsonl"
    training_path.write_text(
        '{"query": "What is retrieval?", "passage": "Retrieval finds evidence.", "label": 1}\n',
        encoding="utf-8",
    )

    dataset = prepare_dataset(str(training_path))

    assert len(dataset) == 1
    assert dataset[0]["label"] == 1.0


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
