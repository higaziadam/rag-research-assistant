import json

import numpy as np
import pytest

from multimodal_rag.data_models import DocumentChunk
from multimodal_rag.evaluation import evaluate_ranking_predictions
from multimodal_rag.retrieval import FAISSRetriever
from multimodal_rag.train_reranker import prepare_dataset


def test_retriever_rejects_invalid_embedding_and_query_dimensions():
    retriever = FAISSRetriever(embedding_dim=2)
    chunk = DocumentChunk(chunk_id="chunk", text="text")

    with pytest.raises(ValueError, match="shape"):
        retriever.add_chunks([chunk], np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))

    retriever.add_chunks([chunk], np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="dimensions"):
        retriever.retrieve(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


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


def test_training_dataset_accepts_jsonl(tmp_path):
    training_path = tmp_path / "training.jsonl"
    training_path.write_text(
        '{"query": "What is retrieval?", "passage": "Retrieval finds evidence.", "label": 1}\n',
        encoding="utf-8",
    )

    dataset = prepare_dataset(str(training_path))

    assert len(dataset) == 1
    assert dataset[0]["label"] == 1.0
