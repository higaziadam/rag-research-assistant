from fastapi.testclient import TestClient
import pytest

import multimodal_rag.api as api
from multimodal_rag.schemas import UploadResponse


class FakeService:
    def __init__(self):
        self.last_session_id = None
        self.uploaded_filenames = []

    def upload_documents(self, files):
        self.uploaded_filenames = [file.filename for file in files]
        return UploadResponse(
            uploaded=self.uploaded_filenames,
            total_chunks=2,
            documents=[{"filename": self.uploaded_filenames[0], "pages": 1, "chunks": 2}],
        )

    def query(self, request):
        self.last_session_id = request.session_id
        return {
            "answer": "Grounded answer",
            "unsupported": False,
            "confidence": 0.9,
            "sources": [],
            "retrieval_scores": [],
            "latency_ms": 12.5,
            "session_id": request.session_id,
            "history": [request.query],
        }


def test_query_endpoint_validates_and_returns_the_public_schema(monkeypatch):
    fake_service = FakeService()
    monkeypatch.setattr(api, "get_service", lambda: fake_service)
    client = TestClient(api.app)

    response = client.post("/query", json={"query": "What is an aldehyde?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Grounded answer"
    assert client.post("/query", json={"query": "", "top_k": 0}).status_code == 422


def test_session_query_and_upload_are_forwarded_to_the_service(monkeypatch):
    fake_service = FakeService()
    monkeypatch.setattr(api, "get_service", lambda: fake_service)
    client = TestClient(api.app)

    session_response = client.post("/session/research-123/query", json={"query": "Summarize this."})
    upload_response = client.post(
        "/upload",
        files=[("files", ("notes.pdf", b"placeholder", "application/pdf"))],
    )

    assert session_response.status_code == 200
    assert fake_service.last_session_id == "research-123"
    assert upload_response.status_code == 200
    assert upload_response.json()["uploaded"] == ["notes.pdf"]


def test_duplicate_upload_is_rejected_before_indexing():
    service = api.RAGService.__new__(api.RAGService)
    service.documents = [{"filename": "notes.pdf", "pages": 1, "chunks": 2}]

    class File:
        filename = "notes.pdf"

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        service.upload_documents([File()])

    assert error.value.status_code == 409


def test_cors_allows_the_configured_frontend_origin():
    client = TestClient(api.app)

    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
