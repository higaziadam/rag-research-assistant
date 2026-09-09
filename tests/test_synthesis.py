from types import SimpleNamespace

import multimodal_rag.synthesis as synthesis
from multimodal_rag.synthesis import OllamaSynthesisClient


EVIDENCE = [
    {
        "source": "report.pdf",
        "page": 4,
        "text": "The report defines a process for evaluating and reducing model risk.",
        "table": "",
        "figure_caption": "",
        "type": "text",
    }
]


def _client() -> OllamaSynthesisClient:
    return OllamaSynthesisClient(
        enabled=True,
        base_url="http://127.0.0.1:11434",
        model="local-test-model",
        timeout_seconds=1,
        max_evidence=5,
        max_evidence_characters=500,
    )


def test_local_synthesis_accepts_only_evidence_citations(monkeypatch):
    def fake_post(*args, **kwargs):
        assert args[0].endswith("/api/generate")
        assert kwargs["json"]["stream"] is False
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "response": "**Answer**\n\nThe report presents a process for evaluating and reducing model risk. [report.pdf, p. 4]"
            },
        )

    monkeypatch.setattr(synthesis.requests, "post", fake_post)

    result = _client().synthesize(query="What does the report do?", intent="explanation", evidence=EVIDENCE)

    assert result is not None
    assert result.provider == "ollama"
    assert "[report.pdf, p. 4]" in result.answer


def test_local_synthesis_rejects_uncited_or_unrecognized_claims(monkeypatch):
    def fake_post(*args, **kwargs):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": "The report proves an unrelated claim. [other.pdf, p. 99]"},
        )

    monkeypatch.setattr(synthesis.requests, "post", fake_post)

    assert _client().synthesize(query="What does the report do?", intent="explanation", evidence=EVIDENCE) is None


def test_local_synthesis_cooldown_avoids_repeated_calls_when_ollama_is_down(monkeypatch):
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise synthesis.requests.ConnectionError("not running")

    monkeypatch.setattr(synthesis.requests, "post", unavailable)
    client = _client()

    assert client.synthesize(query="What does the report do?", intent="explanation", evidence=EVIDENCE) is None
    assert client.synthesize(query="What does the report do?", intent="explanation", evidence=EVIDENCE) is None
    assert len(calls) == 1
