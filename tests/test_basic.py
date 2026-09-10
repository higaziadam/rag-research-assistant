import numpy as np
import pymupdf
import pytest
import torch
from concurrent.futures import Future
from threading import RLock
from fastapi.testclient import TestClient

import multimodal_rag.api as api
from multimodal_rag.api import QueryRequest, RAGService
from multimodal_rag.data_models import DocumentChunk, RetrievalResult
from multimodal_rag.document_extraction import StructuredPdfExtractor, render_pdf_region
from multimodal_rag.evaluation import compute_mrr, compute_ndcg_at_k, compute_recall_at_k
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


def test_completed_ingestion_jobs_do_not_accumulate_future_references():
    class ImmediateExecutor:
        @staticmethod
        def submit(function, *args):
            future = Future()
            function(*args)
            future.set_result(None)
            return future

    service = RAGService.__new__(RAGService)
    service.job_futures = {}
    service.job_executor = ImmediateExecutor()
    service.storage_lock = RLock()
    service._run_ingestion_job = lambda job_id: None

    service._schedule_job("completed-job")

    assert service.job_futures == {}


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
    assert compute_mrr([]) == 0.0


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


def test_summary_sentences_skip_multiline_untrusted_math():
    evidence = [
        {
            "text": (
                "The formula is shown below:\n"
                "s(t) = ∫\n"
                "a\n"
                "b\n"
                "f(u) du."
            )
        }
    ]

    summaries = RAGService._extract_summary_sentences("What is the arc length formula?", evidence, max_sentence_characters=500)

    assert summaries == []


def test_grounded_answer_synthesis_uses_claim_level_citations_and_source_math():
    evidence = [
        {
            "source": "calculus.pdf",
            "page": 106,
            "text": "The arc length of a parametric curve can be calculated by using the formula s = ∫.",
            "equations": [{"bounding_box": [72, 72, 300, 140]}],
        },
        {
            "source": "calculus.pdf",
            "page": 293,
            "text": "A vector-valued function can define a curve whose arc length is evaluated over the parameter interval.",
            "equations": [],
        },
    ]

    answer = RAGService._build_grounded_answer("What is the equation for arc length?", evidence)

    assert "The arc length of a parametric curve can be calculated by using the formula. [calculus.pdf, p. 106]" in answer
    assert "**Supporting context**" in answer
    assert "**Mathematical notation**" in answer


def test_grounded_answer_synthesis_prefers_a_definition_over_a_related_topic():
    evidence = [
        {
            "source": "calculus.pdf",
            "page": 295,
            "text": "An important topic related to arc length is curvature.",
            "equations": [],
        },
        {
            "source": "calculus.pdf",
            "page": 293,
            "text": (
                "We now have a formula for the arc length of a curve defined by a vector-valued function. "
                "If a vector-valued function represents the position of a particle in space as a function of time, "
                "then the arc-length function measures how far that particle travels as a function of time."
            ),
            "equations": [{"bounding_box": [72, 72, 300, 140]}],
        },
    ]

    answer = RAGService._build_grounded_answer("Tell me about arc length", evidence)

    direct_answer = answer.split("**Supporting context**", 1)[0]
    assert "measures how far that particle travels" in direct_answer
    assert "curvature" not in direct_answer


def test_grounded_answer_prefers_the_meaning_of_a_math_concept_over_its_formula():
    evidence = [
        {
            "source": "calculus.pdf",
            "page": 43,
            "text": "If a particle travels from point A to point B along a curve, then the distance that particle travels is the arc length. To develop a formula for arc length, we start with an approximation by line segments.",
            "equations": [],
        },
        {
            "source": "calculus.pdf",
            "page": 106,
            "text": "The arc length of a parametric curve can be calculated by using the formula.",
            "equations": [],
        },
    ]

    answer = RAGService._build_grounded_answer("What does arc length represent for a parametric curve?", evidence)

    assert "distance that particle travels" in answer.split("**Supporting context**", 1)[0]


def test_grounded_answer_routes_summary_requests_to_the_summary_layout_before_purpose_shortcuts():
    evidence = [
        {
            "source": "report.pdf",
            "page": 2,
            "text": "This document is a companion resource for managing risks in AI systems.",
            "equations": [],
        },
        {
            "source": "report.pdf",
            "page": 8,
            "text": "The report identifies data, security, and governance risks for deployed systems.",
            "equations": [],
        },
        {
            "source": "report.pdf",
            "page": 16,
            "text": "Organizations should document limitations and monitor systems after deployment.",
            "equations": [],
        },
    ]

    answer = RAGService._build_grounded_answer("Summarize this document's purpose, risks, and actions.", evidence)

    assert "**Key findings**" in answer
    assert "**Recommendations / implications**" in answer


def test_intent_search_query_requests_foundational_evidence():
    search_query = RAGService._intent_search_query("Tell me about arc length", "Tell me about arc length")

    assert search_query.startswith("Tell me about arc length")
    assert "introductory definition" in search_query


def test_procedure_synthesis_uses_numbered_evidence_backed_steps():
    evidence = [
        {
            "source": "differential-equations.pdf",
            "page": 858,
            "text": "The method of undetermined coefficients uses the form of r(x) to choose a guess for the particular solution.",
            "equations": [],
        },
        {
            "source": "differential-equations.pdf",
            "page": 859,
            "text": "When r(x) is an exponential function, the particular solution can have the form y_p(x) = A e^(3x).",
            "equations": [{"bounding_box": [72, 72, 300, 140]}],
        },
        {
            "source": "differential-equations.pdf",
            "page": 862,
            "text": "Substitute the guess into the differential equation and solve for the unknown coefficient.",
            "equations": [],
        },
    ]

    answer = RAGService._build_grounded_answer(
        "How do I solve undetermined coefficients when r(x) is exponential?",
        evidence,
    )

    assert "**Evidence-based steps**" in answer
    assert "1. " in answer
    assert "2. " in answer
    assert "[differential-equations.pdf, p." in answer


def test_table_claims_preserve_labeled_source_values_for_procedures():
    table = "| r(x) | Initial guess for y p(x) |\n| --- | --- |\n| aeλx | Aeλx |"

    claims = RAGService._extract_table_claims(table)

    assert claims == ["For r(x) = aeλx, the initial guess for y_p(x) is Aeλx."]


def test_non_math_table_is_not_rewritten_as_a_differential_equation_claim():
    table = "| Action ID | Suggested action |\n| MG-1.3-001 | Monitor deployed systems |"

    assert RAGService._extract_table_claims(table) == []


def test_question_intent_covers_general_research_question_types():
    assert RAGService._question_intent("What is organic chemistry?") == "definition"
    assert RAGService._question_intent("How do I calculate arc length?") == "procedure"
    assert RAGService._question_intent("What can double integrals calculate?") == "definition"
    assert RAGService._question_intent("What does arc length represent?") == "definition"
    assert RAGService._question_intent("Compare the two experimental methods") == "comparison"
    assert RAGService._question_intent("Summarize the report") == "summary"
    assert RAGService._question_intent("What does Figure 2 show?") == "visual"


def test_summary_synthesis_has_purpose_findings_recommendations_and_sources():
    evidence = [
        {
            "source": "ai-report.pdf",
            "page": 2,
            "text": "The purpose of this report is to provide a practical framework for managing AI risks.",
            "equations": [],
        },
        {
            "source": "ai-report.pdf",
            "page": 18,
            "text": "The report identifies documentation, testing, and governance as key controls for AI systems.",
            "equations": [],
        },
        {
            "source": "ai-report.pdf",
            "page": 42,
            "text": "Organizations should document model limitations and monitor deployed systems for changing risks.",
            "equations": [],
        },
    ]

    answer = RAGService._build_grounded_answer("Summarize this document", evidence)

    assert "**Purpose**" in answer
    assert "**Key findings**" in answer
    assert "**Recommendations / implications**" in answer
    assert "**Sources**" in answer
    assert "[ai-report.pdf, p. 2]" in answer


def test_summary_candidate_filter_excludes_contents_and_reference_boilerplate():
    contents = DocumentChunk(
        chunk_id="contents",
        source="report.pdf",
        text="1. Introduction ........ 1 2. Scope ........ 3",
        metadata={"page": 1},
    )
    substantive = DocumentChunk(
        chunk_id="substantive",
        source="report.pdf",
        text="This report provides practical guidance for organizations that manage risks throughout the lifecycle of AI systems.",
        metadata={"page": 2},
    )

    assert not RAGService._is_summary_candidate(contents)
    assert RAGService._is_summary_candidate(substantive)


def test_summary_coverage_requires_substantive_evidence_from_multiple_pages():
    evidence = [
        {"source": "report.pdf", "page": 2},
        {"source": "report.pdf", "page": 8},
        {"source": "report.pdf", "page": 21},
    ]

    assert RAGService._has_summary_coverage(evidence, requested_top_k=5)
    assert not RAGService._has_summary_coverage(evidence[:2], requested_top_k=5)


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


def test_local_math_extractor_skips_image_rendering_without_ocr_weights(tmp_path, monkeypatch):
    document = pymupdf.open()
    document.new_page().insert_text((200, 300), "x = 2")
    pdf_bytes = document.tobytes()
    document.close()
    extractor = LocalMathExtractor(enabled=True, checkpoint_path=tmp_path / "weights.pth")
    monkeypatch.setattr(extractor, "_crop_equation", lambda page, rectangle: pytest.fail("OCR crop should not be rendered"))

    pages = extractor.extract_pages(pdf_bytes)

    assert pages[0].equations[0]["status"] == "source_only"


def test_local_math_extractor_groups_positioned_formula_fragments(tmp_path):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((220, 250), "x =")
    page.insert_text((280, 250), "2")
    pdf_bytes = document.tobytes()
    document.close()
    extractor = LocalMathExtractor(enabled=False, checkpoint_path=tmp_path / "weights.pth")

    equations = extractor.extract_pages(pdf_bytes)[0].equations

    assert len(equations) == 1
    x0, y0, x1, y1 = equations[0]["bounding_box"]
    assert x0 <= 220 < x1
    assert y0 <= 250 < y1


def test_local_math_extractor_rejects_table_of_contents_lines(tmp_path):
    extractor = LocalMathExtractor(enabled=False, checkpoint_path=tmp_path / "weights.pth")
    rectangle = pymupdf.Rect(200, 200, 400, 220)

    assert not extractor._looks_like_equation("1. Introduction ........ 1", rectangle, page_width=600)
    assert extractor._looks_like_equation("x = 2", rectangle, page_width=600)


def test_local_math_ocr_uses_a_bounded_document_budget(tmp_path, monkeypatch):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((200, 200), "x = 2")
    page.insert_text((200, 300), "y = 3")
    pdf_bytes = document.tobytes()
    document.close()
    extractor = LocalMathExtractor(
        enabled=True,
        checkpoint_path=tmp_path / "weights.pth",
        max_ocr_equations_per_document=1,
    )
    transcriptions = []
    monkeypatch.setattr(extractor, "_get_model", lambda: object())
    monkeypatch.setattr(extractor, "_crop_equation", lambda page, rectangle: object())
    monkeypatch.setattr(extractor, "_transcribe", lambda model, image: transcriptions.append(image) or "x = 2")

    equations = extractor.extract_pages(pdf_bytes)[0].equations

    assert len(equations) == 2
    assert len(transcriptions) == 1
    assert equations[0]["status"] == "needs_verification"
    assert equations[1]["status"] == "source_only"


def test_equations_attach_to_the_preceding_explanation():
    equations = [{"bounding_box": [520, 130, 700, 180]}]

    attached = RAGService._equations_for_element(equations, [72, 100, 500, 125])

    assert attached == equations


def test_structured_pdf_extraction_preserves_text_layout_and_quality_flags():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "1 Introduction")
    page.insert_text((72, 108), "Layout-aware extraction keeps this paragraph connected to its page and section.")
    pdf_bytes = document.tobytes()
    document.close()

    extracted_page = StructuredPdfExtractor().extract_pages(pdf_bytes)[0]

    text_element = next(element for element in extracted_page.elements if element.type == "text")
    assert text_element.section == "1 Introduction"
    assert text_element.bounding_box[0] >= 0
    assert "Layout-aware extraction" in text_element.text


def test_structured_pdf_extraction_uses_an_injected_equation_extractor_in_one_pass():
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "x = 2")
    pdf_bytes = document.tobytes()
    document.close()

    class FakeEquationExtractor:
        def __init__(self):
            self.page_count = 0

        def extract_page(self, page):
            self.page_count += 1
            return type("Extraction", (), {"equations": []})()

    equation_extractor = FakeEquationExtractor()
    pages = StructuredPdfExtractor().extract_pages(pdf_bytes, equation_extractor=equation_extractor)

    assert len(pages) == 1
    assert equation_extractor.page_count == 1


def test_structured_pdf_extraction_marks_scanned_style_pages_as_low_quality():
    document = pymupdf.open()
    document.new_page()
    pdf_bytes = document.tobytes()
    document.close()

    extracted_page = StructuredPdfExtractor().extract_pages(pdf_bytes)[0]

    assert extracted_page.elements == []
    assert extracted_page.quality_flags == ["no_extractable_text"]


def test_structured_pdf_extraction_creates_a_table_evidence_element():
    document = pymupdf.open()
    page = document.new_page()
    for x in (72, 180, 288):
        page.draw_line((x, 72), (x, 144))
    for y in (72, 96, 120, 144):
        page.draw_line((72, y), (288, y))
    page.insert_text((80, 89), "Method")
    page.insert_text((190, 89), "Score")
    page.insert_text((80, 113), "Dense")
    page.insert_text((190, 113), "0.84")
    pdf_bytes = document.tobytes()
    document.close()

    extracted_page = StructuredPdfExtractor().extract_pages(pdf_bytes)[0]

    table = next(element for element in extracted_page.elements if element.type == "table")
    assert "Method" in table.table
    assert "Score" in table.table


def test_region_renderer_returns_a_png_for_a_valid_citation(tmp_path):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Cited source region")
    pdf_path = tmp_path / "source.pdf"
    document.save(pdf_path)
    document.close()

    preview = render_pdf_region(pdf_path, 1, [50, 50, 250, 120])

    assert preview.startswith(b"\x89PNG")


def test_health_and_metrics_are_available_without_loading_models():
    api.service = None
    client = TestClient(api.app)

    assert client.get("/health").json() == {"status": "ok"}
    metrics = client.get("/metrics").json()
    assert metrics["recall_at_5"] == pytest.approx(0.8306451613)
    assert metrics["citation_accuracy"] == 1.0
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
        def retrieve(self, query_embedding, top_k, sources=None):
            self.top_k = top_k
            self.sources = sources
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
    assert service.embedding_store.last_query.startswith("What is organic chemistry?")
    assert "introductory definition" in service.embedding_store.last_query
    assert response["answer_intent"] == "definition"
    assert service.retriever.top_k >= 30
    assert service.retriever.sources is None
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
            "type": "text",
            "bounding_box": None,
            "quality_flags": [],
        }
    ]


def test_query_accepts_directly_supported_evidence_with_a_negative_reranker_logit():
    result = RetrievalResult(
        chunk_id="nist-1",
        score=0.4,
        text="The National Institute of Standards and Technology (NIST) develops standards.",
        source="nist.pdf",
        metadata={"page": 3},
    )

    class FakeEmbeddings:
        def encode_single(self, query):
            return query

    class FakeRetriever:
        def retrieve(self, query_embedding, top_k, sources=None):
            return [result]

    class FakeReranker:
        def rerank(self, query, candidates, top_k):
            return [(candidates[0], -1.2)]

    service = RAGService.__new__(RAGService)
    service.embedding_store = FakeEmbeddings()
    service.retriever = FakeRetriever()
    service.reranker = FakeReranker()
    service.session_history = {}
    service.storage_lock = RLock()

    response = service.query(QueryRequest(query="What does NIST stand for?"))

    assert response["unsupported"] is False
    assert response["confidence"] >= 0.5
    assert "National Institute of Standards and Technology" in response["answer"]


def test_table_identifier_is_prioritized_and_counts_as_direct_support():
    matching_table = DocumentChunk(
        chunk_id="govern-1-2",
        source="nist.pdf",
        type="table",
        text="GOVERN 1.2 recommends documented training-data provenance.",
        table="| GV-1.2-001 | Establish transparency policies |",
        metadata={"page": 18},
    )
    adjacent_table = DocumentChunk(
        chunk_id="govern-1-3",
        source="nist.pdf",
        type="table",
        text="GOVERN 1.3 describes risk tolerance.",
        metadata={"page": 18},
    )

    class FakeRetriever:
        chunks = [adjacent_table, matching_table]

    service = RAGService.__new__(RAGService)
    service.retriever = FakeRetriever()

    candidates = service._exact_identifier_candidates("What recommendations does GOVERN 1.2 make?", {"nist.pdf"})
    evidence = [{"source": "nist.pdf", "text": matching_table.text, "table": matching_table.table, "figure_caption": ""}]

    assert [candidate.chunk_id for candidate in candidates] == ["govern-1-2"]
    assert service._evidence_supports_query("What recommendations does GOVERN 1.2 make?", evidence, "visual")


def test_table_recommendation_answer_uses_structured_action_rows_not_flattened_prose():
    evidence = [
        {
            "source": "nist.pdf",
            "page": 18,
            "type": "table",
            "text": "Table: GOVERN 1.2: Trustworthy AI characteristics are integrated into organizational practices.. : Action ID",
            "table": (
                "| Action ID | Suggested Action | GAI Risks |\n"
                "| --- | --- | --- |\n"
                "| GV-1.2-001 | Establish transparency policies for training data provenance. | Data Privacy; Information Integrity |\n"
                "| GV-1.2-002 | Evaluate safety capabilities before and after deployment. | Information Security |"
            ),
            "figure_caption": "",
            "equations": [],
        }
    ]

    answer = RAGService._build_grounded_answer("What recommendations does the GOVERN 1.2 table make?", evidence)

    assert "**GOVERN 1.2 recommendations**" in answer
    assert "**GV-1.2-001**" in answer
    assert "**GV-1.2-002**" in answer
    assert "Related risks: Data Privacy; Information Integrity" in answer
    assert "Table:" not in answer


def test_acronym_question_returns_the_explicit_source_expansion():
    evidence = [
        {
            "source": "nist.pdf",
            "page": 3,
            "text": "The National Institute of Standards and Technology (NIST) develops standards.",
            "table": "",
            "figure_caption": "",
            "equations": [],
        }
    ]

    answer = RAGService._build_grounded_answer("What does NIST stand for?", evidence)

    assert "**NIST** stands for **National Institute of Standards and Technology**" in answer


def test_document_purpose_answer_prefers_an_introductory_scope_statement_over_a_disclaimer():
    evidence = [
        {
            "source": "report.pdf",
            "page": 3,
            "text": "Disclaimer: Commercial entities may be identified in this document.",
            "table": "",
            "figure_caption": "",
            "equations": [],
        },
        {
            "source": "report.pdf",
            "page": 5,
            "text": "This document is a companion resource for an AI risk management framework.",
            "table": "",
            "figure_caption": "",
            "equations": [],
        },
    ]

    answer = RAGService._build_grounded_answer("What is the purpose of this document?", evidence)

    assert "**Purpose**" in answer
    assert "companion resource" in answer
    assert "Disclaimer" not in answer


def test_document_purpose_answer_rejects_an_incidental_content_purpose():
    evidence = [
        {
            "source": "report.pdf",
            "page": 2,
            "text": "This report is a companion resource for organizations managing AI risks.",
            "equations": [],
        },
        {
            "source": "report.pdf",
            "page": 9,
            "text": "Generative systems can create content intended to impersonate other people.",
            "equations": [],
        },
    ]

    answer = RAGService._build_document_purpose_answer("What is the purpose of this report?", evidence)

    assert answer is not None
    assert "companion resource" in answer
    assert "impersonate" not in answer


def test_document_purpose_pool_prioritizes_scope_over_front_matter():
    service = RAGService.__new__(RAGService)
    service.retriever = type(
        "Retriever",
        (),
        {
            "chunks": [
                DocumentChunk(
                    chunk_id="front-matter",
                    source="report.pdf",
                    text="Such identification is not intended to imply recommendation or endorsement.",
                    metadata={"page": 3},
                ),
                DocumentChunk(
                    chunk_id="scope",
                    source="report.pdf",
                    text="This document is a companion resource for an AI risk management framework.",
                    metadata={"page": 5},
                ),
            ]
        },
    )()

    candidates = service._document_purpose_candidate_pool({"report.pdf"})

    assert [candidate.chunk_id for candidate in candidates] == ["scope"]


def test_claim_cleanup_removes_a_stray_pdf_page_number():
    assert RAGService._clean_claim("13 As this document was focused on the primary considerations.") == "As this document was focused on the primary considerations."


def test_current_information_query_is_not_supported_by_a_historical_document_example():
    evidence = [{"source": "calculus.pdf", "text": "The price per apple is fifty cents.", "table": "", "figure_caption": ""}]

    assert not RAGService._evidence_supports_query("What is today's stock price for Apple?", evidence, "explanation")


def test_common_themes_across_documents_is_classified_as_a_comparison():
    assert RAGService._question_intent("What common themes appear in both IPCC and NIST reports?") == "comparison"


def test_comparison_search_guidance_takes_priority_over_recommendation_guidance():
    planned = RAGService._intent_search_query(
        "Compare how the NIST and IPCC reports recommend responses to risk.",
        "contextual query",
    )

    assert "evidence for every named or selected source" in planned
    assert "Prefer structured recommendation tables" not in planned


def test_compound_question_plans_one_retrieval_query_per_explicit_aspect():
    planned = RAGService._compound_aspect_queries(
        "Explain risk management across governance, measurement, and ongoing monitoring."
    )

    assert planned == [
        "Explain risk management governance",
        "Explain risk management measurement evaluate measure evidence confidence likelihood severity",
        "Explain risk management ongoing monitoring actions guidance response strategies options mitigate mitigation adaptation reduce prevention",
    ]


def test_comparison_question_plans_coordinated_action_queries():
    planned = RAGService._compound_aspect_queries(
        "Compare how two reports identify, assess, and recommend responses to risk."
    )

    assert planned == [
        "identify risk characterize detect classification hazards exposure vulnerability impacts",
        "assess risk evaluate measure evidence confidence likelihood severity",
        "recommend responses to risk actions guidance response strategies options mitigate mitigation adaptation reduce prevention",
    ]


def test_compound_question_requires_evidence_for_every_explicit_aspect():
    query = "Explain risk management across governance, measurement, and ongoing monitoring."
    complete_evidence = [
        {
            "source": "nist.pdf",
            "text": "The framework covers organizational governance, risk measurement, and continuous monitoring.",
            "table": "",
            "figure_caption": "",
        }
    ]
    incomplete_evidence = [
        {
            "source": "nist.pdf",
            "text": "The framework covers organizational governance and risk measurement.",
            "table": "",
            "figure_caption": "",
        }
    ]

    assert RAGService._evidence_supports_query(query, complete_evidence, "explanation")
    assert not RAGService._evidence_supports_query(query, incomplete_evidence, "explanation")


def test_compound_answer_cites_each_requested_aspect_separately():
    evidence = [
        {
            "source": "nist.pdf",
            "page": 10,
            "text": "Governance establishes ownership and accountability for generative AI risks.",
            "table": "",
            "figure_caption": "",
            "equations": [],
        },
        {
            "source": "nist.pdf",
            "page": 20,
            "text": "Risk measurement evaluates system performance using documented metrics.",
            "table": "",
            "figure_caption": "",
            "equations": [],
        },
        {
            "source": "nist.pdf",
            "page": 30,
            "text": "Ongoing monitoring identifies changes in system risks after deployment.",
            "table": "",
            "figure_caption": "",
            "equations": [],
        },
    ]

    answer = RAGService._build_grounded_answer(
        "Explain risk management across governance, measurement, and ongoing monitoring.",
        evidence,
    )

    assert "**Governance**" in answer
    assert "**Measurement**" in answer
    assert "**Ongoing Monitoring**" in answer
    assert "[nist.pdf, p. 10]" in answer
    assert "[nist.pdf, p. 20]" in answer
    assert "[nist.pdf, p. 30]" in answer


def test_comparison_answer_is_balanced_and_does_not_use_table_recommendation_renderer():
    evidence = [
        {
            "source": "NIST.AI.600-1.pdf",
            "page": 42,
            "text": "NIST establishes processes to identify and monitor emerging generative AI risks.",
            "table": "| Action ID | Suggested action |\n| GV-1.2-001 | Establish transparency policies |",
            "figure_caption": "",
            "equations": [],
            "type": "table",
        },
        {
            "source": "IPCC_AR6_SYR_FullVolume.pdf",
            "page": 126,
            "text": "Effective climate governance enables mitigation and adaptation across policy domains and levels.",
            "table": "",
            "figure_caption": "",
            "equations": [],
            "type": "text",
        },
    ]

    answer = RAGService._build_grounded_answer(
        "Compare how NIST and IPCC recommend responses to risk.",
        evidence,
    )

    assert "**NIST.AI.600-1 approach**" in answer
    assert "**IPCC AR6 SYR FullVolume approach**" in answer
    assert "**Evidence-based contrast**" in answer
    assert "GOVERN 1.2 recommendations" not in answer


def test_math_evidence_is_requested_only_for_mathematical_information_needs():
    assert RAGService._query_requests_math_evidence("State and explain the Divergence Theorem.")
    assert RAGService._query_requests_math_evidence("What equation defines kinetic energy?")
    assert not RAGService._query_requests_math_evidence(
        "What governance principles appear in both NIST and IPCC reports?"
    )


def test_lexical_query_planning_adds_general_intent_terms_without_removing_the_question():
    query = "What impacts of climate change affect vulnerable communities?"

    planned = RAGService._lexical_retrieval_query(query, "explanation")

    assert planned.startswith(query)
    assert "losses damages affected" in planned


def test_common_themes_comparison_requires_and_accepts_evidence_from_both_documents():
    evidence = [
        {"source": "nist.pdf", "text": "NIST recommends risk management practices.", "table": "", "figure_caption": ""},
        {"source": "ipcc.pdf", "text": "The IPCC assesses climate risk management.", "table": "", "figure_caption": ""},
    ]

    assert RAGService._evidence_supports_query(
        "What common themes appear in both IPCC and NIST reports?",
        evidence,
        "comparison",
    )


def test_query_limits_normal_retrieval_to_requested_indexed_documents():
    allowed = RetrievalResult(
        chunk_id="allowed-1",
        score=0.4,
        text="The selected document contains supported evidence.",
        source="allowed.pdf",
        metadata={"page": 2},
    )

    class FakeEmbeddings:
        def encode_single(self, query):
            return query

    class FakeRetriever:
        def retrieve(self, query_embedding, top_k, sources=None):
            self.sources = sources
            return [allowed] if sources == {"allowed.pdf"} else []

    class FakeReranker:
        def rerank(self, query, candidates, top_k):
            return [(candidates[0], 0.9)] if candidates else []

    service = RAGService.__new__(RAGService)
    service.embedding_store = FakeEmbeddings()
    service.retriever = FakeRetriever()
    service.reranker = FakeReranker()
    service.documents = [{"filename": "allowed.pdf", "status": "indexed"}]
    service.session_history = {}
    service.storage_lock = RLock()

    response = service.query(QueryRequest(query="What is in the selected document?", document_names=["allowed.pdf"]))

    assert service.retriever.sources == {"allowed.pdf"}
    assert response["sources"][0]["source"] == "allowed.pdf"


def test_comparison_diversification_keeps_the_best_result_from_each_source():
    def result(source, page, score):
        item = RetrievalResult(
            chunk_id=f"{source}-{page}",
            score=score,
            text="Evidence",
            source=source,
            metadata={"page": page},
        )
        return ((item.text, item), score)

    reranked = [
        result("first.pdf", 1, 0.95),
        result("first.pdf", 2, 0.90),
        result("second.pdf", 4, 0.70),
        result("second.pdf", 5, 0.60),
    ]

    diversified = RAGService._diversify_comparison_reranking(reranked, {"first.pdf", "second.pdf"}, limit=3)

    assert [entry[0][1].source for entry in diversified[:2]] == ["first.pdf", "second.pdf"]
    assert len(diversified) == 3


def test_comparison_aspect_prioritization_covers_each_operation_and_source():
    def result(chunk_id, source, text, score):
        item = RetrievalResult(
            chunk_id=chunk_id,
            score=score,
            text=text,
            source=source,
            metadata={"page": 1},
        )
        return ((item.text, item), score)

    reranked = [
        result("generic-a", "a.pdf", "General risk management context.", 0.99),
        result("identify-a", "a.pdf", "Identify emerging risks.", 0.70),
        result("assess-a", "a.pdf", "Assess risk severity.", 0.69),
        result("respond-a", "a.pdf", "Recommend response actions for risk.", 0.68),
        result("generic-b", "b.pdf", "General risk management context.", 0.98),
        result("identify-b", "b.pdf", "Identify climate risks and hazards.", 0.67),
        result("assess-b", "b.pdf", "Assess climate risk.", 0.66),
        result("respond-b", "b.pdf", "Recommend response policies for climate risk.", 0.65),
    ]

    prioritized = RAGService._prioritize_comparison_aspects(
        reranked,
        {"a.pdf", "b.pdf"},
        "Compare how the reports identify, assess, and recommend responses to risk.",
        limit=6,
    )

    assert [entry[0][1].chunk_id for entry in prioritized] == [
        "identify-a",
        "identify-b",
        "assess-a",
        "assess-b",
        "respond-a",
        "respond-b",
    ]


def test_comparison_candidate_rejects_reference_fragments():
    reference = RetrievalResult(
        chunk_id="reference",
        score=0.5,
        text="Climate Change: The IPCC Response Strategies Report of Working Group III, 1990",
        source="ipcc.pdf",
        section="References",
    )
    finding = RetrievalResult(
        chunk_id="finding",
        score=0.5,
        text="The assessment evaluates interacting climate hazards, exposure, vulnerability, and response options.",
        source="ipcc.pdf",
        section="Risk assessment",
    )

    assert not RAGService._is_comparison_candidate(reference)
    assert RAGService._is_comparison_candidate(finding)


def test_hybrid_rerank_candidates_keep_dense_results_missing_from_fusion_head():
    fused = [
        RetrievalResult(chunk_id="lexical", score=0.8, text="lexical"),
        RetrievalResult(chunk_id="shared", score=0.7, text="shared"),
    ]
    dense = [
        RetrievalResult(chunk_id="dense", score=0.9, text="dense"),
        RetrievalResult(chunk_id="shared", score=0.7, text="shared"),
    ]

    candidates = RAGService._hybrid_rerank_candidates(fused, dense, fused_limit=2, dense_backfill_limit=2)

    assert [candidate.chunk_id for candidate in candidates] == ["lexical", "shared", "dense"]


def test_follow_up_queries_include_recent_history_for_retrieval():
    prepared = RAGService._prepare_query("What about that formula?", ["Explain the arc-length formula."])

    assert "Previous questions: Explain the arc-length formula." in prepared
    assert prepared.endswith("Follow-up question: What about that formula?")
