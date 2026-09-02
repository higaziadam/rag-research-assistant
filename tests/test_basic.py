from multimodal_rag.demo import build_demo_index


def test_demo_index_builds():
    retriever, encoder = build_demo_index()
    assert retriever.index.ntotal > 0
    assert encoder.model is not None
