from __future__ import annotations

from .config import settings
from .data_models import DocumentChunk
from .embeddings import EmbeddingStore
from .retrieval import FAISSRetriever


def build_demo_index():
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            text="The retrieval system uses dense embeddings and a vector index to find relevant evidence from technical literature.",
            table="| Strategy | Recall@5 |\n| --- | ---:|\n| Dense only | 0.58 |",
            figure_caption="Figure 1: Dense retrieval recall by document section.",
            source="research_notes.pdf",
            section="System design",
            metadata={"page": 1},
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            text="Cross-encoder reranking improves ranking quality by scoring query-passage pairs with a transformer model.",
            table="| Model | Recall@5 |\n| --- | ---:|\n| Cross-encoder | 0.82 |",
            figure_caption="Figure 2: Reranker gains on QA tasks.",
            source="research_notes.pdf",
            section="Evaluation",
            metadata={"page": 3},
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            text="LoRA fine-tuning is a parameter-efficient strategy for adapting a general purpose reranker to scientific document QA.",
            table="| Tuning | Latency |\n| --- | ---:|\n| LoRA | 370 ms |",
            figure_caption="Figure 3: PEFT trade-off between quality and latency.",
            source="research_notes.pdf",
            section="Fine-tuning",
            metadata={"page": 6},
        ),
    ]
    encoder = EmbeddingStore(settings.model_name)
    embeddings = encoder.encode([chunk.to_text_for_search() for chunk in chunks])
    retriever = FAISSRetriever(embedding_dim=embeddings.shape[1])
    retriever.add_chunks(chunks, embeddings)
    return retriever, encoder


def main():
    retriever, encoder = build_demo_index()
    query = "How does the reranker improve document retrieval quality?"
    q_embedding = encoder.encode_single(query)
    results = retriever.retrieve(q_embedding, top_k=3)
    print("Query:", query)
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result.chunk_id} | score={result.score:.4f} | section={result.section}")
        print(result.text)
        print("---")


if __name__ == "__main__":
    main()
