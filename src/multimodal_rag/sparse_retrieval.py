"""Local BM25 retrieval over the same chunks stored in the FAISS index."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from .data_models import DocumentChunk, RetrievalResult


# Generic question words and numeric fragments are poor BM25 signals. In
# particular, a rare decimal such as ``1.2`` can otherwise outrank a report
# identifier such as ``GOVERN`` because it appears in a short mathematics
# chunk. Keep domain terms, acronyms, and mixed alphanumeric identifiers.
_QUERY_STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "please",
        "the",
        "this",
        "to",
        "what",
        "when",
        "which",
        "with",
    }
)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    top_k: int,
    rank_constant: int = 60,
) -> List[RetrievalResult]:
    """Merge ranked candidate lists without comparing incompatible score scales."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if rank_constant < 1:
        raise ValueError("rank_constant must be at least 1")

    fused_scores: Dict[str, float] = defaultdict(float)
    first_rank: Dict[str, int] = {}
    by_chunk_id: Dict[str, RetrievalResult] = {}
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            fused_scores[result.chunk_id] += 1 / (rank_constant + rank)
            first_rank.setdefault(result.chunk_id, rank)
            by_chunk_id.setdefault(result.chunk_id, result)

    ranked_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (fused_scores[chunk_id], -first_rank[chunk_id]),
        reverse=True,
    )[:top_k]
    return [by_chunk_id[chunk_id] for chunk_id in ranked_ids]


class BM25Retriever:
    """In-memory inverted index for exact-term and phrase-adjacent retrieval.

    The index is derived from persisted chunk metadata at service startup and
    rebuilt after a document mutation. It intentionally stores term postings,
    not a second copy of document text or embeddings.
    """

    def __init__(self, chunks: Sequence[DocumentChunk], k1: float = 1.5, b: float = 0.75):
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.document_lengths: List[int] = []
        self.average_document_length = 0.0
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.source_indices: Dict[str, List[int]] = defaultdict(list)
        self._build()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Return discriminative lexical terms for BM25 matching.

        Purely numeric tokens are intentionally excluded. Dense retrieval and
        the cross-encoder still see the full passage, while BM25 avoids
        promoting unrelated equations solely because they contain a rare
        number from the user's question.
        """
        terms = re.findall(r"[\w]+(?:[-./][\w]+)*", text.casefold(), flags=re.UNICODE)
        return [
            term
            for term in terms
            if term not in _QUERY_STOP_TERMS and any(character.isalpha() for character in term)
        ]

    def _build(self) -> None:
        for index, chunk in enumerate(self.chunks):
            terms = self.tokenize(chunk.to_text_for_search())
            self.document_lengths.append(len(terms))
            self.source_indices[chunk.source].append(index)
            for term, frequency in Counter(terms).items():
                self.postings[term].append((index, frequency))
        if self.document_lengths:
            self.average_document_length = sum(self.document_lengths) / len(self.document_lengths)

    def retrieve(self, query: str, top_k: int = 5, sources: set[str] | None = None) -> List[RetrievalResult]:
        """Return source-filtered BM25 results for a query."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if sources is not None and not sources:
            return []
        if not self.chunks:
            return []

        allowed_indices = None
        if sources is not None:
            allowed_indices = {index for source in sources for index in self.source_indices.get(source, [])}
            if not allowed_indices:
                return []

        scores: Dict[int, float] = defaultdict(float)
        document_count = len(self.chunks)
        for term in set(self.tokenize(query)):
            posting = self.postings.get(term, [])
            if not posting:
                continue
            document_frequency = len(posting)
            inverse_document_frequency = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            for index, term_frequency in posting:
                if allowed_indices is not None and index not in allowed_indices:
                    continue
                document_length = self.document_lengths[index]
                normalization = self.k1 * (1 - self.b + self.b * document_length / max(self.average_document_length, 1.0))
                scores[index] += inverse_document_frequency * (term_frequency * (self.k1 + 1)) / (term_frequency + normalization)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [self._result(index, score) for index, score in ranked]

    def retrieve_diversified(
        self,
        query: str,
        sources: set[str],
        per_source_k: int,
    ) -> List[RetrievalResult]:
        """Interleave source-local BM25 results for explicit comparisons."""
        if per_source_k < 1:
            raise ValueError("per_source_k must be at least 1")
        source_results = {
            source: self.retrieve(query, top_k=per_source_k, sources={source})
            for source in sorted(sources)
        }
        diversified: List[RetrievalResult] = []
        for rank in range(per_source_k):
            for source in sorted(source_results):
                results = source_results[source]
                if rank < len(results):
                    diversified.append(results[rank])
        return diversified

    def _result(self, index: int, score: float) -> RetrievalResult:
        chunk = self.chunks[index]
        return RetrievalResult(
            chunk_id=chunk.chunk_id,
            score=float(score),
            text=chunk.text,
            table=chunk.table,
            figure_caption=chunk.figure_caption,
            source=chunk.source,
            section=chunk.section,
            metadata=chunk.metadata,
            equations=chunk.equations,
            type=chunk.type,
        )
