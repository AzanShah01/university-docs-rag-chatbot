"""Small in-memory FAISS vector store."""

from __future__ import annotations

import faiss
import numpy as np

from src.config import TOP_K


class FAISSVectorStore:
    """Store normalized embeddings and retrieve chunks by cosine similarity."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """Build an inner-product index from chunks and their embeddings."""
        if not chunks:
            raise ValueError("No chunks were provided for indexing.")
        normalized = self._normalize(embeddings)
        if normalized.shape[0] != len(chunks):
            raise ValueError("The number of embeddings must match the number of chunks.")

        self.index = faiss.IndexFlatIP(normalized.shape[1])
        self.index.add(normalized)
        self.chunks = list(chunks)

    @property
    def is_ready(self) -> bool:
        """Return whether the index contains searchable chunks."""
        return self.index is not None and bool(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K) -> list[dict]:
        """Return the most similar chunks, including normalized similarity scores."""
        if not self.is_ready:
            return []

        query = self._normalize(query_embedding)
        result_count = min(max(1, top_k), len(self.chunks))
        scores, indices = self.index.search(query, result_count)

        results: list[dict] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            result = dict(self.chunks[int(index)])
            result["score"] = float(score)
            results.append(result)
        return results
