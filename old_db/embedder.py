from __future__ import annotations
from typing import Protocol, Sequence


class Embedder(Protocol):
    """
    Minimal interface required by MemoryStore.

    Keeping this as a protocol means the store does not care whether
    embeddings come from sentence-transformers, llama.cpp, an API, etc.
    """

    dimension: int

    def embed(self, text: str) -> Sequence[float]:
        ...
        
class FakeEmbedder:
    """
    Tiny deterministic embedder.

    This is NOT intended for real semantic retrieval.

    It exists so the entire MemoryStore can be tested without downloading
    a model or contacting an external service.
    """

    dimension = 8

    def embed(self, text: str) -> Sequence[float]:
        vector = [0.0] * self.dimension

        for index, character in enumerate(text.encode("utf-8")):
            vector[index % self.dimension] += character

        magnitude = sum(value * value for value in vector) ** 0.5

        if magnitude == 0:
            return vector

        return [
            value / magnitude
            for value in vector
        ]