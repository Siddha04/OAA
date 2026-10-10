from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


class EmbeddingModel(Protocol):
    """Interface for replaceable local or pretrained embedding backends."""

    def embed(self, text: str) -> tuple[float, ...]:
        """Return one finite vector for the supplied text."""
        ...


class HashEmbeddingModel:
    """Deterministic lexical feature hashing; not a semantic pretrained model.

    Unigrams and adjacent-token bigrams are hashed into a fixed-width vector,
    then L2-normalized. The backend has no model downloads or network access.
    """

    def __init__(self, dimensions: int = 512) -> None:
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens = _TOKEN_PATTERN.findall(text.casefold())
        vector = [0.0] * self.dimensions
        features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
        features.extend(
            (f"bigram:{left}_{right}", 0.5)
            for left, right in zip(tokens, tokens[1:])
        )
        for feature, weight in features:
            digest = hashlib.blake2b(
                feature.encode("utf-8"), digest_size=8, person=b"OAA-RAG1"
            ).digest()
            index = int.from_bytes(digest, "little") % self.dimensions
            sign = 1.0 if digest[0] & 1 else -1.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return tuple(vector)
