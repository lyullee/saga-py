"""Small dependency-free vector index primitives used by the hybrid retriever.

The production retriever is deliberately able to run without downloading a
large embedding model.  These hashed character/term vectors are a sparse
semantic *fallback*: they catch spacing, inflection and OCR variants that a
strict FTS query misses.  The storage format is model-labelled so it can be
replaced later by a multilingual embedding provider without changing the
retrieval API.
"""

from __future__ import annotations

import hashlib
import math
import re
from array import array

from .text import normalize_text


VECTOR_DIM = 256
VECTOR_MODEL = "hash-ngram-v1"
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9_.·/-]*")


def _bucket(feature: str) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little", signed=False)
    return value % VECTOR_DIM, (1.0 if value & 1 else -1.0)


def _features(text: str) -> list[tuple[str, float]]:
    normalized = normalize_text(text).lower()
    tokens = TOKEN_RE.findall(normalized)
    features: list[tuple[str, float]] = []
    for token in tokens:
        features.append((f"w:{token}", 2.0))
        compact = re.sub(r"[^가-힣a-z0-9]", "", token)
        if len(compact) >= 2:
            for size in (2, 3):
                for index in range(0, len(compact) - size + 1):
                    features.append((f"c{size}:{compact[index:index + size]}", 0.7))
    compact_text = re.sub(r"\s+", "", normalized)
    if len(compact_text) >= 2:
        # A few document-level n-grams preserve phrase similarity when the PDF
        # extractor inserts/removes spaces between Korean syllables.
        for index in range(0, len(compact_text) - 2):
            features.append((f"p3:{compact_text[index:index + 3]}", 0.25))
    return features


def encode(text: str) -> bytes:
    values = [0.0] * VECTOR_DIM
    for feature, weight in _features(text):
        index, sign = _bucket(feature)
        values[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in values))
    if norm:
        values = [value / norm for value in values]
    return array("f", values).tobytes()


def decode(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return list(values)


def cosine(query: list[float], candidate: list[float]) -> float:
    if len(query) != len(candidate) or not query:
        return 0.0
    return sum(left * right for left, right in zip(query, candidate, strict=True))


def query_vector(text: str) -> list[float]:
    return decode(encode(text))
