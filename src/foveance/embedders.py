"""
Embedders: text -> dense vector, behind a single ``Embedder`` protocol.

The offline default (``HashingEmbedder``) is dependency-free and deterministic so the
core library and CI never need a network or heavyweight ML stack. Production swaps in a
real sentence embedder (``SentenceTransformerEmbedder``, ``[ml]`` extra) or an API embedder
(``APIEmbedder``, OpenAI-compatible) without touching the predictor/allocator.

NOTE ON NOVELTY / PRIOR ART (read docs/NOVELTY.md): the embedder is plumbing, not a
contribution. Any relevance signal works; Foveance's claim is about *what* relevance we
score against (the future), not how the vectors are produced.
"""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable, Sequence


@runtime_checkable
class Embedder(Protocol):
    """A text -> vector encoder. ``embed`` must be deterministic for the offline path."""

    def embed(self, text: str) -> list[float]:  # pragma: no cover - protocol
        ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is empty)."""
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return num / (na * nb)


class HashingEmbedder:
    """Deterministic, dependency-free bag-of-tokens hashing embedder for offline runs.

    Uses a stable token hash (not Python's randomized ``hash``) so vectors are identical
    across processes and runs -- essential for a reproducible, seeded benchmark.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _tok_hash(self, tok: str) -> int:
        h = 1469598103934665603  # FNV-1a 64-bit offset basis
        for ch in tok.encode("utf-8"):
            h ^= ch
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h % self.dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            vec[self._tok_hash(tok)] += 1.0
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]


class SentenceTransformerEmbedder:
    """Real sentence embeddings via ``sentence-transformers`` (``[ml]`` extra)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:  # pragma: no cover - needs heavy dep
        return [float(x) for x in self.model.encode(text, normalize_embeddings=True)]


class APIEmbedder:
    """OpenAI-compatible embeddings endpoint (OpenAI, vLLM, TEI, ...)."""

    def __init__(self, model: str, base_url: str, api_key: str = "x") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def embed(self, text: str) -> list[float]:  # pragma: no cover - needs network
        import json
        import urllib.request

        body = json.dumps({"model": self.model, "input": text}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())
        return [float(x) for x in data["data"][0]["embedding"]]


def as_embed_fn(embedder: "Embedder | object"):
    """Adapt an ``Embedder`` (object with ``.embed``) or a bare callable to ``str -> vec``."""
    if hasattr(embedder, "embed"):
        return embedder.embed  # type: ignore[attr-defined]
    if callable(embedder):
        return embedder
    raise TypeError("embedder must be an Embedder or a callable str->list[float]")
