"""
Anticipatory future-relevance predictor.

THIS IS THE CONCEPTUAL DELTA vs prior art.

  - AFM (Cruz 2025) scores each item by similarity to the *current* query q_t.
  - Foveance scores each item by its expected relevance to the *future* of the trajectory:
    a posterior over what later steps will need, E_{q ~ p(future | history)}[ relevance(item, q) ].

We approximate p(future | history) cheaply and model-agnostically as a forward-drifting
mixture of (a) the current query and (b) a momentum/drift term over recent queries, plus a
recurrence prior (items referenced before tend to be referenced again -- "needles"). A learned
predictor (learned.py) drops in behind the ``FutureRelevancePredictor`` interface to replace
the cosine-similarity relevance with a calibrated probability.

The predictor returns, per item, a *value curve* v_i(level): the expected marginal task-value
of holding item i at each fidelity. The allocator (allocator.py) then solves a budgeted
multiple-choice knapsack over these curves.

``drift`` controls anticipation strength; ``drift = 0`` recovers the reactive (AFM-like)
criterion exactly -- this is the first-class baseline required by docs/NOVELTY.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

from .store import Item, MultiFidelityStore
from .embedders import HashingEmbedder, as_embed_fn, cosine

# Back-compat: an embedder may be a bare callable str->vec or an Embedder object.
Embedder = Callable[[str], list]


def _hash_embed(text: str, dim: int = 256) -> list:
    """Deterministic, dependency-free hashing embedder (kept for back-compat)."""
    return HashingEmbedder(dim).embed(text)


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return cosine(a, b)


class FutureRelevancePredictor(Protocol):
    """Pluggable estimator of an item's expected future relevance in [0, 1].

    ``learned.py`` implements this so a trained model can replace the cosine relevance
    while reusing the same value-curve / allocator machinery.
    """

    def predict(self, item: Item, history: "PredictorContext") -> float:  # pragma: no cover
        ...


@dataclass
class PredictorContext:
    """Snapshot passed to a learned relevance model: enough to compute its features."""

    turn: int
    future_posterior: list
    query_history: list
    cfg: "PredictorConfig"


@dataclass
class PredictorConfig:
    half_life: float = 12.0        # recency half-life in turns (AFM default for fair comparison)
    drift: float = 0.6             # weight on query momentum (anticipation); 0 -> reactive (AFM)
    recurrence_prior: float = 0.5  # bonus for items referenced before
    recency_weight: float = 0.25   # weight on recency term in base value
    kind_prior: Optional[dict] = None  # e.g. {"error": 1.5, "decision": 1.4, "tool_output": 1.0}
    # fidelity "information yield": fraction of an item's value realized at each level
    yield_by_level: tuple = (0.0, 0.45, 0.8, 1.0)  # POINTER, GIST, DIGEST, FULL


class AnticipatoryPredictor:
    """Scores items by expected *future* relevance and emits per-item value curves."""

    def __init__(
        self,
        store: MultiFidelityStore,
        embedder: "Embedder | object" = HashingEmbedder(),
        config: Optional[PredictorConfig] = None,
        future_model: Optional[FutureRelevancePredictor] = None,
    ) -> None:
        self.store = store
        self.embed = as_embed_fn(embedder)
        self.cfg = config or PredictorConfig()
        self.future_model = future_model
        self._query_history: list[list] = []   # embeddings of past queries

    # -------------------------------------------------------------- query posterior
    def observe_query(self, query_text: str) -> None:
        """Record a query; updates the forward-drifting future posterior."""
        self._query_history.append(self.embed(query_text))

    def _future_query_posterior(self) -> list:
        """Forward-drifting estimate of the embedding of upcoming information needs:
        current query + drift * momentum (current - previous)."""
        if not self._query_history:
            return []
        cur = self._query_history[-1]
        if len(self._query_history) == 1 or self.cfg.drift == 0.0:
            return cur
        prev = self._query_history[-2]
        fut = [c + self.cfg.drift * (c - p) for c, p in zip(cur, prev)]
        n = math.sqrt(sum(x * x for x in fut)) or 1.0
        return [x / n for x in fut]

    def posterior_debug(self) -> dict:
        """Introspection hook for analysis/tests: the current posterior and its provenance."""
        fut = self._future_query_posterior()
        return {
            "n_queries": len(self._query_history),
            "drift": self.cfg.drift,
            "posterior_norm": math.sqrt(sum(x * x for x in fut)) if fut else 0.0,
            "is_reactive": self.cfg.drift == 0.0,
            "posterior": fut,
        }

    def _context(self, turn: int) -> PredictorContext:
        return PredictorContext(turn=turn, future_posterior=self._future_query_posterior(),
                                query_history=self._query_history, cfg=self.cfg)

    # ----------------------------------------------------------------- value model
    def base_value(self, item: Item, turn: int) -> float:
        """Expected future relevance of ``item`` (independent of fidelity), >= 0."""
        if self.future_model is not None:
            sim = max(0.0, float(self.future_model.predict(item, self._context(turn))))
        else:
            if item.embedding is None:
                item.embedding = self.embed(item.full_text)
            fut = self._future_query_posterior()
            sim = max(0.0, _cos(item.embedding, fut) if fut else 0.0)

        age = turn - max(item.created_turn, item.last_referenced_turn)
        recency = 0.5 ** (age / self.cfg.half_life)
        recur = self.cfg.recurrence_prior if item.last_referenced_turn >= 0 else 0.0
        kind_w = (self.cfg.kind_prior or {}).get(item.kind, 1.0)
        return kind_w * (sim + self.cfg.recency_weight * recency + recur)

    def value_curve(self, item_id: str, turn: int) -> list:
        """v_i(level) for level in Fidelity: expected task-value at each fidelity."""
        item = self.store.items[item_id]
        bv = self.base_value(item, turn)
        return [bv * y for y in self.cfg.yield_by_level]
