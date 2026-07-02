"""
Per-turn controller. Ties the pieces together and exposes the *policy* seam so the
benchmark can swap allocation strategies (the experimental arms):

  - "full"          : always FULL (replay everything)  -> accuracy ceiling, token floor-buster
  - "recency"       : FULL for last-k items, POINTER otherwise
  - "reactive"/"reactive_afm" : AFM-style -- score by the CURRENT query only (drift=0)
  - "foveance"        : anticipatory -- score by the FUTURE-query posterior (drift>0)
  - "oracle"        : exact DP allocation on the foveance value curves (upper bound for the gap)

All arms share the SAME store, budget, model, predictor machinery and tasks, so any
difference isolates the policy. ``reactive_afm`` and ``foveance`` differ ONLY in the predictor's
drift (audited by the benchmark; see docs/NOVELTY.md).

Two-sided refinement: when the model emits a retrieve request (``RETRIEVE <item_id>``), the
controller re-inflates that item to FULL on the next turn via ``store.retrieve_full`` and logs
the re-inflation event. This is the deployable face of the successive-refinement theory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .store import MultiFidelityStore, Item, Fidelity, default_renderer, Renderer
from .predictor import AnticipatoryPredictor, PredictorConfig, FutureRelevancePredictor
from .embedders import HashingEmbedder
from .llm import LLM, Completion
from . import baselines

_RETRIEVE_RE = re.compile(r"RETRIEVE\s+([A-Za-z0-9_]+)")


@dataclass
class TurnRecord:
    turn: int
    query: str
    answer: str
    input_tokens: int
    output_tokens: int
    peak_tokens: int
    latency_s: float
    budget: int
    reinflations: int = 0


@dataclass
class RunResult:
    policy: str
    records: list = field(default_factory=list)

    @property
    def total_input(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def peak(self) -> int:
        return max((r.peak_tokens for r in self.records), default=0)

    @property
    def wall_s(self) -> float:
        return sum(r.latency_s for r in self.records)

    @property
    def total_reinflations(self) -> int:
        return sum(r.reinflations for r in self.records)


class Controller:
    """Drives one policy arm over a trajectory against one model."""

    def __init__(
        self,
        llm: LLM,
        budget: int,
        policy: str = "foveance",
        recency_k: int = 4,
        drift: float = 0.6,
        token_counter: Optional[Callable[[str], int]] = None,
        renderer: Renderer = default_renderer,
        embedder: object = None,
        future_model: Optional[FutureRelevancePredictor] = None,
        kind_prior: Optional[dict] = None,
        retrieve_enabled: bool = True,
        fidelity_cost: float = 0.0,
        system: str = (
            "You are a careful agent. Use only the provided context. When asked to recall a "
            "key such as 'recall k3', reply with that key's exact value from a line of the form "
            "'FACT k3=<value>'. Answer with just the value. If it is not present, reply UNKNOWN."
        ),
    ) -> None:
        self.llm = llm
        self.budget = budget
        self.policy = policy
        self.recency_k = recency_k
        self.retrieve_enabled = retrieve_enabled
        self.fidelity_cost = fidelity_cost
        self.system = system
        self._prev_levels: dict = {}
        self.store = MultiFidelityStore(renderer, token_counter)
        is_reactive = policy in ("reactive", "reactive_afm")
        cfg = PredictorConfig(drift=(0.0 if is_reactive else drift), kind_prior=kind_prior)
        self.pred = AnticipatoryPredictor(
            self.store, embedder or HashingEmbedder(), config=cfg, future_model=future_model
        )
        self._pending_reinflate: set[str] = set()

    def add_item(self, item: Item) -> None:
        self.store.add(item)

    # --- policies -> per-item fidelity assignment ---
    def _levels(self, turn: int) -> dict:
        if self.policy == "recency":
            levels = baselines.recency(self.store, self.pred, self.budget, turn, k=self.recency_k)
        else:
            fn = baselines.POLICIES.get(self.policy)
            if fn is None:
                raise ValueError(f"unknown policy {self.policy!r}")
            levels = fn(self.store, self.pred, self.budget, turn)
        # Two-sided refinement: anything explicitly retrieved last turn is forced to FULL.
        for iid in self._pending_reinflate:
            if iid in levels:
                levels[iid] = Fidelity.FULL
        return levels

    def _handle_retrieve(self, answer: str, turn: int) -> int:
        """Parse retrieve requests, re-inflate referenced items next turn. Returns count."""
        if not self.retrieve_enabled:
            return 0
        self._pending_reinflate = set()
        for iid in _RETRIEVE_RE.findall(answer or ""):
            if iid in self.store.items:
                self.store.retrieve_full(iid, turn)
                self._pending_reinflate.add(iid)
        return len(self._pending_reinflate)

    def _refidelity_cost(self, levels: dict) -> int:
        """Re-render/re-fetch tokens charged when an item's fidelity is *raised* vs last turn.

        Models the suite's fidelity-change cost (paper Thm "locality gap", re-fetch penalty
        eta > 0): anticipation that pre-stages soon-needed items pays fewer raises. Charged
        identically to every arm, so it never advantages foveance by construction.
        """
        if self.fidelity_cost <= 0.0:
            return 0
        extra = 0.0
        for iid, lvl in levels.items():
            prev = self._prev_levels.get(iid, Fidelity.POINTER)
            if lvl > prev:
                extra += self.fidelity_cost * (self.store.cost(iid, lvl) - self.store.cost(iid, prev))
        return int(extra)

    def step(self, query: str, turn: int, probe: Optional[str] = None) -> TurnRecord:
        # The predictor observes ``probe`` (defaults to the query). When the suite hides the
        # target from the query (name_target=False), the probe carries no lexical key, forcing
        # reliance on the anticipatory posterior rather than current-query lexical match.
        self.pred.observe_query(probe if probe is not None else query)
        levels = self._levels(turn)
        ctx, ntok = self.store.assemble(levels, system=self.system)
        comp: Completion = self.llm.generate(ctx, query)
        reinf = self._handle_retrieve(comp.text, turn)
        refid = self._refidelity_cost(levels)
        self._prev_levels = dict(levels)
        return TurnRecord(
            turn=turn, query=query, answer=comp.text,
            input_tokens=comp.input_tokens + refid, output_tokens=comp.output_tokens,
            peak_tokens=ntok, latency_s=comp.latency_s, budget=self.budget,
            reinflations=reinf,
        )

    def run(self, turns) -> RunResult:
        """Convenience: run a sequence of objects with ``.new_items`` and ``.query``."""
        result = RunResult(policy=self.policy)
        for t, turn in enumerate(turns):
            for it in getattr(turn, "new_items", []):
                self.add_item(it)
            result.records.append(self.step(turn.query, t, probe=getattr(turn, "probe", None)))
        return result
