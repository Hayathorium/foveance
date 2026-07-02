"""
Policy arms / baselines, all sharing one signature so the controller and benchmark can swap
them freely:

    policy(store, predictor, budget, turn) -> dict[item_id, Fidelity]

Arms
----
* ``full``         -- every item at FULL (accuracy ceiling, token cost ceiling).
* ``recency``      -- FULL for the last ``k`` items, POINTER otherwise (cheap, myopic).
* ``truncate``     -- budget-aware recency: newest items FULL until the budget is spent, rest
  POINTER (a sliding context window; relevance-blind).
* ``uniform``      -- every item at the single highest fidelity tier that fits the budget
  (relevance-blind; isolates the value of *allocating* fidelity by relevance).
* ``reactive_afm`` -- faithful AFM-style baseline: score by the *current* query
  (predictor with ``drift = 0``) + half-life recency + kind importance, then pack under the
  budget with the index allocator. This is the ``drift = 0`` special case of Foveance.
* ``foveance``       -- anticipatory: identical machinery to ``reactive_afm`` but the predictor's
  ``drift > 0`` so it scores by the *future*-query posterior. The ONLY difference between
  ``reactive_afm`` and ``foveance`` is the predictor's drift (docs/NOVELTY.md; audited by the benchmark).
* ``oracle``       -- exact DP allocation on the foveance value curves (greedy-gap upper bound).
* ``noop``         -- no compression (== full); a control.
* ``llmlingua2``   -- optional LLMLingua-2 prompt-compression wrapper (``[bench]`` + ``llmlingua``).

NOTE ON NOVELTY: ``reactive_afm`` is shipped as a first-class arm precisely because the
multi-fidelity-store-under-budget mechanism is prior art (AFM). The package must *contain* the
comparison, not merely assert it.
"""
from __future__ import annotations

from typing import Callable

from .store import Fidelity, MultiFidelityStore
from .allocator import index_allocate, dp_allocate, lp_bound

# A policy maps (store, predictor, budget, turn) -> {item_id: Fidelity}.
Policy = Callable[[MultiFidelityStore, object, int, int], dict]


def _index_levels(store: MultiFidelityStore, predictor, budget: int, turn: int) -> dict:
    """Budgeted index (Lagrangian) allocation on the predictor's value curves."""
    ids = store.order
    if not ids:
        return {}
    cost_fn = lambda iid, lv: store.cost(iid, lv)
    vc = lambda iid: predictor.value_curve(iid, turn)
    levels, _, _ = index_allocate(ids, vc, cost_fn, budget)
    return levels


def full(store: MultiFidelityStore, predictor, budget: int, turn: int) -> dict:
    return {i: Fidelity.FULL for i in store.order}


def noop(store: MultiFidelityStore, predictor, budget: int, turn: int) -> dict:
    """No compression: pass everything through verbatim (a control == full)."""
    return full(store, predictor, budget, turn)


def recency(store: MultiFidelityStore, predictor, budget: int, turn: int, k: int = 4) -> dict:
    keep = set(store.order[-k:])
    return {i: (Fidelity.FULL if i in keep else Fidelity.POINTER) for i in store.order}


def truncate(store: MultiFidelityStore, predictor, budget: int, turn: int) -> dict:
    """Budget-aware recency: keep the newest items verbatim (FULL) until the budget is spent, then
    drop the rest to POINTER. A common, relevance-blind baseline (sliding context window)."""
    levels = {i: Fidelity.POINTER for i in store.order}
    spent = sum(store.cost(i, Fidelity.POINTER) for i in store.order)
    for iid in reversed(store.order):  # newest first
        extra = store.cost(iid, Fidelity.FULL) - store.cost(iid, Fidelity.POINTER)
        if spent + extra <= budget:
            levels[iid] = Fidelity.FULL
            spent += extra
    return levels


def uniform(store: MultiFidelityStore, predictor, budget: int, turn: int) -> dict:
    """Relevance-blind uniform allocation: put every item at the single highest fidelity tier that
    collectively fits the budget. Isolates the value of allocating fidelity by relevance rather than
    spreading it evenly."""
    ids = store.order
    if not ids:
        return {}
    chosen = Fidelity.POINTER
    for lvl in (Fidelity.FULL, Fidelity.DIGEST, Fidelity.GIST):
        if sum(store.cost(i, lvl) for i in ids) <= budget:
            chosen = lvl
            break
    return {i: chosen for i in ids}


def reactive_afm(store: MultiFidelityStore, predictor, budget: int, turn: int) -> dict:
    """AFM-style reactive packing. Requires ``predictor.cfg.drift == 0`` (current-query)."""
    return _index_levels(store, predictor, budget, turn)


def foveance(store: MultiFidelityStore, predictor, budget: int, turn: int) -> dict:
    """Anticipatory packing. Same code path as ``reactive_afm``; predictor drift > 0."""
    return _index_levels(store, predictor, budget, turn)


def oracle(store: MultiFidelityStore, predictor, budget: int, turn: int, scale: int = 4) -> dict:
    ids = store.order
    if not ids:
        return {}
    cost_fn = lambda iid, lv: store.cost(iid, lv)
    vc = lambda iid: predictor.value_curve(iid, turn)
    levels, _, _ = dp_allocate(ids, vc, cost_fn, budget, scale=scale)
    return levels


def lp_value(store: MultiFidelityStore, predictor, budget: int, turn: int) -> float:
    """The LP upper bound on achievable value (not a renderable policy -- a frontier point)."""
    ids = store.order
    if not ids:
        return 0.0
    cost_fn = lambda iid, lv: store.cost(iid, lv)
    vc = lambda iid: predictor.value_curve(iid, turn)
    return lp_bound(ids, vc, cost_fn, budget)


# ------------------------------------------------------------------- optional: LLMLingua-2
def llmlingua2_available() -> bool:
    try:
        import llmlingua  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def llmlingua2(store: MultiFidelityStore, predictor, budget: int, turn: int) -> dict:  # pragma: no cover
    """Compress each item with LLMLingua-2 to fit the budget. Skipped if dep absent."""
    if not llmlingua2_available():
        raise RuntimeError("llmlingua not installed; skip this arm (pip install llmlingua)")
    # Treated as FULL items whose renderer is LLMLingua-2 (wired by the harness renderer seam).
    return {i: Fidelity.FULL for i in store.order}


#: Name -> policy, for the controller/benchmark to look up arms by string.
POLICIES: dict[str, Policy] = {
    "full": full,
    "noop": noop,
    "recency": recency,
    "truncate": truncate,
    "uniform": uniform,
    "reactive": reactive_afm,      # alias kept for the seed's arm name
    "reactive_afm": reactive_afm,
    "foveance": foveance,
    "oracle": oracle,
    "llmlingua2": llmlingua2,
}

#: Arms whose only difference must be the predictor drift (audited by the benchmark, GATE 2).
DRIFT_TWINS = ("reactive_afm", "foveance")
