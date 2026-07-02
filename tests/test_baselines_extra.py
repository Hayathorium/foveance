"""Tests for the extra allocation baselines (truncate, uniform) added for the broader comparison."""
from foveance import baselines
from foveance.store import Fidelity, Item, MultiFidelityStore


def _store(n=8, lines=30):
    st = MultiFidelityStore()
    for t in range(n):
        body = "FACT k=v\n" + "\n".join(f"log {i} status=ok" for i in range(lines))
        st.add(Item(f"o{t}", "tool_output", body, created_turn=t))
    return st


def _cost(store, levels):
    return sum(store.cost(i, levels[i]) for i in store.order)


def test_truncate_respects_budget_and_prefers_recent():
    st = _store()
    B = 400
    levels = baselines.truncate(st, None, B, turn=8)
    assert set(levels) == set(st.order)
    assert all(v in (Fidelity.POINTER, Fidelity.FULL) for v in levels.values())
    assert _cost(st, levels) <= B
    # among the FULL items, the newest item is kept before older ones
    full_ids = [i for i in st.order if levels[i] == Fidelity.FULL]
    if full_ids:
        assert st.order[-1] in full_ids


def test_uniform_is_single_tier_and_within_budget():
    st = _store()
    B = 900
    levels = baselines.uniform(st, None, B, turn=8)
    tiers = set(levels.values())
    assert len(tiers) == 1                       # one fidelity for everyone (relevance-blind)
    assert _cost(st, levels) <= B or tiers == {Fidelity.POINTER}


def test_new_arms_registered():
    for name in ("truncate", "uniform"):
        assert name in baselines.POLICIES
