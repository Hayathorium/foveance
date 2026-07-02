"""Property-based tests (Hypothesis): the allocator theorems must hold on random instances.

These back the paper's claims empirically: budget feasibility, monotonicity in budget, the
one-item greedy bound (Thm. index-policy near-optimality), and the IDX <= OPT <= LP sandwich.
"""
from hypothesis import given, settings, strategies as st

from foveance import Fidelity
from foveance.allocator import index_allocate, dp_allocate, lp_bound

LEVELS = list(Fidelity)


@st.composite
def instances(draw, max_items=8):
    """Random MCKP instances with nondecreasing cost ladders and bounded values."""
    n = draw(st.integers(min_value=1, max_value=max_items))
    ids = [f"i{j}" for j in range(n)]
    base = {i: draw(st.floats(min_value=0.05, max_value=5.0)) for i in ids}
    ymul = (0.0, 0.45, 0.8, 1.0)
    cost = {}
    for i in ids:
        c0 = draw(st.integers(min_value=1, max_value=4))
        steps = [draw(st.integers(min_value=1, max_value=8)) for _ in range(3)]
        ladder = [c0]
        for s in steps:
            ladder.append(ladder[-1] + s)   # strictly increasing cost with fidelity
        for lv, c in zip(LEVELS, ladder):
            cost[(i, lv)] = c
    vc = lambda i: [base[i] * y for y in ymul]      # noqa: E731
    cf = lambda i, lv: cost[(i, lv)]                # noqa: E731
    # Budget must be at least the all-POINTER floor (the minimum-cost feasible assignment);
    # below it the instance is infeasible and no policy can respect the budget.
    floor = sum(cost[(i, Fidelity.POINTER)] for i in ids)
    full = sum(cost[(i, Fidelity.FULL)] for i in ids)
    budget = draw(st.integers(min_value=floor, max_value=full + 1))
    return ids, vc, cf, budget


@given(instances())
@settings(max_examples=300, deadline=None)
def test_budget_respected_and_levels_valid(inst):
    ids, vc, cf, budget = inst
    levels, value, spent = index_allocate(ids, vc, cf, budget)
    assert spent <= budget
    assert set(levels) == set(ids)
    assert all(isinstance(v, Fidelity) for v in levels.values())
    assert value >= -1e-9


@given(instances())
@settings(max_examples=200, deadline=None)
def test_monotone_in_budget(inst):
    ids, vc, cf, budget = inst
    _, v_lo, _ = index_allocate(ids, vc, cf, budget)
    _, v_hi, _ = index_allocate(ids, vc, cf, budget * 3)
    assert v_hi >= v_lo - 1e-9


@given(instances())
@settings(max_examples=200, deadline=None)
def test_greedy_one_item_bound_and_lp_sandwich(inst):
    ids, vc, cf, budget = inst
    _, v_idx, _ = index_allocate(ids, vc, cf, budget)
    _, v_dp, _ = dp_allocate(ids, vc, cf, budget, scale=1)
    v_lp = lp_bound(ids, vc, cf, budget)
    max_item = max(vc(i)[Fidelity.FULL] - vc(i)[Fidelity.POINTER] for i in ids)
    assert v_dp - v_idx <= max_item + 1e-6        # Thm: within one item's value of optimum
    assert v_idx <= v_dp + 1e-6                   # IDX <= OPT
    assert v_dp <= v_lp + 1e-6                    # OPT <= LP
