import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foveance import Fidelity
from foveance.allocator import index_allocate, dp_allocate, lp_bound, _concave_envelope


def _toy(n=8, seed=0):
    import random
    rng = random.Random(seed)
    ids = [f"i{j}" for j in range(n)]
    base = {i: rng.uniform(0.1, 2.0) for i in ids}
    # value curve: concave-ish yield by level; cost grows with level
    ymul = (0.0, 0.45, 0.8, 1.0)
    cost = {(i, lv): int(2 + 6 * int(lv) + rng.randint(0, 4)) for i in ids for lv in Fidelity}
    vc = lambda i: [base[i] * y for y in ymul]
    cf = lambda i, lv: cost[(i, lv)]
    return ids, vc, cf


def test_index_respects_budget():
    ids, vc, cf = _toy()
    budget = 60
    levels, val, spent = index_allocate(ids, vc, cf, budget)
    assert spent <= budget
    assert set(levels) == set(ids)
    assert all(isinstance(v, Fidelity) for v in levels.values())


def test_index_monotone_in_budget():
    ids, vc, cf = _toy(seed=3)
    _, v_lo, _ = index_allocate(ids, vc, cf, 40)
    _, v_hi, _ = index_allocate(ids, vc, cf, 120)
    assert v_hi >= v_lo - 1e-9  # more budget never hurts


def test_greedy_gap_small_vs_oracle():
    # The paper's Thm 4 claim: index policy is within one item's value of the MCKP optimum.
    ids, vc, cf = _toy(n=10, seed=7)
    budget = 80
    _, v_idx, _ = index_allocate(ids, vc, cf, budget)
    _, v_opt, _ = dp_allocate(ids, vc, cf, budget, scale=1)
    max_item = max(sum(vc(i)) for i in ids)
    assert v_opt - v_idx <= max_item + 1e-6
    # and in practice the gap is tiny
    assert v_idx >= 0.9 * v_opt


def test_concave_envelope_diminishing_returns():
    vals = [0.0, 0.45, 0.8, 1.0]
    costs = [1, 5, 9, 13]
    ups = _concave_envelope(vals, costs)
    slopes = [dv / dc for dv, dc, *_ in ups]
    assert slopes == sorted(slopes, reverse=True)  # non-increasing value/token


def test_lp_bound_sandwiches_index_and_dp():
    # The paper's frontier ordering:  index <= OPT(DP) <= LP_bound.
    ids, vc, cf = _toy(n=9, seed=11)
    budget = 70
    _, v_idx, _ = index_allocate(ids, vc, cf, budget)
    _, v_dp, _ = dp_allocate(ids, vc, cf, budget, scale=1)
    v_lp = lp_bound(ids, vc, cf, budget)
    assert v_idx <= v_dp + 1e-6
    assert v_dp <= v_lp + 1e-6


def test_lp_bound_huge_budget_equals_full_value():
    ids, vc, cf = _toy(n=6, seed=2)
    full_value = sum(vc(i)[Fidelity.FULL] for i in ids)
    v_lp = lp_bound(ids, vc, cf, budget=10_000)  # budget never binds
    assert abs(v_lp - full_value) < 1e-6


def test_index_drops_unaffordable_upgrade_but_continues():
    # one expensive item + cheap items; greedy must still fill with cheap upgrades.
    ids = ["big", "a", "b"]
    base = {"big": 5.0, "a": 1.0, "b": 1.0}
    ymul = (0.0, 0.45, 0.8, 1.0)
    cost = {"big": (1, 50, 90, 200), "a": (1, 3, 5, 7), "b": (1, 3, 5, 7)}
    vc = lambda i: [base[i] * y for y in ymul]
    cf = lambda i, lv: cost[i][int(lv)]
    levels, val, spent = index_allocate(ids, vc, cf, budget=20)
    assert spent <= 20
    assert levels["a"] > Fidelity.POINTER or levels["b"] > Fidelity.POINTER


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        fn(); print(f"ok: {fn.__name__}")
    print("all tests passed")
