"""
Budgeted fidelity allocator.

Problem (per turn): choose a fidelity level f_i in {POINTER, GIST, DIGEST, FULL} for every
live item to MAXIMIZE   sum_i v_i(f_i)   SUBJECT TO   sum_i cost_i(f_i) <= B.

This is a Multiple-Choice Knapsack Problem (MCKP). We solve it two ways:

  1. index_allocate(): the deployable O(N log N) greedy index policy. We sort all
     *incremental upgrades* (level k -> k+1) by value-per-token (the Lagrangian/Whittle
     index lambda) and buy upgrades in decreasing-index order while budget remains, after
     first taking the convex-hull (concave envelope) of each item's value curve so that
     every purchased upgrade is on the efficient frontier. This is exactly the LP-relaxation
     greedy and is provably within one item's value of the MCKP optimum.

  2. dp_allocate(): an exact pseudo-polynomial DP, used in tests/experiments to MEASURE the
     greedy gap (Theorem: greedy gap is small iff cross-item value curves are near-concave
     and budget is non-degenerate -- see paper Sec. 4).

The index policy is what Foveance runs in production; the DP is the oracle we benchmark against.
"""
from __future__ import annotations

import heapq
import itertools
from typing import Callable
from .store import Fidelity

LEVELS = list(Fidelity)  # [POINTER, GIST, DIGEST, FULL]


def _concave_envelope(values: list, costs: list):
    """Return upgrades on the upper-left convex hull of (cost, value) points, as a list of
    (delta_value, delta_cost, from_level, to_level), each with non-increasing value/cost.
    Guarantees the greedy index policy only buys efficient upgrades."""
    # points sorted by cost
    pts = sorted(range(len(values)), key=lambda i: costs[i])
    hull = [pts[0]]
    for idx in pts[1:]:
        # keep only points that improve value as cost grows, with diminishing returns
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # slope a->b vs slope b->idx; drop b if it's dominated (concavity)
            s_ab = (values[b] - values[a]) / max(1e-9, costs[b] - costs[a])
            s_bi = (values[idx] - values[b]) / max(1e-9, costs[idx] - costs[b])
            if s_bi >= s_ab:        # b not on concave hull
                hull.pop()
            else:
                break
        if values[idx] >= values[hull[-1]]:
            hull.append(idx)
    upgrades = []
    for a, b in zip(hull, hull[1:]):
        dv, dc = values[b] - values[a], costs[b] - costs[a]
        if dc > 0 and dv > 0:
            upgrades.append((dv, dc, a, b))
    return upgrades


def index_allocate(item_ids: list, value_curve: Callable, cost_fn: Callable, budget: int):
    """Greedy Lagrangian-index MCKP. Returns {item_id: Fidelity}, total_value, total_cost.

    Implemented with a binary heap so the deployable policy runs in true
    ``O(sum_i L_i log sum_i L_i)`` time (Thm. index-policy near-optimality): each of the
    O(nL) concave-envelope upgrades is pushed/popped once. The value curve and cost ladder
    are evaluated exactly once per item. A monotonic counter breaks index ties deterministically
    and keeps the heap from ever comparing the (non-orderable) payloads.
    """
    levels = {iid: Fidelity.POINTER for iid in item_ids}
    spent = 0
    value = 0.0
    tie = itertools.count()
    heap: list = []  # (-value_per_token, tiebreak, iid, to_level, dv, dc, ups, k)
    for iid in item_ids:
        vs = value_curve(iid)                       # list over LEVELS, evaluated once
        cs = [cost_fn(iid, lv) for lv in LEVELS]
        spent += cs[Fidelity.POINTER]
        value += vs[Fidelity.POINTER]
        ups = _concave_envelope(vs, cs)
        if ups:
            dv, dc, _a, b = ups[0]
            heapq.heappush(heap, (-(dv / dc), next(tie), iid, b, dv, dc, ups, 0))

    while heap:
        _negidx, _t, iid, b, dv, dc, ups, k = heapq.heappop(heap)
        if spent + dc <= budget:
            levels[iid] = Fidelity(b)
            spent += dc
            value += dv
            if k + 1 < len(ups):                    # queue this item's next efficient upgrade
                dv2, dc2, _a2, b2 = ups[k + 1]
                heapq.heappush(heap, (-(dv2 / dc2), next(tie), iid, b2, dv2, dc2, ups, k + 1))
        # if it doesn't fit, drop it: the same item's later upgrades cost strictly more.
    return levels, value, spent


def lp_bound(item_ids: list, value_curve: Callable, cost_fn: Callable, budget: int) -> float:
    """LP-relaxation upper bound on the MCKP optimum (the theory plots' frontier).

    Same greedy as ``index_allocate`` on the concavified curves, but the final upgrade that
    would overflow the budget is taken *fractionally*. By LP duality this value upper-bounds
    the integral optimum, so ``index <= OPT <= lp_bound`` always holds (Thm: index policy).
    """
    value = sum(value_curve(iid)[Fidelity.POINTER] for iid in item_ids)
    spent = sum(cost_fn(iid, Fidelity.POINTER) for iid in item_ids)
    ups = []  # (index, dv, dc)
    for iid in item_ids:
        vs = value_curve(iid)
        cs = [cost_fn(iid, lv) for lv in LEVELS]
        for dv, dc, _a, _b in _concave_envelope(vs, cs):
            ups.append((dv / dc, dv, dc))
    ups.sort(reverse=True)
    for _idx, dv, dc in ups:
        if spent + dc <= budget:
            spent += dc
            value += dv
        else:  # take the overflowing upgrade fractionally (LP relaxation)
            frac = max(0.0, (budget - spent)) / dc
            value += frac * dv
            break
    return value


def dp_allocate(item_ids: list, value_curve: Callable, cost_fn: Callable, budget: int,
                scale: int = 1):
    """Exact MCKP via DP over a (down-scaled) token budget. Oracle for measuring greedy gap."""
    B = budget // scale
    NEG = float("-inf")
    dp = [NEG] * (B + 1)
    dp[0] = 0.0
    # process items one by one
    cur = dp[:]
    back: list = []
    for iid in item_ids:
        vs = value_curve(iid)
        cs = [max(0, cost_fn(iid, lv) // scale) for lv in LEVELS]
        nxt = [NEG] * (B + 1)
        bk: list = [None] * (B + 1)
        for b in range(B + 1):
            if cur[b] == NEG:
                continue
            for li, lv in enumerate(LEVELS):
                nb = b + cs[li]
                if nb <= B and cur[b] + vs[li] > nxt[nb]:
                    nxt[nb] = cur[b] + vs[li]
                    bk[nb] = (b, lv)
        cur = nxt
        back.append(bk)
    # best over all budgets <= B
    best_b = max(range(B + 1), key=lambda b: cur[b])
    best_v = cur[best_b]
    # reconstruct
    levels = {}
    b = best_b
    for iid, bk in zip(reversed(item_ids), reversed(back)):
        pb, lv = bk[b]
        levels[iid] = lv
        b = pb
    return levels, best_v, best_b * scale
