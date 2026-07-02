#!/usr/bin/env python3
"""
Measure the per-turn Foveance overhead (predictor scoring + index allocation), to back the
paper's claim that it is negligible beside a model call. No model is invoked here.

Usage: python bench/overhead.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foveance import MultiFidelityStore, Item, AnticipatoryPredictor  # noqa: E402
from foveance.allocator import index_allocate  # noqa: E402


def measure(n_items: int, budget: int, reps: int = 50) -> float:
    store = MultiFidelityStore()
    for i in range(n_items):
        store.add(Item(f"i{i}", "tool_output", f"FACT k{i}=v{i}\n" + "noise line\n" * 12, i))
    pred = AnticipatoryPredictor(store)
    pred.observe_query("recall k0")
    pred.observe_query("recall k1")
    ids = store.order

    def cost_fn(iid, lv):
        return store.cost(iid, lv)

    def value_curve(iid):
        return pred.value_curve(iid, n_items)

    t0 = time.perf_counter()
    for _ in range(reps):
        index_allocate(ids, value_curve, cost_fn, budget)
    return (time.perf_counter() - t0) / reps * 1000.0  # ms/allocation


def main() -> None:
    print("Foveance per-turn allocation overhead (predictor value curves + index policy):\n")
    print(f"{'items':>8} {'budget':>8} {'ms/turn':>10}")
    for n in (10, 50, 200, 1000, 4000):
        ms = measure(n, budget=n * 6)
        print(f"{n:>8} {n*6:>8} {ms:>10.2f}")
    print("\nA single LLM call is tens of milliseconds (API) to tens of seconds (local CPU),")
    print("so the allocation overhead is negligible in the agent loop.")


if __name__ == "__main__":
    main()
