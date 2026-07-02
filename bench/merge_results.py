#!/usr/bin/env python3
"""Merge results_v2 (small models) into results/ for a 5-model comparison, keeping the rich
mock greedy_gap + ablations already in results/. v2 rows win on model overlap (gemma2:2b)."""
from __future__ import annotations

import csv
import os

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
V2 = os.path.join(HERE, "results_v2")


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def merge(name):
    base = load(os.path.join(RES, name))
    add = load(os.path.join(V2, name))
    v2_models = {r["model"] for r in add}
    # keep v2 rows for its models; keep base rows for models v2 does not cover
    merged = add + [r for r in base if r["model"] not in v2_models]
    if not merged:
        return
    # align fieldnames (union, base order first)
    fields = list(dict.fromkeys(list(merged[0].keys()) + [k for r in merged for k in r.keys()]))
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"{name}: {len({r['model'] for r in merged})} models, {len(merged)} rows")


if __name__ == "__main__":
    merge("by_seed.csv")
    merge("per_turn.csv")
