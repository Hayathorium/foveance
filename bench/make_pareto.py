#!/usr/bin/env python3
"""
Plot a clean accuracy-vs-token Pareto frontier from a by_seed.csv produced over a budget sweep.

Unlike the per-model panels in plots.py, this is meant for a *tight*-budget sweep where accuracy
genuinely trades off with the budget, so the frontier is informative and monotone. Each arm is a
curve over budgets, with 95%% bootstrap CIs; full replay and recency are single reference points.

Usage: python bench/make_pareto.py --indir bench/results_pareto --out bench/plots/pareto_frontier.png
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["full", "recency", "truncate", "uniform", "reactive_afm", "foveance", "oracle"]
LABEL = {"full": "full", "recency": "recency", "truncate": "truncate", "uniform": "uniform",
         "reactive_afm": "reactive (AFM)", "foveance": "foveance", "oracle": "oracle (DP)"}
MARK = {"full": "s", "recency": "X", "truncate": "v", "uniform": "P", "reactive_afm": "^",
        "foveance": "o", "oracle": "D"}


def boot_ci(xs, n=10000, seed=0):
    a = np.asarray(xs, float)
    if len(a) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), size=(n, len(a)))].mean(1)
    return float(a.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=os.path.join(os.path.dirname(__file__), "results_pareto"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "plots", "pareto_frontier.png"))
    args = ap.parse_args()

    path = os.path.join(args.indir, "by_seed.csv")
    if not os.path.exists(path):
        print(f"no {path}")
        return 2
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["budget"] = int(float(r["budget"]))
        r["accuracy"] = float(r["accuracy"])
        r["total_tok"] = int(float(r["in_tok"])) + int(float(r["out_tok"]))
    model = sorted({r["model"] for r in rows})[0]
    g = defaultdict(list)
    for r in rows:
        g[(r["policy"], r["budget"])].append(r)
    budgets = sorted({r["budget"] for r in rows})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(6.0, 4.2))
    tokspan = (max(r["total_tok"] for r in rows) - min(r["total_tok"] for r in rows)) or 1.0
    budgeted = [p for p in ORDER if p not in ("full", "recency") and any(r["policy"] == p for r in rows)]
    for p in [p for p in ORDER if any(r["policy"] == p for r in rows)]:
        pts = []
        for b in budgets:
            rs = g.get((p, b))
            if not rs:
                continue
            acc, lo, hi = boot_ci([r["accuracy"] for r in rs])
            tok = float(np.mean([r["total_tok"] for r in rs]))
            pts.append((tok, acc, acc - lo, hi - acc))
        if not pts:
            continue
        pts.sort()
        xs = [a for a, *_ in pts]
        ys = [b for _, b, *_ in pts]
        yl = [c for *_, c, _ in pts]
        yh = [d for *_, d in pts]
        if p in ("full", "recency"):  # budget-independent: single averaged reference point
            plt.errorbar([float(np.mean(xs))], [float(np.mean(ys))], marker=MARK[p],
                         markersize=9, capsize=3, label=LABEL[p])
        else:  # small jitter so tied budgeted arms (reactive, foveance) don't hide each other
            jit = (budgeted.index(p) - (len(budgeted) - 1) / 2) * 0.02 * tokspan
            plt.errorbar([x + jit for x in xs], ys, yerr=[yl, yh], marker=MARK[p], markersize=7,
                         capsize=3, alpha=0.9, label=LABEL[p])
    plt.xlabel("total tokens (sum over the trajectory)")
    plt.ylabel("task accuracy")
    plt.title(f"Accuracy--token frontier over a budget sweep ({model})")
    plt.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
