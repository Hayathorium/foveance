#!/usr/bin/env python3
"""Plot the single-shot baseline comparison (bench/results_baselines/single_shot.csv) into
bench/plots/baseline_comparison.png: recall accuracy by method across budgets, plus token cost."""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import argparse

HERE = os.path.dirname(__file__)
ORDER = ["full", "recency", "truncate", "uniform", "llmlingua2", "reactive_afm", "foveance"]
LABEL = {"full": "full", "recency": "recency", "truncate": "truncate", "uniform": "uniform",
         "llmlingua2": "LLMLingua-2", "reactive_afm": "reactive (AFM)", "foveance": "foveance"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "results_baselines", "single_shot.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "plots", "baseline_comparison.png"))
    ap.add_argument("--title", default="single-shot recall")
    args = ap.parse_args()
    SRC, OUT = args.src, args.out
    if not os.path.exists(SRC):
        print(f"no {SRC}; run bench/compare_baselines.py first")
        return 2
    rows = list(csv.DictReader(open(SRC)))
    budgets = sorted({int(r["budget"]) for r in rows})
    acc = {(r["method"], int(r["budget"])): float(r["accuracy"]) for r in rows}
    tok = {(r["method"], int(r["budget"])): float(r["avg_in_tok"]) for r in rows}
    methods = [m for m in ORDER if any(r["method"] == m for r in rows)]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    x = np.arange(len(methods))
    w = 0.8 / len(budgets)
    for j, b in enumerate(budgets):
        ax1.bar(x + j * w, [acc.get((m, b), 0) for m in methods], w, label=f"budget {b}")
        ax2.bar(x + j * w, [tok.get((m, b), 0) for m in methods], w, label=f"budget {b}")
    ticks = x + w * (len(budgets) - 1) / 2
    for ax in (ax1, ax2):
        ax.set_xticks(ticks)
        ax.set_xticklabels([LABEL[m].replace(" (", "\n(") for m in methods], fontsize=8)
        ax.grid(alpha=.3, axis="y")
    ax1.set_ylabel("recall accuracy")
    ax1.set_ylim(0, 1.08)
    ax1.set_title(f"(a) Keeps the load-bearing fact? ({args.title})", fontsize=10)
    ax2.set_ylabel("model input tokens")
    ax2.set_title("(b) Token cost (full replay is the dashed reference)", fontsize=10)
    full_tok = max((tok.get(("full", b), 0) for b in budgets), default=0)
    if full_tok:
        ax2.axhline(full_tok, color="gray", ls="--", lw=1, alpha=.7)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(budgets), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
