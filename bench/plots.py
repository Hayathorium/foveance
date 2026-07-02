#!/usr/bin/env python3
"""Generate paper figures from results/summary.csv (+ greedy_gap.csv) into bench/plots/."""
from __future__ import annotations
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)
POLICIES = ["full", "recency", "truncate", "uniform", "reactive_afm", "foveance", "oracle"]
MARK = {"full": "s", "recency": "x", "truncate": "v", "uniform": "P", "reactive_afm": "^",
        "foveance": "o", "oracle": "D"}


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    rows = load(os.path.join(RES, "summary.csv"))
    if not rows:
        print("no summary.csv -- run analyze.py first"); return
    for r in rows:
        r["budget"] = int(float(r["budget"]))
        for k in ("acc_mean", "acc_lo", "acc_hi", "total_tok_mean", "tok_per_correct"):
            r[k] = float(r[k])
    models = sorted({r["model"] for r in rows})

    plabel = {"full": "full", "recency": "recency", "truncate": "truncate", "uniform": "uniform",
              "reactive_afm": "reactive (AFM)", "foveance": "foveance", "oracle": "oracle (DP)"}
    present = [p for p in POLICIES if any(r["policy"] == p for r in rows)]

    # Pareto frontier per model. A small horizontal jitter separates arms that coincide
    # (e.g. reactive and foveance) so neither hides the other.
    for m in models:
        plt.figure(figsize=(5.6, 4))
        mr = [r for r in rows if r["model"] == m]
        xspan = (max(r["total_tok_mean"] for r in mr) - min(r["total_tok_mean"] for r in mr)) or 1.0
        for k, p in enumerate(present):
            pts = sorted([(r["total_tok_mean"], r["acc_mean"], r["acc_lo"], r["acc_hi"])
                          for r in rows if r["model"] == m and r["policy"] == p])
            if not pts:
                continue
            jit = (k - (len(present) - 1) / 2) * 0.012 * xspan
            xs = [a + jit for a, *_ in pts]; ys = [b for _, b, *_ in pts]
            lo = [b-l for _, b, l, _ in pts]; hi = [h-b for _, b, _, h in pts]
            plt.errorbar(xs, ys, yerr=[lo, hi], marker=MARK.get(p, "."), capsize=3,
                         markersize=7, alpha=0.85, label=plabel.get(p, p))
        plt.xlabel("total tokens"); plt.ylabel("accuracy")
        plt.title(f"Accuracy–token Pareto: {m}")
        plt.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        plt.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"pareto_{m.replace(':','_')}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # tokens-per-correct bar at richest budget (only arms with data)
    B = max(r["budget"] for r in rows)
    plt.figure(figsize=(6, 4))
    width = 0.8 / max(1, len(models)); xs = range(len(present))
    for i, m in enumerate(models):
        vals = []
        for p in present:
            cand = [r["tok_per_correct"] for r in rows if r["model"] == m and r["policy"] == p and r["budget"] == B]
            vals.append(cand[0] if cand else 0)
        plt.bar([x + i*width for x in xs], vals, width, label=m)
    plt.xticks([x + width*(len(models)-1)/2 for x in xs],
               [plabel.get(p, p).replace(" (", "\n(") for p in present], fontsize=8)
    plt.ylabel("tokens per correct answer (↓)"); plt.title(f"Token efficiency at budget {B}")
    plt.legend(fontsize=8, title="model", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "tok_per_correct.png"), dpi=150, bbox_inches="tight"); plt.close()

    # ---- comparison visuals (the "how good is it" charts) ----
    LABEL = {"full": "full", "recency": "recency", "truncate": "truncate", "uniform": "uniform",
             "reactive_afm": "reactive (AFM)", "foveance": "foveance", "oracle": "oracle (DP)"}

    def best_acc_budget(m, p):
        cand = [r for r in rows if r["model"] == m and r["policy"] == p]
        return max(cand, key=lambda r: (r["acc_mean"], -r["total_tok_mean"])) if cand else None

    def full_ref(m):
        rs = [r for r in rows if r["model"] == m and r["policy"] == "full"]
        return (max(r["acc_mean"] for r in rs), float(np.mean([r["total_tok_mean"] for r in rs]))) \
            if rs else (1.0, 1.0)

    import numpy as np

    # (1) Iso-accuracy token savings vs full, for the budgeted arms that reach full accuracy.
    plt.figure(figsize=(6.4, 4))
    arms = [a for a in ["reactive_afm", "foveance", "oracle"] if a in present]
    width = 0.8 / max(1, len(arms))
    xs = range(len(models))
    for j, p in enumerate(arms):
        vals = []
        for m in models:
            facc, ftok = full_ref(m)
            reached = [r["total_tok_mean"] for r in rows
                       if r["model"] == m and r["policy"] == p and r["acc_mean"] >= facc - 0.01]
            vals.append(100.0 * (ftok - min(reached)) / ftok if reached and ftok else 0.0)
        plt.bar([x + j * width for x in xs], vals, width, label=LABEL[p])
    plt.xticks([x + width * (len(arms) - 1) / 2 for x in xs], models, fontsize=8, rotation=10)
    plt.ylabel("% tokens saved at iso-accuracy vs full (↑)")
    plt.title("Token savings while matching full-replay accuracy", pad=24)
    plt.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=len(arms), frameon=False)
    plt.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "savings_vs_full.png"), dpi=150, bbox_inches="tight"); plt.close()

    # (2) Headline comparison: accuracy (left) and token cost (right) by policy, per model.
    arms2 = [a for a in ["full", "recency", "truncate", "uniform", "reactive_afm", "foveance",
                         "oracle"] if any(r["policy"] == a for r in rows)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.8 / max(1, len(arms2))
    xs = list(range(len(models)))
    colors = plt.cm.tab10.colors
    for j, p in enumerate(arms2):
        accs, toks = [], []
        for m in models:
            r = best_acc_budget(m, p)
            accs.append(r["acc_mean"] if r else 0.0)
            toks.append((r["total_tok_mean"] / 1000.0) if r else 0.0)
        pos = [x + j * width for x in xs]
        ax1.bar(pos, accs, width, label=LABEL.get(p, p), color=colors[j])
        ax2.bar(pos, toks, width, label=LABEL.get(p, p), color=colors[j])
    ticks = [x + width * (len(arms2) - 1) / 2 for x in xs]
    ax1.set_xticks(ticks); ax1.set_xticklabels(models, fontsize=8, rotation=12)
    ax2.set_xticks(ticks); ax2.set_xticklabels(models, fontsize=8, rotation=12)
    ax1.set_ylabel("task accuracy"); ax1.set_ylim(0, 1.05)
    ax1.set_title("(a) Accuracy by policy (higher is better)", fontsize=10)
    ax2.set_ylabel("total tokens (thousands)")
    ax2.set_title("(b) Token cost by policy (lower is better)", fontsize=10)
    ax1.grid(alpha=.3, axis="y"); ax2.grid(alpha=.3, axis="y")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(arms2), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(os.path.join(OUT, "efficiency.png"), dpi=150, bbox_inches="tight"); plt.close()

    # greedy-gap histogram
    gg = load(os.path.join(RES, "greedy_gap.csv"))
    if gg:
        import numpy as np
        rg = np.array([float(r["rel_gap"]) for r in gg])
        plt.figure(figsize=(5, 3.6))
        plt.hist(rg, bins=30); plt.axvline(rg.mean(), color="k", ls="--",
                                           label=f"mean={rg.mean():.4f}")
        plt.xlabel("relative gap (DP − index)/DP"); plt.ylabel("count")
        plt.title("Index-policy greedy gap (Thm 3)")
        plt.legend(fontsize=8, loc="upper right", frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "greedy_gap.png"), dpi=150, bbox_inches="tight"); plt.close()

    # drift sweep: tokens-per-correct vs drift for reactive_afm vs foveance (Thm 2/4)
    abl = load(os.path.join(RES, "ablations.csv"))
    drift_rows = [r for r in abl if r.get("ablation") == "drift"]
    if drift_rows:
        for nt in ("True", "False"):
            sub = [r for r in drift_rows if r["setting"].endswith(f"name_target={nt}")]
            if not sub:
                continue
            plt.figure(figsize=(5, 3.6))
            for arm in ("reactive_afm", "foveance"):
                pts = sorted((float(r["setting"].split(",")[0].split("=")[1]),
                              float(r["tok_per_correct"])) for r in sub if r["arm"] == arm)
                if pts:
                    plt.plot([d for d, _ in pts], [v for _, v in pts], marker="o",
                             label=("reactive (AFM)" if arm == "reactive_afm" else arm))
            plt.xlabel("drift (cross-turn dependency)"); plt.ylabel("tokens per correct (↓)")
            plt.title(f"Anticipation vs drift (name_target={nt})")
            plt.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            plt.grid(alpha=.3); plt.tight_layout()
            plt.savefig(os.path.join(OUT, f"drift_sweep_nt{nt}.png"), dpi=150, bbox_inches="tight")
            plt.close()

    print(f"wrote plots to {OUT}/")


if __name__ == "__main__":
    main()
