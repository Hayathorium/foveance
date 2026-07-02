#!/usr/bin/env python3
"""
Analyze benchmark output (results/by_seed.csv [+ greedy_gap.csv]) and produce:
  - results/summary.csv     mean +/- 95% bootstrap CI per (model, policy, budget)
  - results/headline.json   machine-readable key comparisons (for the paper)
  - report.md               human-readable headline report

Stats are self-contained (numpy only): bootstrap CIs, paired-bootstrap CI of the
foveance-vs-reactive accuracy difference, and a Wilcoxon signed-rank normal approximation.
For a publication, PROMPT_2 swaps in scipy.stats.wilcoxon; results agree for n>=8.
"""
from __future__ import annotations
import csv, json, os, math
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
EPS = 0.01  # iso-accuracy tolerance


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def boot_ci(xs, n=10000, alpha=0.05, seed=0):
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = xs[rng.integers(0, len(xs), size=(n, len(xs)))].mean(1)
    return float(xs.mean()), float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))


def wilcoxon_p(diffs):
    """Two-sided Wilcoxon signed-rank, normal approx w/ continuity correction."""
    d = np.asarray([x for x in diffs if abs(x) > 1e-12], float)
    n = len(d)
    if n < 1:
        return float("nan")
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    W = np.sum(ranks[d > 0])
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2*n + 1) / 24.0)
    if sigma == 0:
        return float("nan")
    z = (W - mu - 0.5*np.sign(W - mu)) / sigma
    return float(math.erfc(abs(z)/math.sqrt(2)))  # 2-sided


def pareto_auc(points):
    """Area under accuracy-vs-tokens curve, normalized by token span. points: list[(tokens,acc)]."""
    pts = sorted(set(points))
    if len(pts) < 2:
        return float(pts[0][1]) if pts else float("nan")
    xs = np.array([p[0] for p in pts], float); ys = np.array([p[1] for p in pts], float)
    area = float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2.0))
    return area / (xs.max() - xs.min())


def main():
    rows = load(os.path.join(RES, "by_seed.csv"))
    assert rows, "no by_seed.csv -- run bench/run_bench.py first"
    for r in rows:
        for k in ("budget", "seed", "in_tok", "out_tok", "peak_tok", "n_recall", "n_correct", "n_turns"):
            r[k] = int(float(r[k]))
        r["reinflations"] = int(float(r.get("reinflations", 0)))
        r["accuracy"] = float(r["accuracy"]); r["wall_s"] = float(r["wall_s"])

    models = sorted({r["model"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})
    policies = ["full", "recency", "truncate", "uniform", "reactive_afm", "foveance", "oracle"]

    # group
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["policy"], r["budget"])].append(r)

    # summary.csv with CIs
    summary = []
    for m in models:
        for p in policies:
            for b in budgets:
                rs = g.get((m, p, b))
                if not rs:
                    continue
                accs = [r["accuracy"] for r in rs]
                tot = [r["in_tok"] + r["out_tok"] for r in rs]
                mean, lo, hi = boot_ci(accs)
                summary.append({
                    "model": m, "policy": p, "budget": b,
                    "acc_mean": round(mean, 4), "acc_lo": round(lo, 4), "acc_hi": round(hi, 4),
                    "in_tok_mean": round(np.mean([r["in_tok"] for r in rs]), 1),
                    "total_tok_mean": round(np.mean(tot), 1),
                    "peak_tok_mean": round(np.mean([r["peak_tok"] for r in rs]), 1),
                    "ms_per_turn": round(1000*np.mean([r["wall_s"]/max(1, r["n_turns"]) for r in rs]), 2),
                    "tok_per_correct": round(np.mean([ (r["in_tok"]/max(1e-9, r["n_correct"])) for r in rs]), 1),
                    "reinflations_mean": round(np.mean([r["reinflations"] for r in rs]), 2),
                    "n_seeds": len(rs),
                })
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    # pareto.csv: the accuracy-vs-tokens frontier points per (model, policy, budget).
    pareto = [{"model": s["model"], "policy": s["policy"], "budget": s["budget"],
               "total_tok_mean": s["total_tok_mean"], "acc_mean": s["acc_mean"],
               "acc_lo": s["acc_lo"], "acc_hi": s["acc_hi"]} for s in summary]
    with open(os.path.join(RES, "pareto.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pareto[0].keys())); w.writeheader(); w.writerows(pareto)

    def acc_at(m, p, b):
        rs = g.get((m, p, b), []); return np.mean([r["accuracy"] for r in rs]) if rs else float("nan")
    def tok_at(m, p, b):
        rs = g.get((m, p, b), []); return np.mean([r["in_tok"]+r["out_tok"] for r in rs]) if rs else float("nan")

    # Use the budget shared by the most models for the main table (robust to mixed-budget merges).
    budget_coverage = {b: sum(1 for m in models if g.get((m, "foveance", b))) for b in budgets}
    table_budget = max(budgets, key=lambda b: (budget_coverage[b], b))
    headline = {"models": models, "budgets": budgets, "budget_for_table": table_budget, "per_model": {}}
    for m in models:
        full_acc = max((acc_at(m, "full", b) for b in budgets), default=float("nan"))
        full_tok = np.nanmean([tok_at(m, "full", b) for b in budgets])
        # iso-accuracy: cheapest foveance budget reaching full accuracy
        reached = [(tok_at(m, "foveance", b), b) for b in budgets if acc_at(m, "foveance", b) >= full_acc - EPS]
        if reached:
            aux_tok, aux_b = min(reached)
            savings = 100.0 * (full_tok - aux_tok) / full_tok if full_tok else float("nan")
            reached_full = True
        else:
            best_b = max(budgets, key=lambda b: acc_at(m, "foveance", b))
            aux_tok, aux_b, savings, reached_full = tok_at(m, "foveance", best_b), best_b, float("nan"), False
        # foveance vs reactive: delta acc at each (iso-token) budget + paired tests at richest budget
        deltas = {b: round(acc_at(m, "foveance", b) - acc_at(m, "reactive_afm", b), 4) for b in budgets}
        bb = budgets[len(budgets)//2]  # a contested mid budget
        pa = sorted(g.get((m, "foveance", bb), []), key=lambda r: r["seed"])
        pr = sorted(g.get((m, "reactive_afm", bb), []), key=lambda r: r["seed"])
        pair = [a["accuracy"] - b["accuracy"] for a, b in zip(pa, pr)]
        dmean, dlo, dhi = boot_ci(pair) if pair else (float("nan"),)*3
        headline["per_model"][m] = {
            "full_acc": round(float(full_acc), 4), "full_tokens": round(float(full_tok), 1),
            "foveance_reached_full": reached_full,
            "foveance_iso_tokens": round(float(aux_tok), 1), "foveance_iso_budget": int(aux_b),
            "foveance_iso_savings_pct_vs_full": (round(float(savings), 1) if savings==savings else None),
            "foveance_vs_reactive_delta_by_budget": deltas,
            "foveance_vs_reactive_max_delta": max(deltas.values()),
            "foveance_vs_reactive_delta_ci_at_mid": [round(dmean,4), round(dlo,4), round(dhi,4)],
            "foveance_vs_reactive_wilcoxon_p_at_mid": round(wilcoxon_p(pair), 4) if pair else None,
            "pareto_auc": {p: round(pareto_auc([(tok_at(m,p,b), acc_at(m,p,b)) for b in budgets]), 4)
                           for p in policies},
        }

    gg = load(os.path.join(RES, "greedy_gap.csv"))
    if gg:
        rg = np.array([float(r["rel_gap"]) for r in gg])
        gap = {"n": len(rg), "mean_rel_gap": round(float(rg.mean()), 5),
               "p95_rel_gap": round(float(np.quantile(rg, 0.95)), 5),
               "max_rel_gap": round(float(rg.max()), 5)}
        if "idx_over_lp" in gg[0]:
            il = np.array([float(r["idx_over_lp"]) for r in gg])
            gap["mean_index_over_lp"] = round(float(il.mean()), 5)
            gap["min_index_over_lp"] = round(float(il.min()), 5)
        headline["greedy_gap"] = gap

    # audit + ablations passthrough into headline
    audit = {}
    ap = os.path.join(RES, "drift_twin_audit.json")
    if os.path.exists(ap):
        with open(ap) as f:
            audit = json.load(f)
        headline["drift_twin_audit"] = audit
    ablations = load(os.path.join(RES, "ablations.csv"))
    src = "real" if any(m != "mock" for m in models) else "mock"
    headline["source"] = src

    with open(os.path.join(RES, "headline.json"), "w") as f:
        json.dump(headline, f, indent=2)

    # report.md
    banner = ("> **Source: REAL model runs.**" if src == "real" else
              "> **Source: MOCK model (illustrative; offline CI). Re-run with Ollama for real numbers.**")
    L = ["# Foveance benchmark report", "", banner, "",
         f"- models: {', '.join(models)}", f"- budgets: {budgets}",
         f"- seeds per cell: {summary[0]['n_seeds']}",
         (f"- drift-twin audit: reactive_afm vs foveance differ only in drift = "
          f"{audit.get('only_difference_is_drift')}" if audit else ""), ""]
    for m in models:
        h = headline["per_model"][m]
        L += [f"## {m}",
              f"- full replay: acc={h['full_acc']}, total tokens={h['full_tokens']:.0f}",
              (f"- **foveance reaches full accuracy at budget {h['foveance_iso_budget']} using "
               f"{h['foveance_iso_tokens']:.0f} tokens => {h['foveance_iso_savings_pct_vs_full']}% fewer than full**"
               if h["foveance_reached_full"] else
               f"- foveance did not reach full accuracy within tested budgets (best at budget {h['foveance_iso_budget']})"),
              f"- foveance vs reactive (AFM) max Δacc across budgets: {h['foveance_vs_reactive_max_delta']}",
              f"- foveance vs reactive paired Δacc CI (mid budget): {h['foveance_vs_reactive_delta_ci_at_mid']}, "
              f"Wilcoxon p={h['foveance_vs_reactive_wilcoxon_p_at_mid']}",
              f"- Pareto AUC: " + ", ".join(f"{p}={h['pareto_auc'][p]}" for p in policies), ""]
    if "greedy_gap" in headline:
        gh = headline["greedy_gap"]
        lp_line = (f"; mean index/LP = {gh['mean_index_over_lp']} (min {gh['min_index_over_lp']})"
                   if "mean_index_over_lp" in gh else "")
        L += ["## Greedy gap (Thm 3, index vs exact DP vs LP bound)",
              f"- mean relative gap = {gh['mean_rel_gap']}, p95 = {gh['p95_rel_gap']}, "
              f"max = {gh['max_rel_gap']} over {gh['n']} measurements{lp_line}", ""]
    if ablations:
        L += ["## Ablations", "", "| ablation | setting | arm | accuracy | in_tok | tok/correct |",
              "|---|---|---|---|---|---|"]
        for a in ablations:
            L.append(f"| {a['ablation']} | {a['setting']} | {a['arm']} | {a['accuracy']} "
                     f"| {a['in_tok']} | {a['tok_per_correct']} |")
        L += ["", "Drift sweep tests Thm 2/4 (anticipation gain vs cross-turn dependency); "
              "retrieve/fidelity-cost rows test two-sided refinement and the re-fetch penalty.", ""]
    L += ["## Reproduce", ""]
    if src == "real":
        L += ["Real-model run (Ollama). Token savings come from the budget binding on growing "
              "context; the greedy-gap and drift-sweep ablations are model-independent allocator "
              "and predictor measurements.", "", "```bash",
              "python bench/run_bench.py --backend ollama \\",
              f"    --models {','.join(models)} --suite synthetic \\",
              f"    --budgets {','.join(str(b) for b in budgets)} --tasks {summary[0]['n_seeds']} "
              "--turns 8 --n-facts 3 --block-lines 18 --drift 0.7 --greedy-gap --ablations",
              "python bench/analyze.py && python bench/plots.py", "```", ""]
    else:
        L += ["```bash",
              "python bench/run_bench.py --backend mock --models mock --suite synthetic \\",
              "    --budgets 600,1200,1600,2500,4000 --tasks 6 --turns 40 --drift 0.7 \\",
              "    --name-target false --fidelity-cost true --greedy-gap --ablations",
              "python bench/analyze.py && python bench/plots.py", "```", ""]
    with open(os.path.join(HERE, "..", "bench", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    # also drop a copy at repo bench/report.md path used by paper
    print("wrote summary.csv, headline.json, report.md")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
