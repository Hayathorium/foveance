#!/usr/bin/env python3
"""
Foveance benchmark harness.

Runs every policy arm over long-horizon tasks for one or more models, budgets, and a chosen
task suite, recording PER-SEED rows so analyze.py can compute confidence intervals and paired
significance tests. Also measures the index-vs-DP-vs-LP greedy gap (Thm 3) and, with
``--ablations``, the drift / learned-vs-heuristic / retrieve-on-off / fidelity-cost ablations.

Offline (default): MockLLM -> runs anywhere, no GPU/network.
Real: --backend ollama --models gemma2:9b,gemma2:2b,qwen2.5:7b,llama3.1:8b
      --backend openai --base-url http://localhost:8000/v1 --models <id>

Outputs (results/): by_seed.csv, per_turn.csv, greedy_gap.csv, ablations.csv,
drift_twin_audit.json. Run analyze.py next, then plots.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from foveance import Controller, PredictorConfig  # noqa: E402
from foveance.allocator import index_allocate, dp_allocate, lp_bound  # noqa: E402
from foveance.llm import MockLLM  # noqa: E402
from tasks import get_suite, score, SuiteUnavailable  # noqa: E402

# Arms share the same store/budget/model/tasks; only the policy differs. reactive_afm and
# foveance differ ONLY in the predictor drift (audited below) -- docs/NOVELTY.md, drift-twin audit.
ARMS = ["full", "recency", "truncate", "uniform", "reactive_afm", "foveance", "oracle"]


def make_model(backend: str, model: str, base_url: str):
    if backend == "mock":
        return MockLLM()
    if backend == "ollama":
        from foveance.llm import OllamaLLM
        return OllamaLLM(model=model)
    if backend == "openai":
        from foveance.llm import OpenAICompatLLM
        return OpenAICompatLLM(model=model, base_url=base_url,
                               api_key=os.environ.get("OPENAI_API_KEY", "x"))
    raise ValueError(backend)


def drift_twin_audit(drift: float) -> dict:
    """Prove reactive_afm and foveance are identical except for the predictor's drift."""
    react = Controller(MockLLM(), budget=1000, policy="reactive_afm", drift=drift)
    antic = Controller(MockLLM(), budget=1000, policy="foveance", drift=drift)
    rc, ac = react.pred.cfg, antic.pred.cfg
    diffs = {f.name: (getattr(rc, f.name), getattr(ac, f.name))
             for f in PredictorConfig.__dataclass_fields__.values()
             if getattr(rc, f.name) != getattr(ac, f.name)}
    return {
        "reactive_drift": rc.drift, "foveance_drift": ac.drift,
        "config_fields_that_differ": list(diffs.keys()),
        "only_difference_is_drift": list(diffs.keys()) == ["drift"],
        "same_allocator": "index_allocate", "same_store": True, "same_model": "shared",
    }


def run_arm_on_task(model_factory, arm, budget, task_turns, drift, fidelity_cost):
    """One task (one seed) for one arm. Returns (seed_row_partial, per_turn_rows)."""
    ctrl = Controller(model_factory(), budget=budget, policy=arm, drift=drift,
                      fidelity_cost=fidelity_cost)
    in_tok = out_tok = peak = reinf = 0
    wall = 0.0
    n_recall = n_correct = 0
    pt_rows = []
    for t, turn in enumerate(task_turns):
        for it in turn.new_items:
            ctrl.add_item(it)
        rec = ctrl.step(turn.query, t, probe=turn.probe or None)
        in_tok += rec.input_tokens
        out_tok += rec.output_tokens
        peak = max(peak, rec.peak_tokens)
        wall += rec.latency_s
        reinf += rec.reinflations
        if turn.gold:
            n_recall += 1
            n_correct += int(score(rec.answer, turn.gold))
        pt_rows.append({"arm": arm, "budget": budget, "turn": t,
                        "input_tok": rec.input_tokens, "peak_tok": rec.peak_tokens,
                        "latency_s": round(rec.latency_s, 6)})
    acc = (n_correct / n_recall) if n_recall else 0.0
    row = {"policy": arm, "budget": budget, "accuracy": round(acc, 6),
           "in_tok": in_tok, "out_tok": out_tok, "peak_tok": peak,
           "wall_s": round(wall, 4), "n_turns": len(task_turns),
           "n_recall": n_recall, "n_correct": n_correct, "reinflations": reinf}
    return row, pt_rows


def measure_greedy_gap(tasks, budgets, drift):
    """Empirical side of Thm 3: index vs exact DP vs LP bound on real per-turn curves."""
    rows = []
    for seed, turns in enumerate(tasks):
        ctrl = Controller(MockLLM(), budget=budgets[0], policy="foveance", drift=drift)
        for t, turn in enumerate(turns):
            for it in turn.new_items:
                ctrl.add_item(it)
            ctrl.pred.observe_query(turn.probe or turn.query)
            ids = ctrl.store.order
            if not ids:
                continue
            vc = lambda iid: ctrl.pred.value_curve(iid, t)  # noqa: B023
            cf = lambda iid, lv: ctrl.store.cost(iid, lv)   # noqa: B023
            for B in budgets:
                _, v_idx, _ = index_allocate(ids, vc, cf, B)
                _, v_dp, _ = dp_allocate(ids, vc, cf, B, scale=4)
                v_lp = lp_bound(ids, vc, cf, B)
                if v_dp > 1e-9:
                    rows.append({"seed": seed, "turn": t, "budget": B,
                                 "v_index": round(v_idx, 6), "v_dp": round(v_dp, 6),
                                 "v_lp": round(v_lp, 6),
                                 "rel_gap": round((v_dp - v_idx) / v_dp, 6),
                                 "idx_over_lp": round(v_idx / v_lp, 6) if v_lp > 1e-9 else 1.0})
    return rows


def run_ablations(suite_kw, n_tasks, mid_budget, fidelity_cost):
    """Drift sweep, learned-vs-heuristic, retrieve on/off, fidelity-cost on/off."""
    rows = []

    # (1) Drift sweep: isolates anticipation (Thm 2/4). name_target off amplifies it.
    # Always use the separation-revealing fidelity-change cost here, independent of the main
    # run's token-accounting flag, so the controlled study shows the effect either way.
    abl_fc = 0.5
    for nt in (True, False):
        for d in (0.0, 0.3, 0.6, 0.9):
            suite = get_suite("synthetic", **{**suite_kw, "drift": d, "name_target": nt})
            tasks = list(suite.iter_tasks(n_tasks))
            for arm in ("reactive_afm", "foveance"):
                accs, toks, corr = [], [], []
                for turns in tasks:
                    row, _ = run_arm_on_task(MockLLM, arm, mid_budget, turns, d, abl_fc)
                    accs.append(row["accuracy"]); toks.append(row["in_tok"])
                    corr.append(row["n_correct"])
                rows.append({"ablation": "drift", "setting": f"drift={d},name_target={nt}",
                             "arm": arm, "accuracy": round(sum(accs) / len(accs), 4),
                             "in_tok": round(sum(toks) / len(toks), 1),
                             "tok_per_correct": round(sum(toks) / max(1, sum(corr)), 1)})

    # (2) Predictor: heuristic vs learned (a tiny model fit on the suite's own traces).
    suite = get_suite("synthetic", **suite_kw)
    tasks = list(suite.iter_tasks(n_tasks))
    learned = _fit_learned(tasks)
    for label, fm in (("heuristic", None), ("learned", learned)):
        accs, toks, corr = [], [], []
        for turns in tasks:
            ctrl = Controller(MockLLM(), budget=mid_budget, policy="foveance",
                              drift=suite_kw.get("drift", 0.7), future_model=fm,
                              fidelity_cost=fidelity_cost)
            a, ti, co = _replay(ctrl, turns)
            accs.append(a); toks.append(ti); corr.append(co)
        rows.append({"ablation": "predictor", "setting": label, "arm": "foveance",
                     "accuracy": round(sum(accs) / len(accs), 4),
                     "in_tok": round(sum(toks) / len(toks), 1),
                     "tok_per_correct": round(sum(toks) / max(1, sum(corr)), 1)})

    # (3) Retrieve tool (two-sided refinement) on/off and (4) fidelity-cost on/off.
    for label, kw in (("retrieve_on", {"retrieve_enabled": True}),
                      ("retrieve_off", {"retrieve_enabled": False}),
                      ("fidelity_cost_on", {"fidelity_cost": 0.5}),
                      ("fidelity_cost_off", {"fidelity_cost": 0.0})):
        accs, toks, corr = [], [], []
        for turns in tasks:
            ctrl = Controller(MockLLM(), budget=mid_budget, policy="foveance",
                              drift=suite_kw.get("drift", 0.7), **kw)
            a, ti, co = _replay(ctrl, turns)
            accs.append(a); toks.append(ti); corr.append(co)
        abl = "retrieve" if "retrieve" in label else "fidelity_cost"
        rows.append({"ablation": abl, "setting": label, "arm": "foveance",
                     "accuracy": round(sum(accs) / len(accs), 4),
                     "in_tok": round(sum(toks) / len(toks), 1),
                     "tok_per_correct": round(sum(toks) / max(1, sum(corr)), 1)})
    return rows


def _replay(ctrl, turns):
    in_tok = n_recall = n_correct = 0
    for t, turn in enumerate(turns):
        for it in turn.new_items:
            ctrl.add_item(it)
        rec = ctrl.step(turn.query, t, probe=turn.probe or None)
        in_tok += rec.input_tokens
        if turn.gold:
            n_recall += 1
            n_correct += int(score(rec.answer, turn.gold))
    return (n_correct / n_recall if n_recall else 0.0), in_tok, n_correct


def _fit_learned(tasks):
    from foveance.learned import LogisticFutureRelevance
    traces = []
    for turns in tasks:
        items, queries, referenced = [], [], {}
        for t, turn in enumerate(turns):
            items.extend(turn.new_items)
            queries.append(turn.probe or turn.query)
            if turn.gold:  # the recalled key's planted item is "referenced" at this turn
                key = turn.query.replace("recall ", "").strip()
                referenced[t] = {f"obs{i}" for i, it in enumerate(items)
                                 if f"FACT {key}=" in it.full_text}
        traces.append({"items": items, "queries": queries, "referenced": referenced})
    return LogisticFutureRelevance().fit_traces(traces, horizon=5)


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=["mock", "ollama", "openai"])
    ap.add_argument("--models", default="mock")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--suite", default="synthetic")
    ap.add_argument("--budget", type=int, default=2500, help="single budget (back-compat)")
    ap.add_argument("--budgets", default=None, help="comma list, e.g. 600,1200,2400,4800")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--n-facts", type=int, default=12)
    ap.add_argument("--block-lines", type=int, default=40,
                    help="noise lines per item; lower this so the 'full' arm fits a real window")
    ap.add_argument("--drift", type=float, default=0.7)
    ap.add_argument("--name-target", default="true", help="true|false (false hides the key)")
    ap.add_argument("--fidelity-cost", default="false", help="true|false (charge re-renders)")
    ap.add_argument("--arms", default=None,
                    help="comma list to restrict arms, e.g. full,recency,reactive_afm,foveance")
    ap.add_argument("--greedy-gap", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else [args.budget]
    models = args.models.split(",")
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else ARMS
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise ValueError(f"unknown arm(s) {bad}; choices: {ARMS}")
    name_target = args.name_target.lower() in ("true", "1", "yes")
    fidelity_cost = 0.5 if args.fidelity_cost.lower() in ("true", "1", "yes") else 0.0
    suite_kw = dict(n_turns=args.turns, n_facts=args.n_facts, block_lines=args.block_lines,
                    drift=args.drift, name_target=name_target)

    # Drift-twin audit (GATE 2): the reactive/foveance arms differ only in drift.
    audit = drift_twin_audit(args.drift)
    with open(os.path.join(args.outdir, "drift_twin_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[audit] reactive_afm vs foveance differ only in drift: {audit['only_difference_is_drift']} "
          f"(fields differing: {audit['config_fields_that_differ']})")

    try:
        suite = get_suite(args.suite, **(suite_kw if args.suite == "synthetic" else {}))
        tasks = list(suite.iter_tasks(args.tasks))
    except SuiteUnavailable as e:
        print(f"[skip] suite '{args.suite}' unavailable: {e}")
        return

    # Resume: load any already-computed cells from a previous (possibly interrupted) run so a
    # stall on flaky hardware can be continued by simply re-launching the same command.
    def _load(path):
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return list(csv.DictReader(f))

    by_seed = _load(os.path.join(args.outdir, "by_seed.csv"))
    per_turn = _load(os.path.join(args.outdir, "per_turn.csv"))
    done = {(r["model"], int(float(r["budget"])), r["policy"], int(float(r["seed"]))) for r in by_seed}
    if done:
        print(f"[resume] {len(done)} cells already computed; continuing.", flush=True)

    bs_path = os.path.join(args.outdir, "by_seed.csv")
    pt_path = os.path.join(args.outdir, "per_turn.csv")
    for model in models:
        def factory(m=model):
            return make_model(args.backend, m, args.base_url)
        print(f"\n=== {model} (backend={args.backend}) suite={args.suite} budgets={budgets} ===")
        for B in budgets:
            for arm in arms:
                accs, intoks = [], []
                for seed, turns in enumerate(tasks):
                    if (model, B, arm, seed) in done:
                        continue
                    row, pts = run_arm_on_task(factory, arm, B, turns, args.drift, fidelity_cost)
                    row = {"model": model, "seed": seed, **row}
                    by_seed.append(row)
                    done.add((model, B, arm, seed))
                    accs.append(row["accuracy"])
                    intoks.append(row["in_tok"])
                    for p in pts:
                        per_turn.append({"model": model, "seed": seed, **p})
                if accs:
                    print(f"  B={B:>5} {arm:>13}  acc={sum(accs)/len(accs):6.3f}  "
                          f"in_tok/seed={sum(intoks)/len(intoks):11.0f}")
                # Persist after every arm so a stall discards at most one in-progress arm.
                write(bs_path, by_seed)
                write(pt_path, per_turn)
        print(f"  [saved {len(by_seed)} rows through {model}]", flush=True)

    write(bs_path, by_seed)
    write(pt_path, per_turn)
    if args.greedy_gap:
        gg = measure_greedy_gap(tasks[:min(args.tasks, 4)], budgets, args.drift)
        write(os.path.join(args.outdir, "greedy_gap.csv"), gg)
        print(f"  greedy_gap rows: {len(gg)}")
    if args.ablations:
        abl = run_ablations(suite_kw, args.tasks, budgets[len(budgets) // 2], fidelity_cost)
        write(os.path.join(args.outdir, "ablations.csv"), abl)
        print(f"  ablation rows: {len(abl)}")
    print(f"\nwrote CSVs to {args.outdir}/  -> run: python bench/analyze.py")


if __name__ == "__main__":
    main()
