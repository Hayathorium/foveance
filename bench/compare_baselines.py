#!/usr/bin/env python3
"""Fast single-shot baseline comparison on a real model.

For each policy we build one long trajectory with a load-bearing fact planted early amid filler,
apply that policy's allocation under a fixed token budget, render the context, and make ONE model
call asking to recall the fact. This isolates *which methods keep the needle under the budget* with
far fewer model calls than the full trajectory harness, so it runs in minutes on a CPU.

It reports real recall accuracy and the model's real input-token count per method, and writes
bench/results_baselines/single_shot.csv. Honest by construction: whatever the model returns is what
is recorded. Usage: python bench/compare_baselines.py --model llama3.2:1b --budget 500 --seeds 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import urllib.request

from foveance import baselines
from foveance.embedders import HashingEmbedder
from foveance.predictor import AnticipatoryPredictor, PredictorConfig
from foveance.store import Item, MultiFidelityStore

ARMS = ["full", "recency", "truncate", "uniform", "reactive_afm", "foveance"]

_LLMLINGUA = {}


def llmlingua2_compress(text: str, target_token: int) -> str:
    """Compress raw context with the real LLMLingua-2 (query-agnostic). Lazily loaded/cached."""
    if "c" not in _LLMLINGUA:
        from llmlingua import PromptCompressor  # type: ignore
        _LLMLINGUA["c"] = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True,
            device_map="cpu")
    out = _LLMLINGUA["c"].compress_prompt(text, target_token=max(50, target_token))
    return out["compressed_prompt"]


def ollama(prompt: str, model: str, host: str = "http://localhost:11434") -> tuple[str, int]:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(host + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d.get("response", ""), int(d.get("prompt_eval_count", 0))


def build_items(rng: random.Random, secret: str, n_turns: int, lines: int) -> list[Item]:
    items = [Item("o0", "tool_output",
                  f"API_KEY={secret}\n" + "\n".join(f"log 0.{j} status=ok" for j in range(lines)), 0)]
    for t in range(1, n_turns):
        body = "\n".join(f"log {t}.{j} status=ok path=/srv/{rng.randint(1, 99)} lat={rng.randint(1, 400)}ms"
                         for j in range(lines))
        items.append(Item(f"o{t}", "tool_output", body, t))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:1b")
    ap.add_argument("--budgets", default="250,350,500")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--lines", type=int, default=18)
    ap.add_argument("--drift", type=float, default=0.7)
    ap.add_argument("--with-llmlingua", action="store_true",
                    help="also run the real LLMLingua-2 arm (needs the llmlingua package)")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    arms = ARMS + (["llmlingua2"] if args.with_llmlingua else [])

    query = "What is the API_KEY value from the earlier log? Reply with only the value."
    rows = []
    for budget in budgets:
        agg = {a: {"correct": 0, "n": 0, "tok": 0} for a in arms}
        for seed in range(args.seeds):
            rng = random.Random(seed)
            secret = f"SECRET-{rng.randint(1000, 9999)}"
            items = build_items(rng, secret, args.turns, args.lines)
            for arm in arms:
                if arm == "llmlingua2":
                    full_text = "\n".join(it.full_text for it in items)
                    try:
                        ctx = llmlingua2_compress(full_text, budget)
                    except Exception as e:  # keep the run honest: record a miss with a note
                        print(f"  llmlingua2 unavailable: {e}", flush=True)
                        continue
                else:
                    store = MultiFidelityStore()
                    for it in items:
                        store.add(it)
                    drift = 0.0 if arm == "reactive_afm" else args.drift
                    pred = AnticipatoryPredictor(store, HashingEmbedder(),
                                                 config=PredictorConfig(drift=drift))
                    pred.observe_query(query)
                    levels = baselines.POLICIES[arm](store, pred, budget, len(items))
                    ctx, _ = store.assemble(levels, system="Context (may be compressed):")
                ans, ptok = ollama(ctx + "\n\n" + query, args.model)
                ok = secret in (ans or "")
                agg[arm]["correct"] += int(ok)
                agg[arm]["n"] += 1
                agg[arm]["tok"] += ptok
                print(f"  B={budget} seed{seed} {arm:>13}  {'OK ' if ok else 'MISS'}  "
                      f"in_tok={ptok:5d}", flush=True)
        for a in arms:
            acc = agg[a]["correct"] / max(1, agg[a]["n"])
            tok = agg[a]["tok"] / max(1, agg[a]["n"])
            rows.append({"budget": budget, "method": a, "accuracy": round(acc, 3),
                         "avg_in_tok": round(tok, 1), "n": agg[a]["n"]})

    outdir = os.path.join(os.path.dirname(__file__), "results_baselines")
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== single-shot recall on {args.model}, {args.seeds} seeds ===")
    print(f"{'budget':>7} {'method':>14}  {'accuracy':>8}  {'avg in_tok':>10}")
    for r in rows:
        print(f"{r['budget']:>7} {r['method']:>14}  {r['accuracy']:8.3f}  {r['avg_in_tok']:10.0f}")
    with open(os.path.join(outdir, "single_shot.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["budget", "method", "accuracy", "avg_in_tok", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {os.path.join(outdir, 'single_shot.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
