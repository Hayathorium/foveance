#!/usr/bin/env python3
"""
Fetch a small LongBench slice into the JSONL layout the ``LongBenchSuite`` adapter expects, then
point ``FOVEANCE_LONGBENCH_PATH`` at the output directory.

Each output line is ``{"context": str, "input": str, "answers": [str]}`` (the adapter chunks
``context`` into store items and treats the final turn as the graded query).

Usage:
    python bench/fetch_longbench.py --task narrativeqa --n 5 --out data/longbench
    FOVEANCE_LONGBENCH_PATH=data/longbench python bench/run_bench.py \
        --backend ollama --models gemma2:2b --suite longbench --budgets 1200,3000 --tasks 5

Requires the optional dependency ``datasets`` (``pip install datasets``). Network access needed.
LongBench contexts are long, so meaningful accuracy needs a real model (ideally a GPU box).
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="narrativeqa",
                    help="LongBench subtask (e.g. narrativeqa, qasper, hotpotqa, multifieldqa_en)")
    ap.add_argument("--n", type=int, default=5, help="number of examples to fetch")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "longbench"))
    ap.add_argument("--dataset", default="THUDM/LongBench")
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        print("This fetcher needs `datasets`: pip install datasets")
        return 2

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.task}.jsonl")
    n = 0
    try:
        if args.dataset in ("hotpot_qa", "hotpotqa", "hotpotqa/hotpot_qa"):
            # Clean parquet multi-doc QA: context = the supporting paragraphs (no loader script).
            ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
            with open(path, "w", encoding="utf-8") as f:
                for rec in ds:
                    titles = rec["context"]["title"]
                    sents = rec["context"]["sentences"]
                    ctx = "\n\n".join(f"{t}: {' '.join(s)}" for t, s in zip(titles, sents))
                    row = {"context": ctx, "input": rec["question"], "answers": [rec["answer"]]}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
                    if n >= args.n:
                        break
        else:
            ds = load_dataset(args.dataset, args.task, split="test", streaming=True)
            with open(path, "w", encoding="utf-8") as f:
                for rec in ds:
                    answers = rec.get("answers") or ([rec["answer"]] if rec.get("answer") else [""])
                    row = {"context": rec.get("context", ""),
                           "input": rec.get("input") or rec.get("question", ""),
                           "answers": list(answers)}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
                    if n >= args.n:
                        break
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            print("This dataset uses a loader script unsupported by datasets>=3. Either:\n"
                  "  pip install 'datasets<3'   (then re-run for THUDM/LongBench), or\n"
                  "  use the clean parquet path: python bench/fetch_longbench.py --dataset hotpot_qa")
            return 2
        raise
    print(f"wrote {n} examples to {path}")
    print(f"now run: FOVEANCE_LONGBENCH_PATH={args.out} python bench/run_bench.py "
          f"--backend ollama --models gemma2:2b --suite longbench --budgets 1200,3000 --tasks {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
