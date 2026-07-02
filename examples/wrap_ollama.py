#!/usr/bin/env python3
"""
Wrap a local Ollama model (Gemma/Qwen/Llama) with Foveance anticipatory allocation.

Prereqs: `ollama serve` running and the model pulled (`ollama pull gemma2:9b`).
Run:      python examples/wrap_ollama.py --model gemma2:9b --budget 1500
"""
from __future__ import annotations

import argparse

from foveance import Controller, Item
from foveance.llm import OllamaLLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma2:9b")
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--policy", default="foveance", help="foveance | reactive_afm | full | recency")
    args = ap.parse_args()

    ctrl = Controller(OllamaLLM(model=args.model), budget=args.budget,
                      policy=args.policy, drift=0.7)

    # Plant a fact early, bury it in noise, recall it many turns later.
    ctrl.add_item(Item("obs0", "tool_output",
                       "FACT deploy_token=ZX-9931\n" + "\n".join(f"log {i} ok" for i in range(80)),
                       created_turn=0))
    for t in range(1, 8):
        ctrl.add_item(Item(f"obs{t}", "tool_output",
                           "\n".join(f"log {t}.{i} ok" for i in range(40)), created_turn=t))
        ctrl.step(f"status check {t}", t)

    rec = ctrl.step("recall deploy_token", turn=8)
    print(f"[{args.policy}] answer: {rec.answer}")
    print(f"  input tokens this turn: {rec.input_tokens}  (budget {args.budget})")
    print(f"  re-inflations: {rec.reinflations}")


if __name__ == "__main__":
    main()
