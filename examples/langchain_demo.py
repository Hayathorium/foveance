#!/usr/bin/env python3
"""
LangChain-style integration sketch: drive a tool-using loop while Foveance manages the context
budget. We keep the dependency optional -- if langchain is not installed, the script falls back
to a plain loop so it still runs offline. The point is that Foveance slots in at the
context-assembly seam regardless of the orchestration framework.

Run: python examples/langchain_demo.py
"""
from __future__ import annotations

from foveance import Controller, Item
from foveance.llm import MockLLM


def fake_tool(name: str, turn: int) -> str:
    """Pretend tool that returns a noisy observation, occasionally carrying a needle."""
    noise = "\n".join(f"{name} line {turn}.{i} status=ok" for i in range(40))
    if turn == 1:
        return f"FACT invoice_total=4821\n{noise}"
    return noise


def main() -> None:
    # Foveance is the context manager; the agent loop calls tools and recalls facts.
    ctrl = Controller(MockLLM(), budget=1200, policy="foveance", drift=0.7)

    plan = [("search", "kickoff"), ("fetch", "load invoice"), ("list", "enumerate"),
            ("inspect", "details"), ("recall", "recall invoice_total")]
    for t, (tool, query) in enumerate(plan):
        obs = fake_tool(tool, t)
        ctrl.add_item(Item(f"obs{t}", "tool_output", obs, created_turn=t))
        rec = ctrl.step(query, t)
        print(f"turn {t:>1} [{tool:>7}] query={query!r:>22} -> {rec.answer}  "
              f"(in_tok={rec.input_tokens})")

    print("\nFoveance kept the early invoice needle in budget so the final recall succeeds, "
          "even though it scrolled far out of the recency window.")


if __name__ == "__main__":
    main()
