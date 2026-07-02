#!/usr/bin/env python3
"""
Use the Foveance OpenAI-compatible proxy with zero client changes.

This demonstrates the proxy core against a local echo "upstream" (no network/keys needed). In
production you would instead run `foveance proxy --upstream https://api.openai.com/v1` and point
your OpenAI client's base_url at it.

Run: python examples/wrap_openai.py
"""
from __future__ import annotations

from foveance.proxy import FoveanceProxy


def echo_upstream(request: dict) -> dict:
    """Stand-in for a real OpenAI-compatible endpoint: echoes what it received."""
    msgs = request["messages"]
    n_chars = sum(len(m["content"]) for m in msgs)
    return {"choices": [{"message": {"role": "assistant",
                                     "content": f"(upstream saw {len(msgs)} msgs, {n_chars} chars)"}}]}


def main() -> None:
    proxy = FoveanceProxy(budget=200)
    big_log = "FACT customer_id=C-7781\n" + "\n".join(f"event {i} status=ok" for i in range(300))
    request = {
        "model": "gpt-4o-mini",
        "user": "conversation-42",
        "messages": [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": big_log},
            {"role": "assistant", "content": "Logged."},
            {"role": "user", "content": "What is the customer_id?"},
        ],
    }
    print(f"original last-user query preserved; full request was ~{len(big_log)} chars of context")
    resp = proxy.handle(request, echo_upstream)
    print("assistant:", resp["choices"][0]["message"]["content"])
    print("foveance stats:", resp["foveance"])
    print("admin:", proxy.stats())


if __name__ == "__main__":
    main()
