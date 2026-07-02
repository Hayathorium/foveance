"""
Token counting, cost, and statistics helpers.

Token counts come from the provider where possible (Ollama ``prompt_eval_count`` /
``eval_count``, OpenAI ``usage``); this module is the *fallback* counter and the place where
cost/latency/efficiency aggregates and bootstrap CIs live. Everything is numpy-optional so the
core stays dependency-free; ``[bench]`` pulls in numpy/tiktoken for the real harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


# ----------------------------------------------------------------------------- token counting
def whitespace_counter(text: str) -> int:
    """Cheapest offline counter: ~4 chars/token, min 1. Deterministic, no deps."""
    return max(1, len(text) // 4)


def make_token_counter(encoding: str = "cl100k_base") -> Callable[[str], int]:
    """Return a ``str -> int`` token counter.

    Tries tiktoken (``cl100k_base``/``o200k_base``), then a HuggingFace tokenizer name, then
    falls back to the whitespace/4-char heuristic. The fallback keeps CI offline-safe.
    """
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding(encoding)
        return lambda s: len(enc.encode(s))
    except Exception:
        pass
    try:  # pragma: no cover - optional HF path
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(encoding)
        return lambda s: len(tok.encode(s))
    except Exception:
        return whitespace_counter


# ------------------------------------------------------------------------------------- cost
@dataclass
class CostModel:
    """USD cost from token counts. Rates are per 1M tokens (provider price sheets)."""

    input_per_m: float = 0.0
    output_per_m: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_m + output_tokens * self.output_per_m) / 1e6


# -------------------------------------------------------------------------------- aggregates
def tokens_per_correct(input_tokens: int, n_correct: int) -> float:
    """Headline efficiency: prompt tokens spent per correct answer (lower is better)."""
    return input_tokens / n_correct if n_correct > 0 else float("inf")


def peak_tokens(per_turn_peaks: Sequence[int]) -> int:
    return max(per_turn_peaks, default=0)


def bootstrap_ci(
    xs: Sequence[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(mean, lo, hi) bootstrap CI. Uses numpy when available, else a pure-python resampler."""
    xs = list(xs)
    if not xs:
        return (float("nan"), float("nan"), float("nan"))
    try:
        import numpy as np

        a = np.asarray(xs, float)
        rng = np.random.default_rng(seed)
        means = a[rng.integers(0, len(a), size=(n_resamples, len(a)))].mean(1)
        return (float(a.mean()), float(np.quantile(means, alpha / 2)),
                float(np.quantile(means, 1 - alpha / 2)))
    except Exception:
        import random

        prng = random.Random(seed)
        means = []
        for _ in range(min(n_resamples, 2000)):
            sample = [xs[prng.randrange(len(xs))] for _ in xs]
            means.append(sum(sample) / len(sample))
        means.sort()
        lo = means[int((alpha / 2) * len(means))]
        hi = means[int((1 - alpha / 2) * len(means)) - 1]
        return (sum(xs) / len(xs), lo, hi)


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


@dataclass
class LatencyStats:
    mean_s: float
    lo_s: float
    hi_s: float
    ms_per_turn: float


def latency_stats(per_turn_latencies: Sequence[float]) -> LatencyStats:
    m, lo, hi = bootstrap_ci(list(per_turn_latencies))
    return LatencyStats(m, lo, hi, 1000.0 * (mean(per_turn_latencies) or 0.0))
