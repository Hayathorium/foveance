"""
Task suites for the Foveance benchmark.

An abstract ``Suite`` yields, per task/seed, an ordered list of ``Turn``s; each turn carries a
``query`` (what the model answers, used for grading), a ``gold`` answer, ``new_items`` to add
to the store, and an anticipation ``probe`` (what the *predictor* observes -- defaults to the
query, but differs under ``name_target=False`` so the query no longer lexically reveals the
target and the policy must rely on the anticipatory posterior).

* ``SyntheticSuite`` -- offline needle-reuse-with-drift; the engine for CI and the ablations.
  Knobs: n_turns, n_facts, block_lines, drift (cross-turn dependency), distractor_rate,
  name_target. A fidelity-change cost is applied by the controller (``fidelity_cost``), making
  pre-staging (anticipation) measurably cheaper.
* ``LongBenchSuite`` / ``RulerSuite`` -- real long-context QA adapters.
* ``AppWorldSuite`` / ``OfficeBenchSuite`` -- agentic tool-use adapters (the suites ACON used).

The real-dataset adapters raise ``SuiteUnavailable`` (with fetch instructions) when the data is
absent, so the harness skips them gracefully and never fabricates results.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

from foveance import Item


@dataclass
class Turn:
    query: str
    gold: str
    new_items: list = field(default_factory=list)  # list[Item]
    probe: str = ""                                 # predictor cue; "" -> use query


class SuiteUnavailable(Exception):
    """Raised when a real dataset is not present; the harness skips the suite cleanly."""


class Suite:
    """Abstract task suite. ``iter_tasks`` yields one ``list[Turn]`` per task/seed."""

    name = "abstract"

    def iter_tasks(self, n_tasks: int):  # pragma: no cover - interface
        raise NotImplementedError


# ----------------------------------------------------------------------- synthetic suite
def _noisy_block(turn: int, n_lines: int, rng: random.Random, needle: str = "") -> str:
    lines = [f"log line {turn}.{j} status=ok latency={rng.randint(1, 400)}ms "
             f"path=/srv/{rng.choice(['a', 'b', 'c'])}/{rng.randint(100, 999)}"
             for j in range(n_lines)]
    if needle:
        lines.insert(rng.randint(0, len(lines)), needle)
    return "\n".join(lines)


def make_task(seed: int, n_turns: int = 40, n_facts: int = 12, block_lines: int = 40,
              drift: float = 0.7, distractor_rate: float = 1.0,
              name_target: bool = True) -> list:
    """Return a list[Turn]: ~first third plants facts; the rest recalls them with drift."""
    rng = random.Random(seed)
    keys = [f"k{i}" for i in range(n_facts)]
    vals = {k: f"v{rng.randint(1000, 9999)}" for k in keys}
    topic = {k: f"topic{i}" for i, k in enumerate(keys)}
    n_lines = max(1, int(block_lines * distractor_rate))

    turns: list[Turn] = []
    plant_turns = max(n_facts, n_turns // 3)

    # Phase 1: plant facts inside noisy tool outputs (no graded queries yet).
    for t in range(plant_turns):
        k = keys[t % n_facts]
        needle = f"FACT {k}={vals[k]} {topic[k]}"
        item = Item(item_id=f"obs{t}", kind="tool_output",
                    full_text=_noisy_block(t, n_lines, rng, needle), created_turn=t)
        probe = f"note {k}" if name_target else f"working on {topic[k]}"
        turns.append(Turn(query=f"note {k}", gold="", new_items=[item], probe=probe))

    # Phase 2: recall with drift -- next key correlated with previous (random walk over index).
    idx = 0
    for t in range(plant_turns, n_turns):
        if rng.random() < drift:
            idx = (idx + rng.choice([0, 1, 1, 2])) % n_facts   # forward drift
        else:
            idx = rng.randrange(n_facts)                       # occasional jump
        k = keys[idx]
        item = Item(item_id=f"obs{t}", kind="tool_output",
                    full_text=_noisy_block(t, n_lines, rng), created_turn=t)
        # The query names the key (so the model can be graded); under name_target=False the
        # *probe* the predictor sees does not, removing the lexical shortcut.
        probe = f"recall {k}" if name_target else f"working on {topic[k]}"
        turns.append(Turn(query=f"recall {k}", gold=vals[k], new_items=[item], probe=probe))

    return turns


class SyntheticSuite(Suite):
    name = "synthetic"

    def __init__(self, n_turns: int = 40, n_facts: int = 12, block_lines: int = 40,
                 drift: float = 0.7, distractor_rate: float = 1.0, name_target: bool = True):
        self.kw = dict(n_turns=n_turns, n_facts=n_facts, block_lines=block_lines, drift=drift,
                       distractor_rate=distractor_rate, name_target=name_target)

    def iter_tasks(self, n_tasks: int):
        for seed in range(n_tasks):
            yield make_task(seed, **self.kw)


def score(answer: str, gold: str) -> float:
    if not gold:
        return 1.0  # planting / non-recall turns don't count against accuracy
    return 1.0 if gold in (answer or "") else 0.0


# -------------------------------------------------------------- real-dataset adapters
def _require_path(env_var: str, human: str, fetch: str) -> str:
    path = os.environ.get(env_var)
    if not path or not os.path.exists(path):
        raise SuiteUnavailable(
            f"{human} not found (set {env_var} to its path). Fetch: {fetch}")
    return path


class LongBenchSuite(Suite):
    """LongBench v2 multi-doc QA. Set FOVEANCE_LONGBENCH_PATH to a prepared JSONL directory."""

    name = "longbench"

    def __init__(self):
        self.path = _require_path(
            "FOVEANCE_LONGBENCH_PATH", "LongBench",
            "https://github.com/THUDM/LongBench (download, then point the env var at it)")

    def iter_tasks(self, n_tasks: int):  # pragma: no cover - needs dataset
        import json
        files = sorted(f for f in os.listdir(self.path) if f.endswith(".jsonl"))
        count = 0
        for fn in files:
            with open(os.path.join(self.path, fn), encoding="utf-8") as fh:
                for line in fh:
                    if count >= n_tasks:
                        return
                    rec = json.loads(line)
                    ctx = rec.get("context", "")
                    chunks = [ctx[i:i + 2000] for i in range(0, len(ctx), 2000)] or [""]
                    items = [Item(f"doc{i}", "doc", c, 0) for i, c in enumerate(chunks)]
                    q = rec.get("input") or rec.get("question", "")
                    gold = (rec.get("answers") or [rec.get("answer", "")])[0]
                    turns = [Turn(query="ingest", gold="", new_items=[it]) for it in items]
                    turns.append(Turn(query=q, gold=str(gold), new_items=[]))
                    yield turns
                    count += 1


class RulerSuite(Suite):
    """RULER needle-in-haystack. Set FOVEANCE_RULER_PATH to a prepared JSONL directory."""

    name = "ruler"

    def __init__(self):
        self.path = _require_path(
            "FOVEANCE_RULER_PATH", "RULER",
            "https://github.com/NVIDIA/RULER (generate the synthetic data, then set the env var)")

    def iter_tasks(self, n_tasks: int):  # pragma: no cover - needs dataset
        return LongBenchSuite.iter_tasks(self, n_tasks)  # same JSONL shape


class AppWorldSuite(Suite):
    """AppWorld agentic trajectories. Set FOVEANCE_APPWORLD_PATH (install the appworld package)."""

    name = "appworld"

    def __init__(self):
        self.path = _require_path(
            "FOVEANCE_APPWORLD_PATH", "AppWorld",
            "https://github.com/StonyBrookNLP/appworld (pip install appworld; set env var to data)")

    def iter_tasks(self, n_tasks: int):  # pragma: no cover - needs dataset
        raise SuiteUnavailable("AppWorld adapter present; wire task replay to your install.")


class OfficeBenchSuite(Suite):
    """OfficeBench agentic trajectories. Set FOVEANCE_OFFICEBENCH_PATH."""

    name = "officebench"

    def __init__(self):
        self.path = _require_path(
            "FOVEANCE_OFFICEBENCH_PATH", "OfficeBench",
            "https://github.com/zhuohaoyu/OfficeBench (download; set the env var)")

    def iter_tasks(self, n_tasks: int):  # pragma: no cover - needs dataset
        raise SuiteUnavailable("OfficeBench adapter present; wire task replay to your install.")


_SUITES = {
    "synthetic": SyntheticSuite,
    "longbench": LongBenchSuite,
    "ruler": RulerSuite,
    "appworld": AppWorldSuite,
    "officebench": OfficeBenchSuite,
}


def get_suite(name: str, **kwargs) -> Suite:
    """Instantiate a suite by name; raises ``SuiteUnavailable`` if its data is absent."""
    if name not in _SUITES:
        raise ValueError(f"unknown suite {name!r}; choices: {sorted(_SUITES)}")
    if name == "synthetic":
        return SyntheticSuite(**kwargs)
    return _SUITES[name]()
