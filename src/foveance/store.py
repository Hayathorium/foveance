"""
Multi-fidelity, reversible context store.

Each context item is held at full fidelity out-of-band (never destroyed) and can be
*rendered* at one of several fidelity levels into the live prompt. This is what makes
re-inflation (two-sided refinement) possible: downgrading an item is not destructive,
so a later turn can upgrade it again at zero information loss from the store's side.

NOTE ON NOVELTY / PRIOR ART (read docs/NOVELTY.md):
  Per-message fidelity *tiers* under a token budget already exist (AFM, Cruz 2025).
  This module is deliberately a thin, standard substrate. Foveance's contribution is NOT
  the tiered store; it is the *anticipatory* allocation policy (predictor.py + allocator.py)
  and the accompanying rate-distortion theory. Keep that boundary honest in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional
import hashlib


class Fidelity(IntEnum):
    POINTER = 0      # a short addressable stub, e.g. "[obs#7: ls /src (42 files) -> see store]"
    GIST = 1         # one-line natural-language gist
    DIGEST = 2       # structured / lossy-but-rich digest (e.g. JSON schema + key rows)
    FULL = 3         # verbatim original


@dataclass
class Item:
    """A single context element (tool output, message, retrieved doc, reasoning chunk)."""
    item_id: str
    kind: str                                   # "tool_output" | "user" | "assistant" | "doc" | "reasoning"
    full_text: str                              # canonical content, retained for the item's lifetime
    created_turn: int
    renders: dict = field(default_factory=dict) # cache: Fidelity -> rendered string
    embedding: Optional[list] = None            # filled lazily by the predictor
    last_referenced_turn: int = -1              # updated when the model retrieves/uses it
    meta: dict = field(default_factory=dict)

    def content_hash(self) -> str:
        return hashlib.sha1(self.full_text.encode("utf-8")).hexdigest()[:12]


# A renderer turns (item, fidelity) -> string. Default renderers are heuristic and offline;
# production swaps in an LLM compressor for GIST/DIGEST (see compressors.py).
Renderer = Callable[[Item, Fidelity], str]


def default_renderer(item: Item, level: Fidelity) -> str:
    text = item.full_text
    if level == Fidelity.FULL:
        return text
    if level == Fidelity.DIGEST:
        # structural digest: preserve SALIENT lines (key=value facts, ALL-CAPS markers,
        # error/decision lines) and drop boilerplate. This is the premise of a "rich but
        # lossy" tier -- a digest must keep the facts a later step might need.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) <= 6:
            return text

        def salient(ln: str) -> bool:
            up = ln.strip()
            return ("=" in up and not up.startswith("log")) or \
                   any(k in up for k in ("FACT", "ERROR", "DECISION", "WARN", "TODO")) or \
                   up.isupper()

        keep = [ln for ln in lines if salient(ln)]
        head, tail = lines[0], lines[-1]
        body = "\n".join(dict.fromkeys([head] + keep + [tail]))  # dedupe, preserve order
        omitted = len(lines) - len(set([head, tail]) | set(keep))
        return f"{body}\n... [{max(0,omitted)} boilerplate lines omitted; full in store#{item.item_id}] ..."
    if level == Fidelity.GIST:
        first = text.strip().splitlines()[0] if text.strip() else ""
        first = first[:160]
        return f"[{item.kind}#{item.item_id}] {first} (+{max(0,len(text)-len(first))} chars in store)"
    # POINTER
    return f"[{item.kind}#{item.item_id} | {item.content_hash()} | retrievable]"


class MultiFidelityStore:
    def __init__(self, renderer: Renderer = default_renderer,
                 token_counter: Optional[Callable[[str], int]] = None):
        self.items: dict[str, Item] = {}
        self.order: list[str] = []            # chronological
        self.renderer = renderer
        self._count = token_counter or (lambda s: max(1, len(s) // 4))  # ~4 chars/token fallback

    def add(self, item: Item) -> None:
        if item.item_id in self.items:
            raise KeyError(f"duplicate item_id {item.item_id}")
        self.items[item.item_id] = item
        self.order.append(item.item_id)

    def render(self, item_id: str, level: Fidelity) -> str:
        item = self.items[item_id]
        if level not in item.renders:
            item.renders[level] = self.renderer(item, level)
        return item.renders[level]

    def cost(self, item_id: str, level: Fidelity) -> int:
        """Token cost of rendering item at `level`."""
        return self._count(self.render(item_id, level))

    def retrieve_full(self, item_id: str, turn: int) -> str:
        """Explicit re-inflation hook (the LLM's 'saccade' / retrieve tool)."""
        self.items[item_id].last_referenced_turn = turn
        return self.items[item_id].full_text

    def assemble(self, levels: dict[str, Fidelity], system: str = "") -> tuple[str, int]:
        """Build the live context string from a per-item fidelity assignment."""
        parts = [system] if system else []
        for iid in self.order:
            lvl = levels.get(iid, Fidelity.POINTER)
            parts.append(self.render(iid, lvl))
        ctx = "\n".join(p for p in parts if p)
        return ctx, self._count(ctx)
