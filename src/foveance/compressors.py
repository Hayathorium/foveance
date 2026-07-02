"""
Fidelity renderers ("compressors"): map (item, fidelity) -> rendered string.

Two implementations behind the ``store.Renderer`` callable seam:

* ``HeuristicCompressor`` -- offline, deterministic, dependency-free. Wraps the seed's
  structural digest (preserve key=value facts / ERROR / DECISION / WARN / TODO / ALL-CAPS,
  drop boilerplate). This is the default and what CI/tests use.
* ``LLMCompressor`` -- uses a small LLM to produce faithful GIST/DIGEST renders. It is
  content-hash cached and prompted to never invent facts (faithfulness > fluency).

``make_renderer`` picks heuristic offline, LLM when a model is supplied.

NOTE ON NOVELTY (docs/NOVELTY.md): the renderer ladder is *substrate* shared with AFM and
the prompt-compression literature. Foveance's contribution is the anticipatory allocation over
these renders, not the renders themselves.
"""
from __future__ import annotations

from typing import Callable, Optional

from .store import Item, Fidelity, default_renderer


class HeuristicCompressor:
    """Offline structural renderer (the seed's logic), exposed as a ``Renderer``."""

    def __call__(self, item: Item, level: Fidelity) -> str:
        return default_renderer(item, level)


# Default per-level instructions for an LLM compressor. Faithfulness is the hard constraint.
DEFAULT_LEVEL_PROMPTS: dict[Fidelity, str] = {
    Fidelity.GIST: (
        "Summarize the following content in ONE short line. Preserve any concrete "
        "identifiers (keys, values, names, error codes). Do NOT invent facts.\n\n{body}"
    ),
    Fidelity.DIGEST: (
        "Produce a compact digest of the following content. KEEP every salient fact "
        "(key=value pairs, ERROR/DECISION/WARN/TODO lines, numbers, identifiers) and drop "
        "boilerplate/log noise. Do NOT invent or alter facts.\n\n{body}"
    ),
}


class LLMCompressor:
    """Render GIST/DIGEST with a small LLM; FULL/POINTER stay heuristic. Cached by content.

    ``model`` is any object exposing ``generate(prompt, query) -> Completion`` (the
    ``foveance.llm.LLM`` interface) or a bare ``Callable[[str], str]``. Renders are cached on
    ``(content_hash, level)`` so the same item is never recompressed, keeping the renderer
    idempotent and cheap.
    """

    def __init__(
        self,
        model: object,
        level_prompts: Optional[dict[Fidelity, str]] = None,
        fallback: Optional[Callable[[Item, Fidelity], str]] = None,
    ) -> None:
        self.model = model
        self.level_prompts = level_prompts or DEFAULT_LEVEL_PROMPTS
        self.fallback = fallback or default_renderer
        self._cache: dict[tuple[str, int], str] = {}

    def _call(self, prompt: str) -> str:
        m = self.model
        if hasattr(m, "generate"):
            return m.generate(prompt, "compress").text  # type: ignore[attr-defined]
        if callable(m):
            return m(prompt)  # type: ignore[misc]
        raise TypeError("LLMCompressor.model must have .generate or be callable")

    def __call__(self, item: Item, level: Fidelity) -> str:
        if level not in self.level_prompts:
            return self.fallback(item, level)
        key = (item.content_hash(), int(level))
        if key in self._cache:
            return self._cache[key]
        prompt = self.level_prompts[level].format(body=item.full_text)
        try:
            out = self._call(prompt).strip()
            if not out:  # empty/failed render -> heuristic, never an empty context slot
                out = self.fallback(item, level)
        except Exception:  # network/model failure must not crash allocation
            out = self.fallback(item, level)
        self._cache[key] = out
        return out


def make_renderer(model: object | None = None, **kwargs) -> Callable[[Item, Fidelity], str]:
    """Renderer factory: heuristic when ``model is None`` (offline), else LLM-backed."""
    if model is None:
        return HeuristicCompressor()
    return LLMCompressor(model, **kwargs)
