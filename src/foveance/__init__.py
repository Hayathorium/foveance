"""Foveance: anticipatory context allocation for long-horizon LLM agents.

Public API:
    from foveance import Controller, Item, Fidelity, MultiFidelityStore
    from foveance import AnticipatoryPredictor, PredictorConfig
    from foveance import index_allocate, dp_allocate, lp_bound
    from foveance.llm import MockLLM, OllamaLLM, OpenAICompatLLM
    from foveance.embedders import HashingEmbedder
    from foveance.compressors import HeuristicCompressor, LLMCompressor, make_renderer
    from foveance.proxy import FoveanceProxy
    from foveance import baselines, metrics

See docs/NOVELTY.md for the honest prior-art positioning:
the multi-fidelity store under a budget is substrate (AFM); Foveance's contribution is the
*anticipatory* allocation policy, the index allocator + greedy-gap result, two-sided
refinement, and the rate-distortion theory.
"""
from .store import MultiFidelityStore, Item, Fidelity, default_renderer
from .predictor import (
    AnticipatoryPredictor,
    PredictorConfig,
    FutureRelevancePredictor,
    PredictorContext,
)
from .allocator import index_allocate, dp_allocate, lp_bound
from .controller import Controller, RunResult, TurnRecord
from .embedders import HashingEmbedder, Embedder, cosine
from . import baselines, metrics

__all__ = [
    "MultiFidelityStore", "Item", "Fidelity", "default_renderer",
    "AnticipatoryPredictor", "PredictorConfig", "FutureRelevancePredictor", "PredictorContext",
    "index_allocate", "dp_allocate", "lp_bound",
    "Controller", "RunResult", "TurnRecord",
    "HashingEmbedder", "Embedder", "cosine",
    "baselines", "metrics",
]
__version__ = "0.1.0"
