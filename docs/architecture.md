# Architecture

Foveance is a thin, black-box layer that sits between an agent and its model. It never touches
model internals and never fine-tunes; it rewrites the prompt.

```
            add_item(x)                    step(query)
 agent  ───────────────►  MultiFidelityStore  ──►  AnticipatoryPredictor  ──►  index_allocate
   ▲                         (reversible)            v_i(level) curves          (MCKP, O(n log n))
   │                              │                                                   │
   │   answer  ◄── black-box LLM ◄── assemble(levels) ◄──────────────────────────────┘
   └──────────────  retrieve tool re-inflates an item to FULL next turn (two-sided refinement)
```

## Modules
- **`store.py`** — `MultiFidelityStore` holds every item at full fidelity out-of-band and
  *renders* it at POINTER/GIST/DIGEST/FULL on demand. Downgrading is non-destructive, which is
  what makes re-inflation free (Thm. successive refinability).
- **`predictor.py`** — `AnticipatoryPredictor` estimates `p(future needs | history)` as a
  forward-drifting query posterior and scores each item's expected *future* relevance, emitting a
  value curve `v_i(level)`. `drift = 0` recovers the reactive (AFM) criterion.
- **`allocator.py`** — `index_allocate` (deployable Lagrangian/Whittle greedy on concave
  envelopes), `dp_allocate` (exact MCKP oracle), `lp_bound` (LP upper bound). These give the
  `index ≤ OPT ≤ LP` sandwich the greedy-gap experiment measures.
- **`controller.py`** — per-turn orchestration and the policy seam
  (`full`/`recency`/`reactive_afm`/`foveance`/`oracle`). Handles the retrieve tool and the
  fidelity-change cost accounting.
- **`compressors.py` / `embedders.py`** — pluggable renderers (heuristic offline, LLM online) and
  embedders (hashing offline, sentence-transformers / API online).
- **`baselines.py`** — the comparison arms as uniform policies; `reactive_afm` and `foveance` share
  one code path and differ only by the predictor's drift.
- **`learned.py`** — optional logistic future-relevance model fit on logged traces, dropped in
  behind the `FutureRelevancePredictor` interface.
- **`proxy.py`** — OpenAI-compatible reverse proxy applying all of the above transparently.
- **`metrics.py`** — token counting, cost, bootstrap CIs.

## Design rules
- The core (`store`/`predictor`/`allocator`/`controller`) is dependency-free and deterministic
  offline (seeded). ML/proxy/bench dependencies live behind extras.
- Everything is injectable (renderer, embedder, token counter, future model, LLM) for testing.
- No hidden global state; no network in unit tests.
