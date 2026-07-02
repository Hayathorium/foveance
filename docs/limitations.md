# Limitations (honest)

We keep this page blunt; reviewers in this area know the near-neighbours.

## The headline can tie the baseline — by design
On low-drift trajectories, or when re-inflation is free and the reactive policy recomputes each
turn, `foveance ≈ reactive_afm`. This is **not** a bug: Thm. "locality gap" predicts it
(`Regret(myopic) ≤ c₁(1−φ)`). The deliverable is the *decision rule* for when the cheap heuristic
suffices, not a universal win. We report the ties (`Δacc` with CI and Wilcoxon p-value) instead of
hiding them.

## The offline suite is synthetic
The needle-reuse-with-drift suite is engineered to *isolate* anticipation, not to stand in for
real agents. Real-model Pareto frontiers (Gemma/Qwen/Llama) and agentic suites
(AppWorld/OfficeBench, LongBench/RULER) are run by `scripts/run_everything.sh`; their adapters
skip cleanly when the datasets are absent and never fabricate rows.

## Task-success distortion assumes a gradable answer
The theory uses `1 − success` distortion. Open-ended generation needs a surrogate metric (e.g. an
LLM judge), which loosens the tightness of the bound, though not its direction.

## Successive refinability assumes nesting
Reversible re-inflation is rate-free only when renders are nested (POINTER ⊂ GIST ⊂ DIGEST ⊂
FULL as a Markov chain). Non-nested LLM digests incur a real penalty `Δ(f,f')`; we model and price
it but do not claim to estimate it perfectly.

## Predictor overhead and estimation error
The anticipatory posterior adds an embedding and an `O(n log n)` allocation per turn — negligible
vs a model call — but a miscalibrated relevance estimator erodes the gain (Lemma: realized value
within `O(ε·n)` of the oracle). A badly wrong predictor degrades toward the reactive baseline
rather than below it.

## Agentic (tool-using) requests are compressed structurally, not by collapsing
A live test routing **Claude Code** through the proxy first showed that *collapsing* an agentic
request (it declares a `tools` array and relies on strict `tool_use`/`tool_result` pairing and
`cache_control`) makes the provider reject it (HTTP 400). The proxy therefore compresses such
requests **structure-preservingly** instead: it detects any request carrying `tools`/`tool_choice`
(or tool blocks), keeps every message, role, and tool_use/tool_result pair intact, protects the most
recent turns (`agentic_protect_last`), and digests only large stale content blocks (big tool
outputs) in older turns (`reason: "agentic-inplace"`). The result is always a valid request, so
agents like Claude Code keep working (verified live), and large old tool output is trimmed (~71%
fewer input tokens on an 8-tool-call transcript in our offline measurement). Two honest caveats:
(1) digesting is lossy — an elided marker tells the model context was trimmed, so set
`agentic_protect_last` high enough that the turns the next step needs are kept full; (2) modifying
old blocks busts the provider's prompt cache for those blocks, so on cache-heavy workloads measure
net cost. The proxy also surfaces the upstream's real status/error body rather than failing
internally. Anticipatory, relevance-ranked selection of which old blocks to keep full is the next
refinement.

### Cache-aware mode (`--cache-aware`) turns caveat (2) into a switch
With `--cache-aware`, the proxy never modifies content at or before the last explicit Anthropic
`cache_control` breakpoint (and a block carrying `cache_control` is never modified in any mode),
so the provider's prompt-cache prefix is never invalidated by the proxy. The arithmetic for when
to flip it: with Anthropic pricing, cached input reads cost 0.1× fresh input, so re-reading a
cached prefix of `P` tokens costs like `0.1·P` fresh tokens — busting it to digest away `S`
tokens pays only when `S > 0.9·P` over the remaining turns that would have hit the cache.
Rule of thumb: **cache-aware on** for long-lived API-billed agent sessions with stable prefixes;
**off** (default) for local models, short sessions, or when the raw context length itself is the
constraint (small context windows, latency).

## Not claimed as novel
The multi-fidelity-store-under-a-budget mechanism (AFM, ContextBudget, ACON, MemAct). We use it as
substrate and say so everywhere.
