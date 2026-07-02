# Baselines & policy arms

All arms share the same store, budget, model, predictor machinery, and tasks, so any difference
isolates the policy. They implement one signature: `policy(store, predictor, budget, turn) ->
{item_id: Fidelity}` (see `foveance.baselines`).

| Arm | What it does | Role |
|---|---|---|
| `full` | every item at FULL | accuracy ceiling, token ceiling |
| `recency` | FULL for the last *k* items, POINTER otherwise | cheap myopic control |
| `reactive_afm` | AFM-style: score by the **current** query (`drift=0`) + half-life recency + kind importance, pack under budget with the index allocator | **primary baseline** = `drift=0` special case |
| `foveance` | anticipatory: same machinery, predictor `drift>0` scores by the **future**-query posterior | our method |
| `oracle` | exact DP allocation on the foveance value curves | greedy-gap upper bound |
| `llmlingua2` | LLMLingua-2 prompt compression (optional, `[bench]` + `llmlingua`) | external compressor |
| `lp_bound` | LP relaxation value, no model call | theoretical frontier point |

## The crucial invariant
`reactive_afm` and `foveance` differ **only** in the predictor's `drift`. The benchmark audits this
every run and writes `bench/results/drift_twin_audit.json`:
```json
{ "config_fields_that_differ": ["drift"], "only_difference_is_drift": true }
```
This is what licenses attributing any measured difference to *anticipation* and nothing else
(see [`NOVELTY.md`](NOVELTY.md)). The multi-fidelity store under a budget is prior art (AFM); shipping
`reactive_afm` as a first-class arm means the package *contains* the comparison rather than merely
asserting it.

## Why reactive often ties foveance offline
On the easy synthetic regime — named targets, free re-inflation, per-turn recompute — the reactive
policy can simply re-solve for the current query each turn, and Thm. "locality gap" says that is
near-optimal. The separation appears with `--name-target false` (the query hides the key) and
`--fidelity-cost true` (raising fidelity costs re-render tokens), and grows with `--drift`. Run the
drift sweep (`--ablations`) to see it.
