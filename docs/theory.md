# Theory (informal)

Full statements and proofs are in the accompanying research manuscript (in preparation)
(Appendices A–G). This page is the plain-language version.

## The object: a trajectory distortion–rate function
At each turn the compressor assigns every context item a fidelity with a token cost; the sum is
the **rate** `R_t`. A black-box model reads the rendered context and answers; the **distortion**
`D_t` is the expected task loss (e.g. `1 − success`), *not* next-token KL. `D*(R)` is the best
average distortion achievable at average rate `R` over the whole trajectory. This is the
sequential, task-success generalization of Nagle et al. (2024); their static token-deletion LP
is the `T=1, L=1`, query-agnostic special case (**Thm. 1**).

## Anticipation is optimal (Thm. 2)
The future need `Q_{>t}` is revealed *after* the compression choice — a *predictive Wyner–Ziv*
structure. A reactive compressor may only use the current query `q_t`; an anticipatory one uses
`p(Q_{>t} | history)`. Since reactive policies are a subset of anticipatory ones,
`D*_antic ≤ D*_react`. The gap is governed by a **conditional mutual information**
`I(item relevance ; Q_{>t} | q_t)` — how predictable the future is *beyond* the present. No
cross-turn dependence ⇒ gap is zero. This is what the drift sweep tests.

## The deployable policy is near-optimal (Thm. 3)
Per-turn fidelity allocation is a multiple-choice knapsack over the value curves. Taking each
curve's concave envelope and buying upgrades by descending value-per-token (the Lagrangian/Whittle
index) gives a value within **one item's value** of the optimum, equal to the LP relaxation up to
one fractional item, in `O(n log n)`. Empirically: `index ≤ OPT(DP) ≤ LP`, with the relative gap
typically ~1–2%.

## When is anticipation necessary? (Thm. 4 — the useful one)
With query mixing `φ` (0 = i.i.d., →1 = highly dependent) and per-item re-inflation cost `η`:
`Regret(myopic) ≤ c₁(1−φ)` and `Regret(myopic) ≥ c₂·φ·η`. So if queries are near-independent
**and** re-inflation is cheap, the reactive heuristic (AFM/Headroom) is near-optimal; as
dependence or re-fetch cost grows, anticipation has a guaranteed, quantified advantage. This is
the engineering decision rule.

## Reversible re-inflation is free (Thm. 5)
Cast the fidelity ladder as Equitz–Cover successive refinement with side information. If the
renders are nested (a Markov chain), holding an item low now and upgrading later costs no more
total rate than committing to the high fidelity up front — so two-sided refinement is
rate-optimal. Non-nested digests incur an explicit penalty the index policy prices.

## Robustness (Lemma)
If the relevance estimator has calibration error `ε`, realized value is within `O(ε·n)` of the
oracle — a learned predictor with bounded error preserves the guarantees, and degrades gracefully
toward the reactive baseline rather than catastrophically.
