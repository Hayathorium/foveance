# Foveance — Novelty & Positioning

This page keeps the project's claims honest. As of **mid-2026** this space is crowded,
and the broad idea is not new; the narrow, defensible core below is what Foveance
actually claims.

## What ALREADY EXISTS (cite, do not claim)

| Prior work | What it already does | Why our headline can't be "this" |
|---|---|---|
| **AFM — Adaptive Focus Memory** (Cruz 2025, arXiv 2511.12712, code on GitHub) | Per-message **fidelity tiers** FULL/COMPRESSED/PLACEHOLDER, scored by semantic-similarity-to-current-query + half-life recency + importance, packed under a **token budget**. | This is the multi-fidelity-store-under-budget mechanism. **Already done.** We treat AFM as the primary baseline and as the "reactive" special case of our policy. |
| **ContextBudget** (2026) | Context compression as a **budget-constrained sequential decision problem**, adapting to remaining window capacity. | "Sequential + budget" is taken. |
| **ACON** (Kang et al. 2025, arXiv 2510.00615) | Black-box, gradient-free compression of agent **observations + history** via failure-driven guideline optimization; API-applicable; distillable. | "Black-box unified agent compression" is taken. |
| **MemAct — Memory-as-Action** (Zhang et al. 2025, arXiv 2510.12635) | Context curation as **RL policy actions** (prune/write), MDP formulation, DCPO training. | "Context management as MDP/RL" is taken (but requires training the agent; we don't). |
| **Sequential Wyner–Ziv for KV cache** (2026, arXiv 2605.25085) | **Rate–distortion** limits of online cache compression as sequential Wyner–Ziv with next-step query as side info. | RD/Wyner–Ziv framing exists — but **white-box, next-step, KL distortion, single generation.** |
| **Nagle et al.** (NeurIPS 2024, arXiv 2407.15504) | Distortion–rate function for **static** black-box token-deletion prompt compression; shows query-awareness matters. | RD for prompts exists — but **one-shot**, hard-deletion only. |
| RCR-Router, DAST, wireless-CE, Quest/TOVA | importance-/relevance-aware budget or fidelity allocation in various single-shot or KV settings. | "importance-aware allocation" is a known pattern. |

## What does NOT appear to exist anywhere (our defensible contribution)

The **intersection** is empty. Our four claims, each of which is individually checkable:

1. **Anticipatory (future-relevance) allocation criterion.** Every prior system scores
   items by relevance to the *current* query (AFM), failure traces (ACON), or remaining
   budget (ContextBudget). We allocate fidelity by **expected relevance to the *future* of
   the trajectory** — an explicit posterior `p(future needs | history)` — and show it
   dominates the reactive criterion exactly when the task has cross-turn dependency
   structure (drift). The reactive policy is the `drift = 0` special case.

2. **A fundamental-limits theory for the black-box, multi-turn, task-success setting.**
   We define the **trajectory distortion–rate function** with *task-success* distortion and
   a **future** side-information variable (predictive Wyner–Ziv), generalizing Nagle
   (static → sequential) and distinct from the KV paper (white-box/next-step/KL →
   black-box/future/task). No existing work couples a fundamental-limits result to a
   deployable black-box agentic allocator.

3. **Two-sided successive refinement (re-inflation) with refinability conditions.** We give
   conditions under which holding items at multiple fidelities is *free* (no rate penalty),
   licensing principled re-inflation. Prior systems compress one-way (AFM can re-upgrade
   reactively but offers no theory; ACON/summary methods are lossy/destructive).

4. **A near-optimal index policy with a measured greedy gap.** We cast per-item fidelity as
   a multiple-choice knapsack / restless-bandit and use a Whittle/Lagrangian index that is
   provably within one item's value of the optimum, and we prove **when local-greedy
   heuristics (AFM/Headroom/ContextBudget) are already near-optimal vs when anticipation is
   necessary**. That "when is anticipation worth it" theorem is the practically useful,
   genuinely absent result.

## One-sentence positioning

> Prior work shows *how* to compress agent context under a budget; Foveance shows the
> *fundamental limit* of doing so over a trajectory, gives the *anticipatory* policy that
> approaches it, and proves *when* anticipation beats the reactive heuristics everyone
> currently ships.

## Honesty guards for the paper

- Lead the Related Work with AFM, ACON, ContextBudget, MemAct, the two RD papers. Position
  against them explicitly; reviewers in this area will know all of them.
- The multi-fidelity store is **substrate, not contribution.** Say so.
- Report the **greedy gap honestly** — if it's small on real tasks, that's a finding
  (it tells practitioners when cheap heuristics suffice), not a weakness.
- Do **not** fabricate numbers. All result tables are filled from real runs (PROMPT_2).
- If on real models `reactive ≈ foveance`, report it. The theory (when anticipation helps)
  must then explain why — that is still a publishable, honest contribution.
