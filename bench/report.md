# Foveance benchmark report

> **Source: REAL model runs.**

- models: gemma2:2b, llama3.2:1b, qwen2.5:1.5b
- budgets: [400, 700, 1200]
- seeds per cell: 5
- drift-twin audit: reactive_afm vs foveance differ only in drift = True

## gemma2:2b
- full replay: acc=1.0, total tokens=10228
- **foveance reaches full accuracy at budget 400 using 3663 tokens => 64.2% fewer than full**
- foveance vs reactive (AFM) max Δacc across budgets: 0.0
- foveance vs reactive paired Δacc CI (mid budget): [-0.0667, -0.2, 0.0], Wilcoxon p=1.0
- Pareto AUC: full=1.0, recency=0.6667, truncate=0.0, uniform=0.0, reactive_afm=1.0, foveance=0.9667, oracle=nan

## llama3.2:1b
- full replay: acc=1.0, total tokens=8290
- **foveance reaches full accuracy at budget 400 using 3108 tokens => 62.5% fewer than full**
- foveance vs reactive (AFM) max Δacc across budgets: 0.0
- foveance vs reactive paired Δacc CI (mid budget): [0.0, 0.0, 0.0], Wilcoxon p=nan
- Pareto AUC: full=1.0, recency=0.6667, truncate=0.142, uniform=1.0, reactive_afm=0.9833, foveance=0.9833, oracle=nan

## qwen2.5:1.5b
- full replay: acc=1.0, total tokens=9931
- **foveance reaches full accuracy at budget 400 using 3622 tokens => 63.5% fewer than full**
- foveance vs reactive (AFM) max Δacc across budgets: 0.0
- foveance vs reactive paired Δacc CI (mid budget): [0.0, 0.0, 0.0], Wilcoxon p=nan
- Pareto AUC: full=1.0, recency=0.6667, truncate=0.0, uniform=0.0, reactive_afm=1.0, foveance=1.0, oracle=nan

## Greedy gap (Thm 3, index vs exact DP vs LP bound)
- mean relative gap = 0.01798, p95 = 0.1272, max = 0.21923 over 800 measurements; mean index/LP = 0.98991 (min 0.92088)

## Ablations

| ablation | setting | arm | accuracy | in_tok | tok/correct |
|---|---|---|---|---|---|
| drift | drift=0.0,name_target=True | reactive_afm | 0.9815 | 63313.0 | 2389.2 |
| drift | drift=0.0,name_target=True | foveance | 0.9815 | 63313.0 | 2389.2 |
| drift | drift=0.3,name_target=True | reactive_afm | 0.9815 | 63260.2 | 2387.2 |
| drift | drift=0.3,name_target=True | foveance | 0.9815 | 63441.3 | 2394.0 |
| drift | drift=0.6,name_target=True | reactive_afm | 0.9815 | 63263.2 | 2387.3 |
| drift | drift=0.6,name_target=True | foveance | 0.9815 | 63484.8 | 2395.7 |
| drift | drift=0.9,name_target=True | reactive_afm | 0.963 | 63266.7 | 2433.3 |
| drift | drift=0.9,name_target=True | foveance | 0.963 | 63742.0 | 2451.6 |
| drift | drift=0.0,name_target=False | reactive_afm | 0.9938 | 63483.7 | 2365.9 |
| drift | drift=0.0,name_target=False | foveance | 0.9938 | 63483.7 | 2365.9 |
| drift | drift=0.3,name_target=False | reactive_afm | 0.9938 | 63526.2 | 2367.4 |
| drift | drift=0.3,name_target=False | foveance | 1.0 | 63647.5 | 2357.3 |
| drift | drift=0.6,name_target=False | reactive_afm | 0.9877 | 63437.2 | 2378.9 |
| drift | drift=0.6,name_target=False | foveance | 0.9938 | 63689.8 | 2373.5 |
| drift | drift=0.9,name_target=False | reactive_afm | 0.9877 | 63322.7 | 2374.6 |
| drift | drift=0.9,name_target=False | foveance | 0.9938 | 63616.3 | 2370.8 |
| predictor | heuristic | foveance | 1.0 | 63730.3 | 2360.4 |
| predictor | learned | foveance | 0.9815 | 63835.2 | 2408.9 |
| retrieve | retrieve_on | foveance | 1.0 | 56919.0 | 2108.1 |
| retrieve | retrieve_off | foveance | 1.0 | 56919.0 | 2108.1 |
| fidelity_cost | fidelity_cost_on | foveance | 1.0 | 63730.3 | 2360.4 |
| fidelity_cost | fidelity_cost_off | foveance | 1.0 | 56919.0 | 2108.1 |

Drift sweep tests Thm 2/4 (anticipation gain vs cross-turn dependency); retrieve/fidelity-cost rows test two-sided refinement and the re-fetch penalty.

## Reproduce

Real-model run (Ollama). Token savings come from the budget binding on growing context; the greedy-gap and drift-sweep ablations are model-independent allocator and predictor measurements.

```bash
python bench/run_bench.py --backend ollama \
    --models gemma2:2b,llama3.2:1b,qwen2.5:1.5b --suite synthetic \
    --budgets 400,700,1200 --tasks 5 --turns 8 --n-facts 3 --block-lines 18 --drift 0.7 --greedy-gap --ablations
python bench/analyze.py && python bench/plots.py
```
