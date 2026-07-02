# Contributing to Foveance

Thanks for your interest! Foveance aims to be a small, rigorous, reviewer-proof codebase.

## Ground rules
- **Never fabricate benchmark numbers.** Every figure in the report traces to a CSV in
  `bench/results/`. Unmeasured = `TODO`, not invented.
- **Keep the novelty boundary honest.** The multi-fidelity store under a budget is prior art
  (AFM, ContextBudget, ACON, MemAct). Don't claim it. See [`docs/NOVELTY.md`](docs/NOVELTY.md).
- **Core stays dependency-free.** `store/predictor/allocator/controller` must run offline with no
  third-party imports. ML/proxy/bench deps live behind extras.

## Dev setup
```bash
pip install -e ".[dev,bench]"
make test     # pytest
make cov      # coverage gate on core modules (target ≥90%)
make lint     # ruff + mypy
```

## Before opening a PR
1. `make lint` is clean (ruff + mypy).
2. `make cov` keeps core coverage ≥90%.
3. New behavior has tests; new policies are added to `foveance.baselines.POLICIES`.
4. If you touch the `reactive_afm`/`foveance` arms, the drift-twin audit
   (`bench/results/drift_twin_audit.json`) must still report `only_difference_is_drift: true`.
5. Docstrings on public API; type hints throughout.

## Commit style
Small, reviewable commits. If you work within the phased build, tag commits `[phase N]`.

## Reporting issues
Use the templates in `.github/ISSUE_TEMPLATE/`. For benchmark discrepancies, include the exact
command, the `headline.json`, and your hardware/model versions.
