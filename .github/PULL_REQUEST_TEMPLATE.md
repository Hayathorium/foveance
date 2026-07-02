# Pull request

## What and why
<!-- One or two sentences on the change and the motivation. -->

## Checklist
- [ ] `make lint` is clean (ruff + mypy)
- [ ] `make cov` keeps core coverage at or above 90%
- [ ] New behavior has tests; new policies are registered in `foveance.baselines.POLICIES`
- [ ] If the `reactive_afm` / `foveance` arms were touched, the drift-twin audit
      (`bench/results/drift_twin_audit.json`) still reports `only_difference_is_drift: true`
- [ ] No fabricated benchmark numbers; any results come from a real run
- [ ] Docstrings on public API; type hints throughout
- [ ] Claims stay within the boundaries in `docs/NOVELTY.md`

## Results / evidence
<!-- For perf or benchmark changes, paste the relevant CSV rows or the headline.json diff. -->
