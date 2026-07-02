---
name: Bug report
about: Something doesn't work as documented
title: "[bug] "
labels: bug
---

**What happened**
A clear description of the bug.

**Reproduce**
Exact command(s) and the smallest input that triggers it:
```bash
python bench/run_bench.py --backend mock ...
```

**Expected**
What you expected instead.

**Environment**
- OS / Python version:
- Foveance version (`foveance version`):
- Extras installed (`[dev]`/`[bench]`/`[proxy]`/`[ml]`):
- If a benchmark issue: attach `bench/results/headline.json` and model/hardware details.

**Logs**
Paste relevant output (and `bench/results/drift_twin_audit.json` if arms behave unexpectedly).
