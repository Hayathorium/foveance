#!/usr/bin/env bash
# Offline end-to-end demo (no GPU, no network): mock model -> benchmark -> analysis -> plots.
# Proves the whole pipeline; numbers are illustrative.
# This is exactly run_everything.sh minus Ollama; swap in real models with run_everything.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install --break-system-packages -q numpy matplotlib 2>/dev/null || true

echo "==> benchmark (mock, budget sweep + greedy gap + ablations)"
python bench/run_bench.py --backend mock --models mock --suite synthetic \
  --budgets 600,1200,1600,2500,4000 --tasks 6 --turns 40 --drift 0.7 \
  --name-target false --fidelity-cost true --greedy-gap --ablations
echo "==> analyze"; python bench/analyze.py
echo "==> plots"; python bench/plots.py
echo "DONE -> bench/report.md, bench/results/, bench/plots/"
