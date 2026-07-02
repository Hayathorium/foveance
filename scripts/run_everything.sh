#!/usr/bin/env bash
# Foveance — one command: install Ollama, pull models, run the real pre/post comparison,
# and do all the analysis + plots.
#
# Usage:
#   bash scripts/run_everything.sh
#   MODELS="gemma2:9b,qwen2.5:7b" BUDGETS="600,1200,2400,4800" TASKS=8 bash scripts/run_everything.sh
#
# Requires: a machine where Ollama can run (Linux/macOS, ideally a GPU). This is the step that
# cannot run in a restricted sandbox — run it on your own box.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-gemma2:9b,gemma2:2b,qwen2.5:7b,llama3.1:8b}"
BUDGETS="${BUDGETS:-600,1200,2400,4800}"
TASKS="${TASKS:-8}"
TURNS="${TURNS:-60}"
DRIFT="${DRIFT:-0.7}"

echo "==> [1/6] Python deps"
pip install -e ".[bench,ml]" >/dev/null 2>&1 || pip install --break-system-packages -e ".[bench,ml]"

echo "==> [2/6] Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  echo "    installing Ollama..."
  if [[ "$(uname)" == "Linux" ]]; then curl -fsSL https://ollama.com/install.sh | sh
  elif [[ "$(uname)" == "Darwin" ]]; then brew install ollama || { echo "install Ollama from https://ollama.com/download"; exit 1; }
  else echo "Install Ollama manually: https://ollama.com/download"; exit 1; fi
fi
# start server if not already up
if ! curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then
  echo "    starting 'ollama serve'..."; (ollama serve >/tmp/ollama.log 2>&1 &) ; sleep 4
fi

echo "==> [3/6] pull models: $MODELS"
IFS=',' read -ra MS <<< "$MODELS"
for m in "${MS[@]}"; do echo "    pull $m"; ollama pull "$m"; done

echo "==> [4/6] benchmark (real models, budget sweep, greedy gap)"
python bench/run_bench.py --backend ollama --models "$MODELS" \
  --budgets "$BUDGETS" --tasks "$TASKS" --turns "$TURNS" --drift "$DRIFT" \
  --name-target false --fidelity-cost true --greedy-gap --ablations

echo "==> [5/6] analysis (CIs, paired Wilcoxon, iso-accuracy savings, Pareto AUC, greedy gap)"
python bench/analyze.py

echo "==> [6/6] plots"
python bench/plots.py

echo
echo "DONE."
echo "  report:  bench/report.md"
echo "  numbers: bench/results/headline.json, summary.csv"
echo "  figures: bench/plots/*.png"
