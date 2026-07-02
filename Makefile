.PHONY: install test cov lint bench demo repro clean

install:
	pip install -e ".[dev,bench]"

# Full pytest suite (falls back to the seed script runner if pytest is absent).
test:
	@python -m pytest -q 2>/dev/null || python tests/test_allocator.py

# Coverage gate on the core modules.
cov:
	python -m pytest -q --cov=foveance.store --cov=foveance.predictor \
	  --cov=foveance.allocator --cov=foveance.controller --cov-report=term-missing

lint:
	@ruff check src tests bench 2>/dev/null || echo "ruff not installed (pip install -e .[dev])"
	@mypy src 2>/dev/null || echo "mypy not installed (pip install -e .[dev])"

# Offline benchmark (mock model) — CI-safe, no GPU/network.
demo:
	foveance demo

bench:
	python bench/run_bench.py --backend mock --models mock --suite synthetic \
	  --budgets 600,1200,1600,2500,4000 --tasks 6 --turns 40 --drift 0.7 \
	  --name-target false --fidelity-cost true --greedy-gap --ablations
	python bench/analyze.py
	python bench/plots.py

# Real benchmark example (needs Ollama running locally).
bench-real:
	python bench/run_bench.py --backend ollama \
	  --models gemma2:9b,gemma2:2b,qwen2.5:7b,llama3.1:8b \
	  --budget 2400 --tasks 8 --turns 60 --drift 0.7

repro: test bench
	@echo "Offline repro complete: CSVs in bench/results/, plots in bench/plots/, report in bench/report.md."

clean:
	rm -rf **/__pycache__ .pytest_cache .mypy_cache .ruff_cache
