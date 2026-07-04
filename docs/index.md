# Foveance

**Cut your LLM token bill by 60%+ — without changing your code or your answers.**

When you chat with an AI agent for a while, the conversation history keeps piling up. You pay for
every old message on every new turn, and past a point the model actually gets *worse* because the
important facts are buried under clutter. Foveance keeps the parts of the history that still matter,
trims the parts that don't, and hands the model a shorter context — same answers, a fraction of the
tokens. Nothing is deleted forever, and you don't change a line of your app.

In real tests it kept full accuracy while using **60–64% fewer tokens**, and correctly recalled a
buried fact that the full, uncompressed history got *wrong*.

## Get started in 30 seconds

### You use a coding agent (Claude Code, Codex, aider, …)

```bash
pip install foveance
foveance wrap claude          # or:  foveance wrap -- codex "fix the tests"
```

It runs your tool exactly as before, just cheaper, and prints how much you saved. Your API key is
untouched, nothing is stored.

### You write Python

```bash
pip install foveance
```
```python
from foveance import shrink

smaller = shrink(messages, budget=2000)   # your OpenAI-style messages list
# ...send `smaller` to your model instead of `messages`. Same answers, fewer tokens.
```

### Just try it (no API key, no GPU)

```bash
pip install foveance
foveance demo
```

## Documentation

- [Usage guide](usage.md) — the proxy, `foveance wrap`, per-tool recipes, and configuration.
- [Architecture](architecture.md) — how the store, predictor, allocator, and controller fit together.
- [Theory](theory.md) — the trajectory rate-distortion framework and the five theorems.
- [Baselines](baselines.md) — the policy arms and how they compare.
- [Limitations](limitations.md) — the honest failure modes and when the cheap heuristic suffices.
- [Novelty & positioning](NOVELTY.md) — what is and isn't claimed as new (prior-art table).

## Links

- **Source:** [github.com/aimaghsoodi/foveance](https://github.com/aimaghsoodi/foveance)
- **PyPI:** [pypi.org/project/foveance](https://pypi.org/project/foveance/) (`pip install foveance`)
- **npm:** [foveance-proxy](https://www.npmjs.com/package/foveance-proxy) (`npx foveance-proxy`)
- **Benchmark data:** [Hugging Face dataset](https://huggingface.co/datasets/AbteeXAILabs/foveance-benchmark)

Apache-2.0 licensed.
