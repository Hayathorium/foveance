# foveance-proxy (npm launcher)

**Cut your LLM token bill by 60%+ without changing your code or your answers.**

Your AI agent's chat history piles up — you pay for every old message on every new turn, and the
model gets worse as important facts get buried. [Foveance](https://github.com/aimaghsoodi/foveance)
automatically keeps what matters and trims the rest, so you get the same answers for a fraction of
the tokens.

This is a tiny Node launcher for the Foveance proxy, so users of Node-based AI tools (Claude Code,
OpenAI Codex, opencode, Crush, Continue, ...) can start it with one command, without a manual
Python step. The proxy is **OpenAI- and Anthropic-compatible**, streams, and forwards your API key
untouched — you only point a client's base URL at it.

## Prerequisites

- Node.js >= 16
- Python >= 3.10 with Foveance installed once:

```bash
pip install foveance
```

## Use

```bash
# OpenAI upstream
npx foveance-proxy --upstream https://api.openai.com/v1

# Anthropic upstream (Claude Code, Anthropic SDK)
npx foveance-proxy --upstream https://api.anthropic.com/v1

# Local Ollama / vLLM / TGI / LM Studio
npx foveance-proxy --upstream http://localhost:11434/v1
```

All flags are forwarded to `foveance proxy` (`--host`, `--port`, `--budget`, `--drift`, `--policy`,
`--upstream`, `--cache-aware`, `--price-per-mtok`), and the `FOVEANCE_UPSTREAM` / `FOVEANCE_BUDGET`
/ `FOVEANCE_DRIFT` / `FOVEANCE_POLICY` environment variables are honoured too. A live tokens-saved
dashboard serves at `http://localhost:8799/` while the proxy runs.

> Even simpler: the Python CLI has `foveance wrap claude` (or `foveance wrap -- <any command>`),
> which starts the proxy, routes the tool through it, and prints a tokens-saved summary on exit.

Then point your tool at it (one variable):

```bash
# OpenAI-style tools, Codex, Ollama-backed clients
export OPENAI_BASE_URL=http://localhost:8799/v1

# Anthropic-style tools, Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8799
```

See the [main README](https://github.com/aimaghsoodi/foveance#cut-your-token-usage-everywhere-drop-in-proxy-zero-client-changes)
and [`docs/usage.md`](https://github.com/aimaghsoodi/foveance/blob/main/docs/usage.md) for per-tool
recipes. This launcher is a convenience wrapper; the proxy itself lives in the Python package.

Apache-2.0.
