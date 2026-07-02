"""Tests for `foveance wrap`, the tokens-saved accounting, cache-aware compression, and the
live dashboard. `wrap` is exercised end-to-end: a real uvicorn proxy is started on a free port,
a real child process (python) sends a chat request through it to a local echo upstream, and the
exit summary is asserted on. Skipped automatically when the [proxy] extra is absent."""
import json
import socket
import sys
import threading
from http.server import HTTPServer

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("uvicorn")
from fastapi.testclient import TestClient  # noqa: E402

from foveance import cli  # noqa: E402
from foveance.proxy import FoveanceProxy, build_app, _digest_block  # noqa: E402
from test_proxy_integration import _EchoUpstream  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_wrap_end_to_end(capsys):
    """`foveance wrap -- <cmd>` starts the proxy, the child reaches it via OPENAI_BASE_URL, the
    request is compressed and forwarded to the upstream, and the exit summary reports savings."""
    server = HTTPServer(("127.0.0.1", 0), _EchoUpstream)
    up_port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    child = (
        "import json,os,urllib.request\n"
        "base = os.environ['OPENAI_BASE_URL']\n"
        "assert os.environ['ANTHROPIC_BASE_URL'].rstrip('/') + '/v1' == base\n"
        "body = {'model': 'x', 'user': 'w1', 'messages': ["
        " {'role': 'user', 'content': 'FACT z=9 ' + 'log '*300},"
        " {'role': 'assistant', 'content': 'ok'},"
        " {'role': 'user', 'content': 'recall z'}]}\n"
        "req = urllib.request.Request(base + '/chat/completions',"
        " data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})\n"
        "resp = urllib.request.urlopen(req, timeout=30)\n"
        "assert resp.status == 200, resp.status\n"
    )
    try:
        rc = cli.main(["wrap", "--port", str(_free_port()), "--budget", "120",
                       "--upstream", f"http://127.0.0.1:{up_port}/v1",
                       "--", sys.executable, "-c", child])
    finally:
        server.shutdown()
    out = capsys.readouterr().out
    assert rc == 0
    assert "Foveance session summary" in out
    assert "requests proxied : 1" in out
    # the compressed request really went upstream (last turn verbatim)
    assert _EchoUpstream.received["messages"][-1]["content"] == "recall z"


def test_wrap_requires_a_command(capsys):
    assert cli.main(["wrap"]) == 2
    assert "usage: foveance wrap" in capsys.readouterr().err


def test_accounting_totals_and_usd_estimate():
    """prepare() feeds the running before/after totals; stats() exposes saved tokens, %, and the
    $-equivalent at the configured price."""
    px = FoveanceProxy(budget=120, price_per_mtok=10.0)
    req = {"model": "x", "user": "acct",
           "messages": [{"role": "user", "content": "FACT k=V\n" + "log line\n" * 300},
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": "recall k"}]}
    fwd, stats = px.prepare(req)
    assert stats["est_tokens_before"] > stats["est_tokens_after"] > 0
    s = px.stats()
    assert s["requests"] == 1 and s["compressed_requests"] == 1
    assert s["est_tokens_saved"] == s["est_tokens_before"] - s["est_tokens_after"]
    assert 0 < s["est_saved_pct"] <= 100
    assert s["est_usd_saved"] == round(s["est_tokens_saved"] * 10.0 / 1e6, 4)


def test_cache_aware_never_touches_cached_prefix():
    """With cache_aware=True, everything at or before the last cache_control breakpoint is
    byte-identical (the provider's prompt cache survives); content after it is still digested.
    With cache_aware=False the same old content IS digested."""
    big = "\n".join(f"line {i}: status=ok value={i}" for i in range(400))
    tools = [{"name": "bash", "description": "x",
              "input_schema": {"type": "object", "properties": {}}}]
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": big}]},
        {"role": "assistant", "content": [{"type": "text", "text": "noted",
                                           "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                      "content": big}]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
    ]
    req = {"system": "s", "messages": msgs, "tools": tools}

    aware = FoveanceProxy(agentic_protect_last=1, cache_aware=True)
    fwd, stats = aware.prepare_anthropic(json.loads(json.dumps(req)))
    assert fwd["messages"][0] == msgs[0]            # before breakpoint: untouched
    assert fwd["messages"][1] == msgs[1]            # the breakpoint itself: untouched
    after = fwd["messages"][2]["content"][0]["content"]
    assert "elided by Foveance" in after            # after breakpoint: still digested
    assert fwd["messages"][-1] == msgs[-1]          # recent turn protected as always

    naive = FoveanceProxy(agentic_protect_last=1, cache_aware=False)
    fwd2, _ = naive.prepare_anthropic(json.loads(json.dumps(req)))
    assert "elided by Foveance" in fwd2["messages"][0]["content"][0]["text"]


def test_digest_block_never_modifies_cache_control_blocks():
    big = "x" * 5000
    blk = {"type": "text", "text": big, "cache_control": {"type": "ephemeral"}}
    assert _digest_block(blk) is blk
    assert "elided" in _digest_block({"type": "text", "text": big})["text"]


def test_dashboard_served_at_root_and_admin():
    app = build_app(FoveanceProxy(), upstream_url="http://127.0.0.1:9/v1")
    client = TestClient(app)
    for path in ("/", "/admin"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert "tokens saved" in r.text and "/admin/stats" in r.text
    # /health stays JSON for orchestrators
    assert client.get("/health").json()["status"] == "ok"
    # the JSON the dashboard polls includes the new totals
    s = client.get("/admin/stats").json()
    for key in ("est_tokens_before", "est_tokens_after", "est_tokens_saved",
                "est_saved_pct", "est_usd_saved", "price_per_mtok"):
        assert key in s
