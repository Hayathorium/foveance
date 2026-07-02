"""End-to-end proxy test over real HTTP.

Spins up a local threaded HTTP server as the upstream, builds the FastAPI proxy app pointed at
it, and drives a chat-completions request through FastAPI's TestClient. This exercises the full
path (route -> FoveanceProxy.transform -> real HTTP forward -> response) rather than just the core.
Skipped automatically when the [proxy] extra is absent.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from foveance.proxy import FoveanceProxy, build_app  # noqa: E402


class _EchoUpstream(BaseHTTPRequestHandler):
    received = {}

    def do_GET(self):  # /models probe
        body = json.dumps({"object": "list", "data": [{"id": "echo-model", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        _EchoUpstream.received = payload
        if payload.get("stream"):  # emit a tiny SSE stream the proxy must pass through
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"data: {{\"chunk\": {i}}}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        chars = sum(len(str(m.get("content", ""))) for m in payload.get("messages", []))
        body = json.dumps({"choices": [{"message": {"role": "assistant",
                                                    "content": f"saw {chars} chars"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


def test_conv_id_fallback_and_prepare():
    """No explicit id -> a stable auto id derived from the first message; distinct first messages
    get distinct stores; prepare() returns a forward-ready, compressed body."""
    px = FoveanceProxy(budget=120)
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "alpha needle=A " + "x " * 200},
            {"role": "user", "content": "recall"}]
    assert px._conv_id({}, msgs).startswith("auto-")
    assert px._conv_id({"user": "u9"}, msgs) == "u9"
    assert px._conv_id({}, msgs) != px._conv_id(
        {}, [{"role": "user", "content": "beta different"}])
    fwd, stats = px.prepare({"model": "x", "messages": msgs})
    assert stats["compressed"] is True and fwd["messages"][-1]["content"] == "recall"


def test_structured_tool_use_passes_through_unchanged():
    """Claude-Code-style structured conversations (tool_use/tool_result blocks, non-string content)
    must NOT be collapsed -- that would break tool-call pairing and the upstream 400s. The proxy
    passes them through verbatim instead (correctness over savings)."""
    px = FoveanceProxy(budget=120)
    # Anthropic: content as block lists, with a tool_use/tool_result pair
    a_msgs = [{"role": "user", "content": [{"type": "text", "text": "run ls " + "x " * 200}]},
              {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash",
                                                 "input": {"cmd": "ls"}}]},
              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                            "content": "file1\nfile2"}]}]
    sys_blocks = [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}]
    fwd, stats = px.prepare_anthropic({"system": sys_blocks, "messages": a_msgs})
    assert stats["reason"] == "agentic-inplace"
    assert fwd["messages"] == a_msgs and fwd["system"] == sys_blocks  # small -> intact, pairing kept

    # OpenAI: assistant tool_calls + tool role must also pass through
    o_msgs = [{"role": "user", "content": "do it " + "y " * 200},
              {"role": "assistant", "content": None,
               "tool_calls": [{"id": "c1", "type": "function",
                               "function": {"name": "f", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "result"},
              {"role": "user", "content": "thanks"}]
    fwd2, stats2 = px.prepare({"messages": o_msgs})
    assert stats2["compressed"] is False and fwd2["messages"] == o_msgs


def test_agentic_request_with_tools_passes_through_unchanged():
    """Agents (Claude Code, Codex) declare a `tools` array on every call -- even before any tool is
    used and with plain-text messages. Such requests are forwarded verbatim so the proxy never
    breaks them; only plain chat (no tools) is compressed."""
    px = FoveanceProxy(budget=120)
    long_hist = [{"role": "user", "content": "investigate " + "z " * 300},
                 {"role": "assistant", "content": "looking"},
                 {"role": "user", "content": "and now?"}]
    tools = [{"name": "bash", "description": "run a shell command",
              "input_schema": {"type": "object", "properties": {}}}]
    # Anthropic with tools -> passthrough
    fwd_a, st_a = px.prepare_anthropic({"system": "s", "messages": long_hist, "tools": tools})
    assert st_a["compressed"] is False and fwd_a["messages"] == long_hist
    # OpenAI with tools -> passthrough
    fwd_o, st_o = px.prepare({"messages": long_hist, "tools": tools})
    assert st_o["compressed"] is False and fwd_o["messages"] == long_hist
    # identical conversation WITHOUT tools -> compressed (the value path still works)
    _, st_plain = px.prepare_anthropic({"system": "s", "messages": long_hist})
    assert st_plain["compressed"] is True


def test_agentic_inplace_compression_shrinks_old_tool_output():
    """The real value path for agents: a large OLD tool_result is digested in place while the
    message list, roles, tool_use<->tool_result pairing, and the recent turns stay byte-intact."""
    import json
    px = FoveanceProxy(agentic_protect_last=1)
    big = "\n".join(f"line {i}: status=ok value={i * 7} path=/srv/{i % 97}" for i in range(400))
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "investigate the logs"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash",
                                           "input": {"cmd": "cat log"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big}]},
        {"role": "assistant", "content": [{"type": "text", "text": "logs look fine"}]},
        {"role": "user", "content": [{"type": "text", "text": "what was on line 5?"}]},
    ]
    tools = [{"name": "bash", "description": "x", "input_schema": {"type": "object", "properties": {}}}]
    fwd, stats = px.prepare_anthropic({"system": "s", "messages": msgs, "tools": tools})
    assert stats["reason"] == "agentic-inplace" and stats["compressed"] is True
    assert [m["role"] for m in fwd["messages"]] == [m["role"] for m in msgs]  # structure intact
    assert fwd["messages"][1]["content"][0]["id"] == "t1"                     # tool_use id intact
    assert fwd["messages"][2]["content"][0]["tool_use_id"] == "t1"            # pairing intact
    shrunk = fwd["messages"][2]["content"][0]["content"]
    assert len(shrunk) < len(big) and "elided by Foveance" in shrunk          # big output digested
    assert fwd["messages"][-1] == msgs[-1]                                    # last turn protected
    assert len(json.dumps(fwd["messages"])) < len(json.dumps(msgs))           # net smaller


def test_responses_inplace_compression_preserves_pairing():
    """OpenAI Responses API (used by Codex / Agents SDK): a large old function_call_output is
    digested in place while item order, types, and function_call<->output call_id pairing stay
    intact and the most recent item is protected."""
    import json
    px = FoveanceProxy(agentic_protect_last=1)
    big = "\n".join(f"row {i}: ok value={i * 3} path=/srv/{i % 64}" for i in range(400))
    inp = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "check logs"}]},
        {"type": "function_call", "call_id": "fc1", "name": "bash", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "fc1", "output": big},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what was row 5?"}]},
    ]
    fwd, stats = px.prepare_responses({"model": "gpt-5", "input": inp,
                                       "tools": [{"type": "function", "name": "bash"}]})
    assert stats["reason"] == "responses-inplace" and stats["compressed"] is True
    assert [it.get("type") for it in fwd["input"]] == [it.get("type") for it in inp]
    assert fwd["input"][1]["call_id"] == "fc1" and fwd["input"][2]["call_id"] == "fc1"  # pairing
    out = fwd["input"][2]["output"]
    assert len(out) < len(big) and "elided by Foveance" in out
    assert fwd["input"][-1] == inp[-1]                                       # last item protected
    assert len(json.dumps(fwd["input"])) < len(json.dumps(inp))             # net smaller
    # string input passes through unchanged
    fwd2, st2 = px.prepare_responses({"model": "gpt-5", "input": "just a string"})
    assert st2["compressed"] is False and fwd2["input"] == "just a string"


def test_anthropic_prepare_folds_context_into_system():
    px = FoveanceProxy(budget=120)
    req = {"model": "claude", "system": "be terse",
           "messages": [{"role": "user", "content": "FACT k=V\n" + "log\n" * 200},
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": "recall k"}]}
    fwd, stats = px.prepare_anthropic(req)
    assert stats["compressed"] is True
    assert "be terse" in fwd["system"] and len(fwd["messages"]) == 1
    assert fwd["messages"][0]["content"] == "recall k"


def test_proxy_compresses_over_real_http():
    server = HTTPServer(("127.0.0.1", 0), _EchoUpstream)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        app = build_app(FoveanceProxy(budget=120), upstream_url=f"http://127.0.0.1:{port}/v1")
        client = TestClient(app)
        big = "FACT token=SECRET42\n" + "\n".join(f"log {i} status=ok" for i in range(150))
        req = {"model": "x", "user": "c1",
               "messages": [{"role": "system", "content": "be brief"},
                            {"role": "user", "content": big},
                            {"role": "assistant", "content": "ok"},
                            {"role": "user", "content": "recall token"}]}
        r = client.post("/v1/chat/completions", json=req)
        assert r.status_code == 200
        body = r.json()
        # upstream received a compressed message list, last user query preserved verbatim
        fwd = _EchoUpstream.received["messages"]
        assert fwd[-1]["content"] == "recall token"
        assert len(json.dumps(fwd)) < len(big)
        assert body["foveance"]["compressed"] is True
        stats = client.get("/admin/stats").json()
        assert stats["requests"] >= 1
        # health + model-list probe both succeed (CLI agents call these)
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/v1/models").json()["data"][0]["id"] == "echo-model"
    finally:
        server.shutdown()


def test_proxy_anthropic_and_streaming_over_real_http():
    server = HTTPServer(("127.0.0.1", 0), _EchoUpstream)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        app = build_app(FoveanceProxy(budget=120), upstream_url=f"http://127.0.0.1:{port}/v1")
        client = TestClient(app)
        # Anthropic Messages route: context folded into system, last turn kept verbatim
        big = "FACT token=SECRET42\n" + "\n".join(f"log {i} ok" for i in range(150))
        areq = {"model": "claude-x", "system": "be brief",
                "messages": [{"role": "user", "content": big},
                             {"role": "assistant", "content": "ok"},
                             {"role": "user", "content": "recall token"}]}
        ar = client.post("/v1/messages", json=areq, headers={"x-api-key": "sk-test"})
        assert ar.status_code == 200
        fwd = _EchoUpstream.received
        assert fwd["messages"][-1]["content"] == "recall token"
        assert "be brief" in fwd["system"]
        # streaming passthrough: stream:true returns the upstream SSE bytes verbatim
        sr = client.post("/v1/chat/completions",
                         json={"model": "x", "stream": True, "user": "s1",
                               "messages": [{"role": "user", "content": "hi " + "x " * 200},
                                            {"role": "user", "content": "go"}]})
        assert sr.status_code == 200
        assert "chunk" in sr.text and "[DONE]" in sr.text
    finally:
        server.shutdown()
