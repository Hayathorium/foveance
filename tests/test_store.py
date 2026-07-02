"""Tests for the multi-fidelity store and its renderers."""
import pytest

from foveance.store import MultiFidelityStore, Item, Fidelity, default_renderer


def _item(text, iid="x", kind="tool_output", turn=0):
    return Item(item_id=iid, kind=kind, full_text=text, created_turn=turn)


def test_fidelity_ordering():
    assert Fidelity.POINTER < Fidelity.GIST < Fidelity.DIGEST < Fidelity.FULL
    assert int(Fidelity.FULL) == 3


def test_content_hash_stable():
    it = _item("hello world")
    assert it.content_hash() == _item("hello world").content_hash()
    assert it.content_hash() != _item("other").content_hash()


def test_renderer_full_returns_verbatim():
    text = "line one\nline two"
    assert default_renderer(_item(text), Fidelity.FULL) == text


def test_renderer_digest_short_text_unchanged():
    text = "a\nb\nc"  # <= 6 non-empty lines -> returned as-is
    assert default_renderer(_item(text), Fidelity.DIGEST) == text


def test_renderer_digest_preserves_salient_lines():
    lines = ["header"] + [f"log line {i} status=ok" for i in range(20)]
    lines.insert(10, "FACT k1=v9999")
    lines.insert(11, "ERROR disk full")
    lines.insert(12, "DECISION=use-cache")
    text = "\n".join(lines)
    digest = default_renderer(_item(text), Fidelity.DIGEST)
    assert "FACT k1=v9999" in digest
    assert "ERROR disk full" in digest
    assert "DECISION=use-cache" in digest
    assert "boilerplate lines omitted" in digest
    assert len(digest) < len(text)


def test_renderer_gist_one_line_and_pointer():
    text = "first line of output\nsecond line\nthird"
    gist = default_renderer(_item(text), Fidelity.GIST)
    assert gist.count("\n") == 0
    assert "first line of output" in gist
    pointer = default_renderer(_item(text), Fidelity.POINTER)
    assert "retrievable" in pointer and "tool_output#x" in pointer


def test_renderer_gist_empty_text():
    gist = default_renderer(_item(""), Fidelity.GIST)
    assert "tool_output#x" in gist


def test_store_add_render_cost_cache():
    store = MultiFidelityStore()
    store.add(_item("FACT a=1\n" + "noise\n" * 30, iid="i0"))
    full = store.render("i0", Fidelity.FULL)
    assert store.render("i0", Fidelity.FULL) is full  # cached object identity
    assert store.cost("i0", Fidelity.FULL) > store.cost("i0", Fidelity.POINTER)


def test_store_duplicate_id_raises():
    store = MultiFidelityStore()
    store.add(_item("a", iid="dup"))
    with pytest.raises(KeyError):
        store.add(_item("b", iid="dup"))


def test_store_retrieve_full_marks_referenced():
    store = MultiFidelityStore()
    store.add(_item("payload", iid="i0"))
    assert store.retrieve_full("i0", turn=7) == "payload"
    assert store.items["i0"].last_referenced_turn == 7


def test_store_assemble_orders_and_counts():
    store = MultiFidelityStore(token_counter=lambda s: len(s.split()))
    store.add(_item("alpha beta", iid="i0"))
    store.add(_item("gamma delta", iid="i1"))
    ctx, ntok = store.assemble({"i0": Fidelity.FULL, "i1": Fidelity.FULL}, system="SYS")
    assert ctx.startswith("SYS")
    assert "alpha beta" in ctx and "gamma delta" in ctx
    assert ntok == len(ctx.split())


def test_store_assemble_defaults_to_pointer():
    store = MultiFidelityStore()
    store.add(_item("x" * 200, iid="i0"))
    ctx, _ = store.assemble({})  # no level given -> POINTER
    assert "retrievable" in ctx
