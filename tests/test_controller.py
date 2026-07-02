"""Tests for the per-turn controller, policy seam, and two-sided refinement."""
import pytest

from foveance import Controller, Item, Fidelity
from foveance.llm import MockLLM, Completion, LLM


def _item(text, iid, turn=0):
    return Item(item_id=iid, kind="tool_output", full_text=text, created_turn=turn)


def test_run_result_aggregates():
    ctrl = Controller(MockLLM(), budget=2000, policy="foveance")
    ctrl.add_item(_item("FACT k1=v1\n" + "noise\n" * 20, "i0"))
    rec = ctrl.step("recall k1", 0)
    assert rec.input_tokens > 0 and rec.peak_tokens > 0
    assert "v1" in rec.answer


def test_drift_set_by_policy():
    assert Controller(MockLLM(), 1000, policy="reactive_afm").pred.cfg.drift == 0.0
    assert Controller(MockLLM(), 1000, policy="reactive").pred.cfg.drift == 0.0
    assert Controller(MockLLM(), 1000, policy="foveance", drift=0.7).pred.cfg.drift == 0.7


@pytest.mark.parametrize("policy", ["full", "recency", "reactive_afm", "foveance", "oracle", "noop"])
def test_all_policies_produce_valid_levels(policy):
    ctrl = Controller(MockLLM(), budget=1500, policy=policy, recency_k=2)
    for t in range(4):
        ctrl.add_item(_item("FACT k%d=v%d\n" % (t, t) + "x\n" * 15, f"i{t}", t))
    ctrl.pred.observe_query("recall k0")
    levels = ctrl._levels(turn=3)
    assert set(levels) == set(ctrl.store.order)
    assert all(isinstance(v, Fidelity) for v in levels.values())


def test_recency_keeps_last_k_full():
    ctrl = Controller(MockLLM(), budget=10_000, policy="recency", recency_k=2)
    for t in range(5):
        ctrl.add_item(_item("data " + str(t), f"i{t}", t))
    levels = ctrl._levels(turn=4)
    assert levels["i3"] == Fidelity.FULL and levels["i4"] == Fidelity.FULL
    assert levels["i0"] == Fidelity.POINTER


def test_unknown_policy_raises():
    ctrl = Controller(MockLLM(), 1000, policy="nope")
    ctrl.add_item(_item("a", "i0"))
    ctrl.pred.observe_query("q")
    with pytest.raises(ValueError):
        ctrl._levels(0)


class _RetrieveLLM(LLM):
    """A model that asks to re-inflate item i0 on the first turn only."""
    name = "retrieve"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, query):
        self.calls += 1
        text = "RETRIEVE i0" if self.calls == 1 else "answer: ok"
        return Completion(text, len(prompt) // 4, 3, 0.0)


def test_retrieve_tool_reinflates_next_turn():
    llm = _RetrieveLLM()
    ctrl = Controller(llm, budget=40, policy="foveance")  # tiny budget -> i0 would be POINTER
    ctrl.add_item(_item("FACT k1=v1\n" + "noise\n" * 40, "i0"))
    r0 = ctrl.step("recall k1", 0)
    assert r0.reinflations == 1
    assert ctrl.store.items["i0"].last_referenced_turn == 0
    ctrl.add_item(_item("more", "i1", 1))
    levels = ctrl._levels(turn=1)
    assert levels["i0"] == Fidelity.FULL  # forced re-inflation despite tiny budget


def test_retrieve_disabled():
    llm = _RetrieveLLM()
    ctrl = Controller(llm, budget=40, policy="foveance", retrieve_enabled=False)
    ctrl.add_item(_item("FACT k1=v1", "i0"))
    r0 = ctrl.step("recall k1", 0)
    assert r0.reinflations == 0


def test_retrieve_ignores_unknown_id():
    class _BadRetrieve(LLM):
        def generate(self, prompt, query):
            return Completion("RETRIEVE ghost", 1, 1, 0.0)
    ctrl = Controller(_BadRetrieve(), budget=1000, policy="foveance")
    ctrl.add_item(_item("a", "i0"))
    rec = ctrl.step("q", 0)
    assert rec.reinflations == 0


def test_fidelity_change_cost_charged_on_raise():
    # Raising an item's fidelity vs the previous turn costs extra "re-render" tokens.
    ctrl = Controller(MockLLM(), budget=10_000, policy="foveance", fidelity_cost=1.0)
    ctrl.add_item(_item("FACT k1=v1\n" + "noise\n" * 30, "i0"))
    r0 = ctrl.step("recall k1", 0)          # POINTER -> FULL raise charged on the first turn
    r1 = ctrl.step("recall k1", 1)          # already FULL -> no raise, no extra charge
    assert r0.input_tokens > r1.input_tokens
    # with fidelity_cost off, no extra accounting
    ctrl2 = Controller(MockLLM(), budget=10_000, policy="foveance", fidelity_cost=0.0)
    ctrl2.add_item(_item("FACT k1=v1\n" + "noise\n" * 30, "i0"))
    assert ctrl2._refidelity_cost({"i0": Fidelity.FULL}) == 0


def test_probe_separate_from_query():
    # The predictor observes the probe; the model still sees (and grades) the query.
    ctrl = Controller(MockLLM(), budget=2000, policy="foveance")
    ctrl.add_item(_item("FACT k1=v1\n" + "noise\n" * 10, "i0"))
    rec = ctrl.step("recall k1", 0, probe="advance step 0")
    assert "v1" in rec.answer
    assert ctrl.pred.posterior_debug()["n_queries"] == 1


def test_run_convenience_and_runresult_props():
    from dataclasses import dataclass

    @dataclass
    class T:
        query: str
        new_items: list

    turns = [T("recall k0", [_item("FACT k0=v0\n" + "n\n" * 10, "i0", 0)]),
             T("recall k0", [_item("more\n" * 5, "i1", 1)])]
    ctrl = Controller(MockLLM(), budget=2000, policy="foveance")
    res = ctrl.run(turns)
    assert len(res.records) == 2
    assert res.total_input > 0
    assert res.peak > 0
    assert res.wall_s >= 0.0
    assert res.total_reinflations == 0
    assert res.total_output >= 0
