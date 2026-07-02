"""Tests for the anticipatory predictor -- the conceptual delta vs AFM."""
from foveance.store import MultiFidelityStore, Item, Fidelity
from foveance.predictor import AnticipatoryPredictor, PredictorConfig, _hash_embed, _cos


def _store_with(items):
    s = MultiFidelityStore()
    for iid, text in items:
        s.add(Item(item_id=iid, kind="tool_output", full_text=text, created_turn=0))
    return s


def test_hash_embed_deterministic_and_normalized():
    a = _hash_embed("the quick brown fox")
    b = _hash_embed("the quick brown fox")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_cos_bounds():
    v = _hash_embed("alpha beta")
    assert abs(_cos(v, v) - 1.0) < 1e-9
    assert _cos(v, []) == 0.0


def test_posterior_empty_and_single():
    s = _store_with([("i0", "alpha")])
    p = AnticipatoryPredictor(s)
    assert p._future_query_posterior() == []
    assert p.posterior_debug()["n_queries"] == 0
    p.observe_query("alpha")
    assert p._future_query_posterior() == p._query_history[-1]


def test_posterior_reactive_drift_zero():
    s = _store_with([("i0", "alpha")])
    p = AnticipatoryPredictor(s, config=PredictorConfig(drift=0.0))
    p.observe_query("alpha")
    p.observe_query("beta")
    # drift 0 -> posterior is exactly the current query embedding (reactive/AFM)
    assert p._future_query_posterior() == p._query_history[-1]
    dbg = p.posterior_debug()
    assert dbg["is_reactive"] and dbg["drift"] == 0.0


def test_posterior_anticipates_with_drift():
    s = _store_with([("i0", "alpha")])
    p = AnticipatoryPredictor(s, config=PredictorConfig(drift=0.9))
    p.observe_query("alpha beta")
    p.observe_query("beta gamma")
    fut = p._future_query_posterior()
    assert fut != p._query_history[-1]
    assert abs(sum(x * x for x in fut) - 1.0) < 1e-9  # renormalized


def test_anticipation_outranks_reactive_on_predicted_next():
    # Item matching the *predicted next* query should outrank one matching only the present.
    s = _store_with([("present", "apple apple apple"),
                     ("future", "banana banana banana")])
    react = AnticipatoryPredictor(s, config=PredictorConfig(drift=0.0))
    antic = AnticipatoryPredictor(_clone(s), config=PredictorConfig(drift=1.0))
    for p in (react, antic):
        p.observe_query("apple")          # last query: about 'present'
        p.observe_query("apple banana")   # momentum points toward 'banana'/'future'
    rv = react.base_value(s.items["future"], turn=2)
    av = antic.base_value(antic.store.items["future"], turn=2)
    assert av > rv  # anticipation lifts the future-relevant item


def _clone(store):
    s = MultiFidelityStore()
    for iid in store.order:
        it = store.items[iid]
        s.add(Item(item_id=iid, kind=it.kind, full_text=it.full_text, created_turn=it.created_turn))
    return s


def test_base_value_components():
    s = _store_with([("i0", "alpha")])
    p = AnticipatoryPredictor(s, config=PredictorConfig(
        recurrence_prior=0.5, kind_prior={"tool_output": 2.0}))
    p.observe_query("alpha")
    it = s.items["i0"]
    it.last_referenced_turn = 0  # triggers recurrence bonus
    v = p.base_value(it, turn=0)
    assert v > 0
    # kind prior doubles the value vs a neutral kind
    s2 = _store_with([("i0", "alpha")])
    p2 = AnticipatoryPredictor(s2, config=PredictorConfig(recurrence_prior=0.5))
    p2.observe_query("alpha")
    s2.items["i0"].last_referenced_turn = 0
    assert v > p2.base_value(s2.items["i0"], turn=0)


def test_value_curve_monotone_in_fidelity():
    s = _store_with([("i0", "alpha beta gamma")])
    p = AnticipatoryPredictor(s)
    p.observe_query("alpha")
    curve = p.value_curve("i0", turn=0)
    assert len(curve) == len(Fidelity)
    assert curve == sorted(curve)  # non-decreasing yield
    assert curve[Fidelity.POINTER] == 0.0


def test_learned_future_model_path():
    from foveance.learned import LogisticFutureRelevance
    s = _store_with([("i0", "alpha")])
    model = LogisticFutureRelevance(weights=[0.0, 5.0, 0.0, 0.0, 0.0, 0.0])  # weight on sim_future
    p = AnticipatoryPredictor(s, config=PredictorConfig(), future_model=model)
    p.observe_query("alpha")
    v = p.base_value(s.items["i0"], turn=0)
    assert v >= 0.0
