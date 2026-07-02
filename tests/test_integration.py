"""Integration: a scripted long-horizon run where a needle planted early and queried late is
retained by budgeted policies within budget but dropped by `recency`."""
from foveance import Controller, Item
from foveance.llm import MockLLM


def _run(policy, budget, turns, drift=0.7):
    ctrl = Controller(MockLLM(), budget=budget, policy=policy, drift=drift, recency_k=3)
    n_recall = n_correct = in_tok = 0
    for t, (q, gold, item) in enumerate(turns):
        ctrl.add_item(item)
        rec = ctrl.step(q, t)
        in_tok += rec.input_tokens
        if gold:
            n_recall += 1
            n_correct += 1.0 if gold in rec.answer else 0.0
    return (n_correct / n_recall if n_recall else 0.0), in_tok


def _needle_trajectory():
    """Plant FACT secret=4242 at turn 0 buried in noise; query it 30 noisy turns later."""
    turns = []
    needle = "FACT secret=4242"
    big = needle + "\n" + "\n".join(f"log {i} status=ok latency={i}ms" for i in range(60))
    turns.append(("note secret", "", Item("obs0", "tool_output", big, 0)))
    for t in range(1, 31):
        noise = "\n".join(f"log {t}.{i} status=ok" for i in range(40))
        turns.append((f"note filler{t}", "", Item(f"obs{t}", "tool_output", noise, t)))
    turns.append(("recall secret", "4242", Item("obs31", "tool_output", "tail", 31)))
    return turns


def test_budgeted_retains_needle_recency_drops_it():
    turns = _needle_trajectory()
    budget = 600  # far smaller than full context; forces selection
    acc_recency, tok_recency = _run("recency", budget, turns)
    acc_foveance, tok_foveance = _run("foveance", budget, turns)
    acc_full, tok_full = _run("full", budget, turns)

    assert acc_full == 1.0                       # full always sees the needle
    assert acc_recency == 0.0                    # needle aged out of the last-k window
    assert acc_foveance == 1.0                     # value-based allocation keeps the needle
    assert tok_foveance < tok_full                 # at a fraction of full's tokens


def test_budgeted_dominates_recency_on_efficiency():
    turns = _needle_trajectory()
    acc_foveance, tok_foveance = _run("foveance", 600, turns)
    acc_recency, tok_recency = _run("recency", 600, turns)
    # foveance strictly Pareto-dominates recency here: higher accuracy, comparable/again-bounded tokens
    assert acc_foveance > acc_recency
