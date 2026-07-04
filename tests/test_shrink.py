"""The dead-simple public one-liner: foveance.shrink(messages)."""
import json

import foveance


def test_shrink_compresses_and_keeps_last_turn():
    """A long history shrinks; the final user turn and system message survive verbatim; the
    result is still a valid OpenAI-style messages list. No extras required (core install)."""
    big = "FACT token=SECRET42\n" + "\n".join(f"log line {i} status=ok" for i in range(300))
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": big},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "recall token"},
    ]
    out = foveance.shrink(messages, budget=120)

    assert isinstance(out, list) and all("role" in m and "content" in m for m in out)
    assert out[-1] == {"role": "user", "content": "recall token"}   # last turn verbatim
    assert any(m["role"] == "system" for m in out)                  # system preserved
    assert len(json.dumps(out)) < len(json.dumps(messages))         # actually smaller
    assert foveance.shrink(messages) is not messages                # returns a new list


def test_shrink_is_exported_and_default_budget_works():
    assert "shrink" in foveance.__all__
    out = foveance.shrink([{"role": "user", "content": "hello"}])
    assert out[-1]["content"] == "hello"


def test_shrink_anthropic_compresses_history():
    """ Ensure old turns shrink under a tight budget while system 
    and the final message remain intact.
    """
    system_prompt = "You are an AI assistant specialized in Python."
    
    # A long message chain that will exceed a tiny token budget
    messages = [
        {"role": "user", "content": "Can you explain recursion? " * 100},
        {"role": "assistant", "content": "Recursion is when a function calls itself... " * 100},
        {"role": "user", "content": "Give me a simple example of it."}
    ]
    
    # Run with a small budget to force Foveance's allocation policy to kick in
    new_system, new_messages = foveance.shrink_anthropic(
        system=system_prompt, 
        messages=messages, 
        budget=100
    )
    
    # 1. System text must include the original system prompt
    assert system_prompt in new_system
    # 2. The compressed context from older turns should have been appended to system
    assert new_system != system_prompt
    
    # 3. Only the last turn should remain in the messages list
    assert len(new_messages) == 1
    assert new_messages[0]["role"] == "user"
    assert new_messages[0]["content"] == "Give me a simple example of it."
