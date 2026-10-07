"""The committed Flame-W 0.3.1 replay file matches its prompts and checkpoint ids.

This does not rerun the automaton. `replay_flamew_031.py --check` does that.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUTOMATON = "1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d"
MEMORY = "5fda5a6c744749932bfba7f18352ce532405e60fca7d64fd99b705c28b928be7"


def _lines(name):
    return [json.loads(line) for line in (HERE / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_replay_files_align():
    prompts = _lines("prompts.jsonl")
    outs = _lines("out.jsonl")
    assert [p["id"] for p in prompts] == [o["id"] for o in outs]
    raw = (HERE / "out.jsonl").read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    for p, o in zip(prompts, outs):
        assert o["prompt"] == p["prompt"]
        assert o["text"] == o["prompt"] + o["continuation"]
        assert o["continuation"].startswith(" ")
        assert o["model"] == "flame-w-0.3.1"
        assert o["decoder"] == "w-sent-bos.5f"
        assert o["answer_mode"] is False
        assert o["automaton_sha256"] == AUTOMATON
        assert o["memory_sha256"] == MEMORY
    by_id = {o["id"]: o["text"] for o in outs}
    assert by_id["s1"] == "I was thinking about how it would affect you."
    assert by_id["s4"] == "I lost my keys and I don't know how to fix it."
