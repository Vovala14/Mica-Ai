"""Guard the live feedback path before it becomes training data."""

import json
import sys

from r1.data import ingest_site_logs as ingest


def test_ember_v031_correction_completes_the_typed_word():
    rows = [
        {"kind": "generation", "id": "a", "model": "ember-v0.3.1", "mode": "complete-word",
         "prompt": "Thank you for be", "output": "ing"},
        {"kind": "feedback", "id": "a", "time": "2026-10-08T00:00:00Z",
         "correction": "tter"},
    ]
    assert ingest.texts(rows, with_prompts=True) == [
        ("correction", "Thank you for better", "Thank you for be")
    ]


def test_fixed_rating_prompt_never_enters_training():
    rows = [
        {"kind": "generation", "id": "a", "model": "flame-w-0.3.1", "mode": "sentence",
         "prompt": "I lost my keys and", "output": " I found them.", "prompt_id": "s4"},
        {"kind": "feedback", "id": "a", "time": "2026-10-08T00:00:00Z",
         "rating": "up", "correction": " I looked for them."},
        {"kind": "generation", "id": "b", "model": "ember-v0.3.1", "mode": "next-word",
         "prompt": "A separate free prompt", "output": " works"},
        {"kind": "feedback", "id": "b", "time": "2026-10-08T00:00:00Z",
         "rating": "up"},
    ]
    assert ingest.texts(rows, with_prompts=True) == [
        ("thumbs_up", "A separate free prompt works", "A separate free prompt")
    ]


def test_latest_feedback_replaces_an_earlier_correction():
    rows = [
        {"kind": "generation", "id": "a", "model": "flame-w-0.3.1", "mode": "sentence",
         "prompt": "A free prompt", "output": " wrong"},
        {"kind": "feedback", "id": "a", "time": "2026-10-08T00:00:00Z",
         "rating": "up", "correction": " proposed ending"},
        {"kind": "feedback", "id": "a", "time": "2026-10-08T00:01:00Z",
         "rating": "down", "correction": ""},
    ]
    assert ingest.texts(rows, with_prompts=True) == [("prompt", "A free prompt", "A free prompt")]


def test_different_corrections_to_same_prompt_stay_in_one_split(tmp_path, monkeypatch):
    rows = [
        {"kind": "generation", "id": "a", "model": "flame-w-0.3.1", "mode": "sentence",
         "prompt": "A useful start", "output": " wrong"},
        {"kind": "feedback", "id": "a", "time": "2026-10-08T00:00:00Z",
         "correction": " first ending"},
        {"kind": "generation", "id": "b", "model": "flame-w-0.3.1", "mode": "sentence",
         "prompt": "A useful start", "output": " wrong"},
        {"kind": "feedback", "id": "b", "time": "2026-10-08T00:00:01Z",
         "correction": " second ending"},
    ]
    export = tmp_path / "export.jsonl"
    export.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    class TinyVocab:
        def encode(self, tokens):
            return [1] if tokens else []

    monkeypatch.setattr(ingest.W.Vocab, "load", lambda _: TinyVocab())
    monkeypatch.setattr(ingest.W, "pack", lambda _: "01")
    monkeypatch.setattr(sys, "argv", ["ingest_site_logs.py", str(export), "--out", str(tmp_path / "out"),
                                   "--val-percent", "50"])
    assert ingest.main() == 0
    train = (tmp_path / "out/ember/train.jsonl").read_text(encoding="utf-8").splitlines()
    val = (tmp_path / "out/ember/val.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted((len(train), len(val))) == [0, 2]
