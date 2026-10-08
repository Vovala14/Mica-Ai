"""Ember v0.3.1 word suggestions from the native integer cellular engine."""
from __future__ import annotations

import re
import sys
import time

from .common import ROOT, clean_prompt, result

sys.path.insert(0, str(ROOT / "ember/runs/ember_v031_20261008"))
from word_model import EmberWord  # noqa: E402

MODEL = "ember-v0.3.1"
SOURCE_SHA = "e35bad48e1db529ccc07f7bfbd5070ba01ec1d0dfbaeafaf07398c7bae63cedd"
HEAD_SHA = "cc8e0135ac38d8afcb11bce34d226e96372b8bd2fe377cae9c6a51dadadaa5d5"

_model: EmberWord | None = None


def model() -> EmberWord:
    global _model
    if _model is None:
        _model = EmberWord()  # verifies checkpoint and readout hashes
    return _model


def run(body: dict) -> dict:
    prompt = clean_prompt(body)
    mode = body.get("mode", "next-word")
    if mode not in ("next-word", "complete-word"):
        raise ValueError("Unknown Ember mode.")
    if mode == "complete-word" and not re.search(r"(?:^|\s)[A-Za-z]{2}$", prompt):
        raise ValueError("End your prompt with exactly two letters of a word, such as 'be'.")

    t0 = time.perf_counter()
    if mode == "complete-word":
        words = model().suggest(prompt, prefix2=True, top_k=5)
        output = words[0][2:] if words else ""
    else:
        context = prompt if prompt[-1].isspace() else prompt + " "
        words = model().suggest(context, top_k=5)
        output = ("" if prompt[-1].isspace() else " ") + words[0] if words else ""

    return result(MODEL, mode, prompt, output, t0,
                  suggestions=words, decoder="integer-word-readout-v0.3.1",
                  model_sha256=SOURCE_SHA, word_heads_sha256=HEAD_SHA)
