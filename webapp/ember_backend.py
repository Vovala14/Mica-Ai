"""Ember v0.3.7 word suggestions from the native integer cellular engine."""
from __future__ import annotations

import re
import sys
import time

from .common import ROOT, clean_prompt, result

sys.path.insert(0, str(ROOT / "ember/runs/ember_v037_20261011"))
from word_model import EmberWord  # noqa: E402

MODEL = "ember-v0.3.7"
SOURCE_SHA = "c2bef375c8dcd6f944e1a44f4b92869645ddf97eb386382fa322cf9519eda506"
HEAD_SHA = "40f3bfa6f9692f4424d05cc3666afdca7ecbd648244df059ce2d181f66561e67"
TYPED = re.compile(r"(?:^|\s)([A-Za-z]+)$")

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
    t0 = time.perf_counter()
    extra = {}
    if mode == "complete-word":
        typed = TYPED.search(prompt)
        if not typed:
            raise ValueError("End your prompt with the first letters of a word, such as 'be'.")
        suggestions = model().complete_with_probabilities(prompt, top_k=5)
        words = [word for word, probability in suggestions]
        prefix_length = len(typed.group(1))
        alpha = model().completion_alpha if prefix_length >= 2 else 0
        gain = model().completion_probability_gain if prefix_length >= 2 else 16
        extra = dict(suggestion_probabilities=[probability for word, probability in suggestions],
                     probability_scope="conditional-on-typed-prefix",
                     completion_alpha=alpha,
                     completion_probability_gain=gain)
        output = words[0][len(typed.group(1)):] if words else ""
    else:
        context = prompt if prompt[-1].isspace() else prompt + " "
        words = model().next_words(context, top_k=5)
        output = ("" if prompt[-1].isspace() else " ") + words[0] if words else ""

    return result(MODEL, mode, prompt, output, t0,
                  suggestions=words, decoder=(f"integer-completion-a{alpha}-gain{gain}-v1" if mode == "complete-word"
                                             else "integer-word-readout-v0.3.7"),
                  **extra,
                  model_sha256=SOURCE_SHA, word_heads_sha256=HEAD_SHA)
