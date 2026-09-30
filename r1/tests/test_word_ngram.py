"""Exercise model-only sentence construction from learned word transitions."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research_baselines.word_ngram import START, WordNgram


class UniformScorer:
    def start(self, prompt: bytes) -> bytes:
        return prompt

    def scores(self, state: bytes) -> np.ndarray:
        return np.zeros(258)

    def advance(self, state: bytes, byte: int) -> bytes:
        return state + bytes((byte,))


def test_word_model_constructs_complete_text_from_blank_prompt():
    model = WordNgram()
    assert model.add("The cat sleeps on a mat.")
    assert model.add("The dog sleeps on a rug.")
    assert not model.add("An unfinished fragment")
    assert model.next_probabilities((START, START))[0][0] == "the"

    result = model.generate(UniformScorer(), "", seed=4,
                            min_words=2, max_words=8)
    assert result.text[0].isupper()
    assert result.text.endswith(".")
    assert result.generated_words >= 2
    assert result.continuation == result.text
