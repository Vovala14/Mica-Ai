"""Behavioral checks for the optional whole-word suggestion decoder."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from mica_r1.suggest import Vocabulary, _log_probs, suggest


class FakeScorer:
    """A branchable model with prompt-dependent byte logits, no MICA/GPU."""

    def __init__(self, preferences):
        self.preferences = preferences

    def start(self, prompt):
        return prompt

    def scores(self, state):
        scores = np.full(258, -10.0)
        for byte, value in self.preferences.get(state, {}).items():
            scores[byte] = value
        return scores

    def advance(self, state, byte):
        return state + bytes((byte,))


def best(scorer, words, prompt, **kwargs):
    result = suggest(scorer, Vocabulary(words), prompt,
                     min_confidence=0.0, **kwargs)
    assert not result.abstained, result.reason
    return result.suggestions[0].continuation


def test_vocab_reads_hex_train_records_and_keeps_unicode(tmp_path):
    corpus = tmp_path / "train.jsonl"
    corpus.write_text("\n".join(s.encode("utf-8").hex() for s in
                                ("café cat", "café dog", "cat bird")) + "\n")
    vocab = Vocabulary.from_corpus(corpus, min_count=2)
    assert vocab.size == 2
    assert vocab.at_prefix("caf").children
    assert vocab.at_prefix("dog") is None


def test_partial_word_finishes_on_utf8_boundary():
    scorer = FakeScorer({
        b"caf": {0xC3: 8}, b"caf\xc3": {0xA9: 8},
        b"caf\xc3\xa9": {257: 8},
    })
    assert best(scorer, ["café"], "caf", max_words=1) == "é"


def test_punctuation_prompt_gets_space_and_completed_word_can_get_period():
    scorer = FakeScorer({
        b"Hi,": {ord(" "): 8}, b"Hi, ": {ord("c"): 8},
        b"Hi, c": {ord("a"): 8}, b"Hi, ca": {ord("t"): 8},
        b"Hi, cat": {ord("."): 8},
    })
    assert best(scorer, ["cat"], "Hi,", max_words=1) == " cat."


def test_two_word_completion_uses_space_without_leaving_partial_word():
    scorer = FakeScorer({
        b"": {ord("c"): 8}, b"c": {ord("a"): 8},
        b"ca": {ord("t"): 8}, b"cat": {ord(" "): 8},
        b"cat ": {ord("d"): 8}, b"cat d": {ord("o"): 8},
        b"cat do": {ord("g"): 8}, b"cat dog": {257: 8},
    })
    assert best(scorer, ["cat", "dog"], "", beam_width=2,
                max_words=2) == "cat dog"


def test_sentence_mode_requires_multiple_words_and_terminal_punctuation():
    scorer = FakeScorer({
        b"": {ord("c"): 8}, b"c": {ord("a"): 8},
        b"ca": {ord("t"): 8}, b"cat": {257: 10, ord(" "): 9},
        b"cat ": {ord("d"): 8}, b"cat d": {ord("o"): 8},
        b"cat do": {ord("g"): 8}, b"cat dog": {ord(" "): 10, ord("."): 9},
    })
    assert best(scorer, ["cat", "dog"], "", max_words=2,
                min_words=2, require_terminal_punctuation=True) == "cat dog."


def test_immediate_word_and_two_word_loops_are_blocked():
    scorer = FakeScorer({
        b"cat dog cat ": {ord("d"): 9, ord("b"): 8},
        b"cat dog cat b": {ord("i"): 8},
        b"cat dog cat bi": {ord("r"): 8},
        b"cat dog cat bir": {ord("d"): 8},
        b"cat dog cat bird": {257: 8},
    })
    assert best(scorer, ["dog", "bird"], "cat dog cat ",
                beam_width=2, max_words=1) == "bird"
    scorer2 = FakeScorer({b"cat ": {ord("c"): 9, ord("d"): 8},
                          b"cat d": {ord("o"): 8},
                          b"cat do": {ord("g"): 8},
                          b"cat dog": {257: 8}})
    assert best(scorer2, ["cat", "dog"], "cat ",
                beam_width=2, max_words=1) == "dog"


def test_beam_recovers_when_greedy_byte_has_bad_word_boundary():
    scorer = FakeScorer({
        b"": {ord("a"): 8, ord("b"): 7},
        b"a": {ord("x"): 10},  # x is outside vocabulary; boundaries poor
        b"b": {257: 10},
    })
    assert best(scorer, ["a", "b"], "", mode="greedy", max_words=1) == "a"
    assert best(scorer, ["a", "b"], "", mode="beam",
                beam_width=2, max_words=1) == "b"


def test_ambiguous_found_words_trigger_abstention():
    scorer = FakeScorer({
        b"": {ord("a"): 8, ord("b"): 8},
        b"a": {257: 8}, b"b": {257: 8},
    })
    result = suggest(scorer, Vocabulary(["a", "b"]), "", max_words=1,
                     beam_width=2, min_confidence=0.6)
    assert result.abstained
    assert result.reason == "low_search_confidence"
    assert result.suggestions[0].confidence <= 0.5
    assert result.suggestions[0].continuation == "a"  # stable lexical tie


def test_unknown_prefix_abstains_without_leaking_partial_bytes():
    result = suggest(FakeScorer({}), Vocabulary(["cat"]), "ele",
                     min_confidence=0.0)
    assert result.abstained
    assert result.reason == "prefix_not_in_vocabulary"
    assert not result.suggestions


def test_complete_final_prompt_word_can_start_a_new_word():
    scorer = FakeScorer({
        b"cat": {ord(" "): 9, ord("."): 10},
        b"cat ": {ord("d"): 9}, b"cat d": {ord("o"): 9},
        b"cat do": {ord("g"): 9}, b"cat dog": {257: 9},
    })
    assert best(scorer, ["cat", "dog"], "cat", max_words=1) == " dog"


def test_complete_word_with_longer_match_can_take_either_path():
    scorer = FakeScorer({
        b"car": {ord(" "): 9, ord("t"): 8},
        b"car ": {ord("d"): 9}, b"car d": {ord("o"): 9},
        b"car do": {ord("g"): 9}, b"car dog": {257: 9},
        b"cart": {257: 9},
    })
    result = suggest(scorer, Vocabulary(["car", "cart", "dog"]), "car",
                     beam_width=2, max_words=1, min_confidence=0.0)
    assert {s.continuation for s in result.suggestions} >= {" dog", "t"}


def test_known_complete_final_word_forces_next_word():
    scorer = FakeScorer({
        b"car": {ord("t"): 10, ord(" "): 9},
        b"car ": {ord("d"): 9}, b"car d": {ord("o"): 9},
        b"car do": {ord("g"): 9}, b"car dog": {257: 9},
        b"cart": {257: 9},
    })
    assert best(scorer, ["car", "cart", "dog"], "car", max_words=1,
                complete_final_word=True) == " dog"
    assert best(scorer, ["dog"], "car", max_words=1,
                complete_final_word=True) == " dog"


def test_bos_is_not_in_the_generation_distribution():
    logits = np.zeros(258)
    logits[256] = 100
    probabilities = np.exp(_log_probs(logits))
    assert probabilities[256] == 0
    assert np.isclose(probabilities.sum(), 1)
