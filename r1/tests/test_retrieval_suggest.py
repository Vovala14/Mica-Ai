"""Focused checks for the separate, train-only retrieval decoder."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from mica_r1.retrieval_suggest import (
    Sentence, SentenceIndex, build_index, suggest_retrieval,
    suggest_retrieval_fallback,
)


class FakeScorer:
    def __init__(self, preferred: bytes = b""):
        self.preferred = preferred
        self.starts = 0

    def start(self, prompt: bytes):
        self.starts += 1
        return prompt

    def scores(self, state: bytes):
        values = np.zeros(258)
        base = b"The quiet room"
        if self.preferred and state.startswith(base):
            offset = len(state) - len(base)
            if offset < len(self.preferred):
                values[self.preferred[offset]] = 6
        return values

    def advance(self, state: bytes, byte: int):
        return state + bytes((byte,))


def fixture_index(sentences):
    entries = [Sentence(text, n, 0, f"{n:064x}")
               for n, text in enumerate(sentences, 1)]
    return SentenceIndex(entries, {"source_split": "train"})


def test_build_scans_whole_train_and_excludes_heldout(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    train = corpus / "train.txt"
    train.write_text("One sunny afternoon the children walked through the park.\n"
                     "The quiet room filled with music as friends arrived.\n"
                     "The quiet room felt calm after the visitors left.\n"
                     "A small bird sang beside the open kitchen window.\n",
                     encoding="utf-8")
    (corpus / "val.txt").write_text(
        "The quiet room felt calm after the visitors left.\n", encoding="utf-8")
    (corpus / "test.txt").write_text(
        "A small bird sang beside the open kitchen window.\n", encoding="utf-8")
    out = tmp_path / "index"
    manifest = build_index(train, out, max_sentences=1)
    loaded = SentenceIndex.load(out)
    assert manifest["counts"]["records_scanned"] == 4
    assert manifest["selected_sentences"] == 1
    assert manifest["counts"]["heldout_rejected"] == 2
    assert loaded.sentences[0].train_line in (1, 2)
    assert loaded.sentences[0].text not in {
        "The quiet room felt calm after the visitors left.",
        "A small bird sang beside the open kitchen window."}
    # The same files always produce the same hash-priority selection.
    second = tmp_path / "second"
    build_index(train, second, max_sentences=1)
    assert (out / "sentences.jsonl").read_bytes() == (second / "sentences.jsonl").read_bytes()


def test_build_rejects_val_as_source_and_missing_exclusions(tmp_path):
    (tmp_path / "train.txt").write_text("A simple sentence with many words is here.\n")
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path / "train.txt", tmp_path / "index")
    (tmp_path / "val.txt").write_text("")
    (tmp_path / "test.txt").write_text("")
    with pytest.raises(ValueError, match="train.txt"):
        build_index(tmp_path / "val.txt", tmp_path / "index")


def test_sentence_and_short_modes_keep_source_attribution():
    index = fixture_index([
        "The quiet room filled with music as friends arrived.",
        "The quiet room felt calm after the visitors left.",
    ])
    scorer = FakeScorer()
    sentence = suggest_retrieval(scorer, index, "The quiet room", mode="sentence")
    short = suggest_retrieval(scorer, index, "The quiet room", mode="short",
                              max_words=2)
    assert sentence.suggestions
    assert sentence.suggestions[0].full_text.endswith(".")
    assert sentence.suggestions[0].source_sentence in {
        item.text for item in index.sentences}
    assert len(short.suggestions[0].continuation.strip().split()) == 2
    assert short.suggestions[0].source_train_line in (1, 2)
    assert scorer.starts == 2


def test_partial_word_and_model_rerank_within_same_match_length():
    index = fixture_index([
        "The quiet room filled with music as friends arrived.",
        "The quiet room felt calm after the visitors left.",
    ])
    result = suggest_retrieval(FakeScorer(), index, "The quiet room fe",
                                mode="sentence")
    assert not result.abstained
    assert result.suggestions[0].continuation.startswith("lt calm")
    assert result.suggestions[0].matched_words >= 2
    reranked = suggest_retrieval(FakeScorer(b" filled"), index,
                                  "The quiet room", mode="sentence")
    assert reranked.suggestions[0].continuation.startswith(" filled")


def test_lexical_match_dominates_model_and_no_match_abstains():
    index = fixture_index([
        "People in the quiet room found a peaceful place to read.",
        "The quiet room filled with music as friends arrived.",
    ])
    result = suggest_retrieval(FakeScorer(b" found"), index, "The quiet room",
                                mode="sentence")
    assert result.suggestions[0].matched_words == 3
    missing = suggest_retrieval(FakeScorer(), index,
                                 "The ancient telescope rusted")
    assert missing.abstained
    assert missing.reason == "no_two_word_train_match"


def test_exact_whole_word_wins_over_a_longer_partial_word():
    index = fixture_index([
        "The quiet room felt calm after the visitors left.",
        "The quiet roommates shared a long conversation over dinner.",
    ])
    result = suggest_retrieval(FakeScorer(), index, "The quiet room",
                                mode="sentence")
    assert result.suggestions
    assert all(s.source_sentence.startswith("The quiet room felt")
               for s in result.suggestions)


def test_space_and_comma_prompt_join_without_duplicate_punctuation():
    index = fixture_index([
        "After dinner, the guests sat together in the garden.",
    ])
    with_space = suggest_retrieval(FakeScorer(), index, "After dinner, ")
    with_comma = suggest_retrieval(FakeScorer(), index, "After dinner,")
    assert with_space.suggestions[0].full_text == (
        "After dinner, the guests sat together in the garden.")
    assert with_comma.suggestions[0].full_text == (
        "After dinner, the guests sat together in the garden.")


def test_generic_tail_requires_topic_match_and_recognizes_raining_as_rain():
    mismatched = fixture_index([
        "The basketball player dunked the ball so hard that the backboard shattered.",
    ])
    result = suggest_retrieval(FakeScorer(), mismatched,
                                "It was raining so hard that")
    assert result.abstained
    related = fixture_index([
        "The basketball player dunked the ball so hard that the backboard shattered.",
        "Rain fell so hard that the road flooded overnight.",
    ])
    result = suggest_retrieval(FakeScorer(), related,
                                "It was raining so hard that")
    assert not result.abstained
    assert result.suggestions[0].continuation == " the road flooded overnight."


def test_long_prompt_does_not_join_unrelated_take_the_source():
    index = fixture_index([
        "We're not going to take the children to school.",
    ])
    result = suggest_retrieval(FakeScorer(), index,
                                "To reach the airport before rush hour, take the")
    assert result.abstained


def test_secondary_corpus_used_only_after_primary_abstains():
    primary = fixture_index([
        "The quiet room felt calm after the visitors left.",
    ])
    secondary = fixture_index([
        "Rain fell so hard that the road flooded overnight.",
    ])
    prompt = "It was raining so hard that"
    result = suggest_retrieval_fallback(FakeScorer(), [primary, secondary],
                                        prompt)
    assert not result.abstained
    assert result.suggestions[0].source_sentence == secondary.sentences[0].text
    direct = suggest_retrieval_fallback(FakeScorer(), [primary, secondary],
                                        "The quiet room")
    assert direct.suggestions[0].source_sentence == primary.sentences[0].text


def test_question_mark_is_aligned_to_prompt_speech_act():
    index = fixture_index([
        "Could you please close the windows before you leave?",
    ])
    imperative = suggest_retrieval(FakeScorer(), index, "Please close the")
    question = suggest_retrieval(FakeScorer(), index,
                                 "Could you please close the")
    assert imperative.suggestions[0].full_text == (
        "Please close the windows before you leave.")
    assert imperative.suggestions[0].constructed_punctuation
    assert question.suggestions[0].full_text == (
        "Could you please close the windows before you leave?")
    assert not question.suggestions[0].constructed_punctuation


def test_purpose_tail_after_time_can_end_at_a_complete_clause():
    index = fixture_index([
        "Donald didn't arrive on time to meet Jessica.",
    ])
    result = suggest_retrieval(FakeScorer(), index,
                                "The package should arrive on")
    assert result.suggestions[0].full_text == "The package should arrive on time."
    assert result.suggestions[0].constructed_punctuation
    short = suggest_retrieval(FakeScorer(), index,
                              "The package should arrive on", mode="short")
    assert short.suggestions[0].full_text == "The package should arrive on time."


def test_train_cooccurrence_is_a_small_topic_bonus():
    index = fixture_index([
        "John might arrive on Tuesday.",
        "Paul might arrive on Friday.",
        "The parcel will come on Friday.",
        "A parcel can be sent on Friday.",
    ])
    result = suggest_retrieval(FakeScorer(), index,
                                "The parcel should arrive on")
    assert result.suggestions[0].continuation == " Friday."
    assert result.suggestions[0].topic_anchor == "parcel"
    assert result.suggestions[0].topic_joint_sentences == 2
