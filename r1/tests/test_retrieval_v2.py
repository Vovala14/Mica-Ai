"""Regression checks for the opt-in v2 retrieval decoder."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_retrieval_suggest import FakeScorer, fixture_index

from mica_r1.retrieval_v2 import _fixed_topic_overlap, suggest_retrieval_v2


def test_fixed_source_prefix_includes_word_directly_before_match():
    source = "A dog stood on the grass."
    index = fixture_index([source, "The cat rested quietly under a chair."])
    overlap = _fixed_topic_overlap(index,
                                   "The dog watched as he stood on the",
                                   source, " grass.", 3)
    assert overlap > 0


def test_v2_vetoes_noun_verb_role_conflict():
    index = fixture_index(["The paint the artist used was bright blue."])
    result = suggest_retrieval_v2(FakeScorer(), index,
                                  "I need to paint the")
    assert result.abstained
    assert result.reason == "no_role_compatible_continuation"


def test_v2_can_choose_topical_shorter_match():
    index = fixture_index([
        "The little boy asked me to open my presents before breakfast.",
        "At the dentist, I had to open my mouth very wide.",
    ])
    result = suggest_retrieval_v2(FakeScorer(), index,
                                  "The dentist asked me to open my")
    assert result.suggestions
    assert result.suggestions[0].continuation == " mouth very wide."
    assert result.suggestions[0].matched_words == 3


def test_v2_keeps_normal_source_completion():
    index = fixture_index([
        "The quiet room filled with music as friends arrived.",
    ])
    result = suggest_retrieval_v2(FakeScorer(), index, "The quiet room")
    assert result.suggestions
    assert result.suggestions[0].full_text == "The quiet room filled with music as friends arrived."
