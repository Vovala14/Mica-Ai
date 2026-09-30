"""Synthetic checks for the opt-in retrieval sentence-form veto."""

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from mica_r1.sentence_form_guard import sentence_form_issue, veto_sentence_fragments


@pytest.mark.parametrize("text,reason", [
    ("A bunch of people.", "nominal_fragment"),
    ("A bunch of water balloons.", "nominal_fragment"),
    ("A meal with the people there.", "nominal_fragment"),
    ("A dog jumping over a log.", "nominal_fragment"),
    ("The red bicycle in the garage.", "nominal_fragment"),
    ("Local clubs for years.", "nominal_fragment"),
    ("The book that I read.", "nominal_fragment"),
    ("Because the parents of other children.", "missing_main_clause"),
    ("When the war broke out.", "missing_main_clause"),
    ("If I don't do that.", "missing_main_clause"),
    ("Because I was tired, in the morning.", "missing_main_clause"),
    ("Can you help me.", "question_terminal_punctuation"),
    ("What is your name.", "question_terminal_punctuation"),
    ('"Can you help me."', "question_terminal_punctuation"),
    ("The children played outside", "missing_terminal_punctuation"),
])
def test_clear_sentence_form_errors_are_vetoed(text, reason):
    assert sentence_form_issue(text) == reason


@pytest.mark.parametrize("text", [
    "A group of boys play baseball.",
    "A group of boys plays baseball.",
    "The dog is jumping over a log.",
    "The room filled with smoke.",  # Past tense / participle is ambiguous.
    "This works.",
    "That appeals to me.",
    "Those differ in color.",
    "The dogs frolic in the park.",
    "A dog jumping up purrs.",
    "When the war broke out, families fled.",
    "If it rains we stay inside.",
    "Because I was tired, I slept.",
    "When did the war break out?",
    "A dog that barks bites.",
    "Give me five minutes and I'll find out.",
])
def test_plausible_complete_sentences_survive(text):
    assert sentence_form_issue(text) is None


@dataclass(frozen=True)
class Suggestion:
    full_text: str


@dataclass(frozen=True)
class Result:
    suggestions: tuple[Suggestion, ...]
    abstained: bool
    reason: str
    method: str = "synthetic retrieval"


def test_veto_keeps_ranked_replacement_and_abstains_if_none_survive():
    bad = Suggestion("A dog jumping over a log.")
    good = Suggestion("A dog jumps over a log.")
    result = Result((bad, good), False, "ok")
    filtered = veto_sentence_fragments(result)
    assert filtered.suggestions == (good,)
    assert not filtered.abstained
    assert filtered.method == result.method
    assert veto_sentence_fragments(result, mode="short") is result
    empty = veto_sentence_fragments(Result((bad,), False, "ok"))
    assert empty.suggestions == ()
    assert empty.abstained
    assert empty.reason == "sentence_form_veto"
