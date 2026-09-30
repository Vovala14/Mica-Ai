"""Focused checks for the opt-in corpus splice compatibility helper."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mica_r1.retrieval_compat import (
    check_splice_role,
    select_compatible_candidates,
)
from mica_r1.retrieval_suggest import Sentence, SentenceIndex


def fixture_index(sentences: list[str]) -> SentenceIndex:
    entries = [Sentence(text, n, 0, f"{n:064x}")
               for n, text in enumerate(sentences, 1)]
    return SentenceIndex(entries, {"source_split": "train"})


def test_noun_verb_role_conflict_vetoes_obvious_splice():
    source = "The paint the artist used was bright blue."
    prompt = "I need to paint the"
    role = check_splice_role(prompt, source,
                             " artist used was bright blue.", 2)
    assert not role.compatible
    assert role.reason == "matched_word_role_conflict:paint"
    selection = select_compatible_candidates(fixture_index([source]), prompt)
    assert selection.raw_matches == 1
    assert selection.role_rejected == 1
    assert not selection.candidates


def test_role_guard_accepts_matching_roles_and_unknown_cues():
    verb = "She planned to paint the fence before dinner."
    noun = "She admired the paint the artist chose for the wall."
    assert check_splice_role("I decided to paint the", verb,
                             " fence before dinner.", 3).compatible
    assert check_splice_role("I admired the paint the", noun,
                             " artist chose for the wall.", 3).compatible
    assert check_splice_role("Please paint the", verb,
                             " fence before dinner.", 2).compatible


def test_context_supported_shorter_match_can_beat_unrelated_longer_match():
    bad = "The little boy asked me to open my presents before breakfast."
    good = "At the dentist, I had to open my mouth very wide."
    prompt = "The dentist asked me to open my"
    selection = select_compatible_candidates(fixture_index([bad, good]), prompt)
    assert selection.role_rejected == 0
    assert [c.matched_words for c in selection.candidates] == [3, 5]
    assert selection.candidates[0].continuation == " mouth very wide."
    assert selection.candidates[0].context_terms == ("dentist",)


def test_missing_context_evidence_is_neutral_for_coverage():
    source = "The quiet room filled with music as friends arrived."
    selection = select_compatible_candidates(fixture_index([source]),
                                              "The quiet room")
    assert selection.candidates
    assert selection.candidates[0].continuation.startswith(" filled")
    assert selection.candidates[0].context_terms == ()


def test_full_prompt_match_stays_first_and_partial_words_stay_available():
    full = "The dentist asked me to open my mouth slowly."
    shorter = "At the dentist, I had to open my mouth very wide."
    index = fixture_index([full, shorter])
    prompt = "The dentist asked me to open my"
    selection = select_compatible_candidates(index, prompt)
    assert selection.candidates[0].source_sentence == full
    partial = select_compatible_candidates(index, "The dentist asked me to op")
    assert partial.candidates
    assert partial.candidates[0].partial_word


def test_role_guard_does_not_pretend_to_solve_topic_mismatch():
    # A grammatical long splice can still be about the wrong person.
    source = "The little boy asked me to open my presents before breakfast."
    selection = select_compatible_candidates(fixture_index([source]),
                                              "The dentist asked me to open my")
    assert selection.candidates
    assert selection.candidates[0].continuation.startswith(" presents")
