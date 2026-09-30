"""Experimental splice checks and candidate selection for corpus retrieval.

This is an opt-in companion to the frozen ``retrieval_suggest`` decoder. It
uses that module's raw matches, checks one high-confidence noun/verb ambiguity,
and keeps several match lengths available for a later reranker. No suggestion
is produced here: callers still need to score, shorten, and attribute the
selected source continuation.

Suggested integration: replace the frozen decoder's longest-match filtering
with ``select_compatible_candidates``. Pass the returned candidates through
the existing model/semantic reranker and abstain when none survive. Keep the
role check as a veto; use ``context_terms`` only as ranking evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .retrieval_suggest import (
    STOPWORDS,
    WORD,
    SentenceIndex,
    _matches,
    _topic_key,
)


DETERMINERS = frozenset(
    "a an the this that these those my your his her our their".split()
)
VERB_CUES = frozenset(
    "to can could will would shall should may might must do does did".split()
)
CLAUSE_PUNCTUATION = re.compile(r"[.!?;:,]")


@dataclass(frozen=True)
class RoleCheck:
    compatible: bool
    reason: str | None = None


@dataclass(frozen=True)
class CompatibleCandidate:
    sentence_index: int
    source_sentence: str
    continuation: str
    matched_words: int
    partial_word: bool
    topic_overlap: float
    context_terms: tuple[str, ...]


@dataclass(frozen=True)
class CandidateSelection:
    candidates: tuple[CompatibleCandidate, ...]
    raw_matches: int
    role_rejected: int


def _source_before_continuation(source: str, continuation: str) -> str | None:
    # _matches may strip leading whitespace from a continuation when the
    # prompt ends in a space, but the remaining text is still a source suffix.
    if not continuation or not source.endswith(continuation):
        return None
    return source[:-len(continuation)]


def _role_cue(text: str, words: list[re.Match[str]], position: int,
              next_word: str) -> str | None:
    """Classify only the unambiguous `the paint the` / `to paint the` shape."""
    if position == 0 or next_word not in DETERMINERS:
        return None
    previous = words[position - 1]
    current = words[position]
    if CLAUSE_PUNCTUATION.search(text[previous.end():current.start()]):
        return None
    token = previous.group().casefold()
    if token in DETERMINERS:
        return "noun"
    if token in VERB_CUES:
        return "verb"
    return None


def check_splice_role(prompt: str, source: str, continuation: str,
                      matched_words: int, *, partial_word: bool = False) -> RoleCheck:
    """Veto a matched word used as a noun on one side and a verb on the other.

    An unknown role is accepted. This deliberately catches a narrow,
    high-precision pattern rather than attempting to parse arbitrary English.
    """
    if partial_word or matched_words < 2:
        return RoleCheck(True)
    prefix = _source_before_continuation(source, continuation)
    if prefix is None:
        return RoleCheck(True)
    prompt_words = list(WORD.finditer(prompt))
    source_words = list(WORD.finditer(prefix))
    if min(len(prompt_words), len(source_words)) < matched_words:
        return RoleCheck(True)
    p_start = len(prompt_words) - matched_words
    s_start = len(source_words) - matched_words
    for offset in range(matched_words - 1):
        p_pos = p_start + offset
        s_pos = s_start + offset
        if (prompt_words[p_pos].group().casefold() !=
                source_words[s_pos].group().casefold()):
            continue
        following = prompt_words[p_pos + 1].group().casefold()
        if following != source_words[s_pos + 1].group().casefold():
            continue
        prompt_role = _role_cue(prompt, prompt_words, p_pos, following)
        source_role = _role_cue(prefix, source_words, s_pos, following)
        if prompt_role and source_role and prompt_role != source_role:
            word = prompt_words[p_pos].group().casefold()
            return RoleCheck(False, f"matched_word_role_conflict:{word}")
    return RoleCheck(True)


def _context_terms(index: SentenceIndex, prompt: str, source: str,
                   continuation: str, matched_words: int) -> tuple[str, ...]:
    """Find distinctive words shared *before* the splice on both sides."""
    prefix = _source_before_continuation(source, continuation)
    if prefix is None:
        return ()
    prompt_words = [m.group().casefold() for m in WORD.finditer(prompt)]
    source_words = [m.group().casefold() for m in WORD.finditer(prefix)]
    if min(len(prompt_words), len(source_words)) < matched_words:
        return ()
    before_prompt = prompt_words[:-matched_words]
    before_source = source_words[:-matched_words]
    source_keys = {_topic_key(word) for word in before_source
                   if word not in STOPWORDS}
    supported = set()
    for word in before_prompt:
        if word in STOPWORDS or len(word) < 4:
            continue
        if _topic_key(word) not in source_keys:
            continue
        # A common word is weak topic evidence. The posting count is an
        # inexpensive upper bound on sentence frequency, including repeats.
        if len(index.postings.get(word, ())) * 2 <= len(index.sentences):
            supported.add(word)
    return tuple(sorted(supported))


def select_compatible_candidates(index: SentenceIndex, prompt: str, *,
                                 min_match_words: int = 2,
                                 max_match_words: int = 8,
                                 max_length_gap: int = 3,
                                 per_length: int = 12) -> CandidateSelection:
    """Keep role-compatible matches from several trailing-match lengths.

    A shorter candidate with explicit source/prompt context overlap ranks
    ahead of an unsupported longer candidate within ``max_length_gap`` words.
    An exact match of the whole prompt remains first because it has no earlier
    context to compare. Missing overlap alone never rejects a candidate.
    """
    if min_match_words < 2 or max_match_words < min_match_words:
        raise ValueError("require at least a two-word trailing match")
    if max_length_gap < 0 or per_length < 1:
        raise ValueError("max_length_gap must be nonnegative; per_length positive")
    matches = _matches(index, prompt, min_match_words=min_match_words,
                       max_match_words=max_match_words)
    if any(not match.partial_word for match in matches):
        matches = [match for match in matches if not match.partial_word]
    raw_matches = len(matches)
    candidates: list[CompatibleCandidate] = []
    rejected = 0
    for match in matches:
        source = index.sentences[match.sentence_index].text
        role = check_splice_role(prompt, source, match.continuation,
                                match.matched_words,
                                partial_word=match.partial_word)
        if not role.compatible:
            rejected += 1
            continue
        candidates.append(CompatibleCandidate(
            sentence_index=match.sentence_index,
            source_sentence=source,
            continuation=match.continuation,
            matched_words=match.matched_words,
            partial_word=match.partial_word,
            topic_overlap=match.topic_overlap,
            context_terms=_context_terms(index, prompt, source,
                                         match.continuation,
                                         match.matched_words),
        ))
    if not candidates:
        return CandidateSelection((), raw_matches, rejected)
    best_length = max(c.matched_words for c in candidates)
    candidates = [c for c in candidates
                  if c.matched_words >= best_length - max_length_gap]
    prompt_length = len(list(WORD.finditer(prompt)))
    use_context = best_length < prompt_length and any(
        c.context_terms for c in candidates)
    candidates.sort(key=lambda c: (
        -(int(bool(c.context_terms)) if use_context else 0),
        -c.matched_words,
        -c.topic_overlap,
        index.sentences[c.sentence_index].normalized_sha256,
        c.continuation,
    ))
    seen_per_length: dict[int, int] = {}
    selected = []
    for candidate in candidates:
        n = candidate.matched_words
        if seen_per_length.get(n, 0) < per_length:
            selected.append(candidate)
            seen_per_length[n] = seen_per_length.get(n, 0) + 1
    return CandidateSelection(tuple(selected), raw_matches, rejected)
