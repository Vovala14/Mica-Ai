"""Opt-in experimental retrieval reranker with splice compatibility checks.

The reference decoder in ``retrieval_suggest.py`` remains unchanged. This
variant keeps several trailing-match lengths, checks an obvious noun/verb
splice ambiguity, and correctly includes the word immediately before a source
match in its topic-overlap calculation. Scores remain heuristics, not calibrated
probabilities of a good continuation.
"""

from __future__ import annotations

from collections import Counter
import math

from .retrieval_compat import select_compatible_candidates
from .retrieval_suggest import (
    STOPWORDS,
    WORD,
    RetrievalResult,
    RetrievalSuggestion,
    SentenceIndex,
    _align_terminal,
    _mean_logp,
    _sentence_clause,
    _shorten,
    _topic_key,
)
from .suggest import ByteScorer


def _fixed_topic_overlap(index: SentenceIndex, prompt: str, source: str,
                         continuation: str, matched_words: int) -> float:
    """Compare unmatched prefix words on both sides of the splice.

    A source match of ``matched_words`` ends immediately before
    ``continuation``. Consequently the source context is all prefix words
    except the last ``matched_words``; the frozen decoder's position slice
    omitted one additional word.
    """
    if not continuation or not source.endswith(continuation):
        return 0.0
    before_continuation = source[:-len(continuation)]
    source_words = [m.group().casefold() for m in WORD.finditer(before_continuation)]
    prompt_words = [m.group().casefold() for m in WORD.finditer(prompt)]
    if min(len(source_words), len(prompt_words)) < matched_words:
        return 0.0
    prompt_context = set(prompt_words[:-matched_words]) - STOPWORDS
    source_context = {_topic_key(word) for word in source_words[:-matched_words]}
    return sum(
        min(6.0, math.log((len(index.sentences) + 1) /
                          (len(index.postings.get(word, ())) + 1)))
        for word in prompt_context if _topic_key(word) in source_context
    )


def suggest_retrieval_v2(scorer: ByteScorer, index: SentenceIndex,
                         prompt: str, *, mode: str = "sentence",
                         max_words: int = 8, min_match_words: int = 2,
                         max_match_words: int = 8, shortlist: int = 48,
                         top_k: int = 3, model_scored_bytes: int = 8,
                         max_length_gap: int = 3) -> RetrievalResult:
    """Suggest source excerpts with a guarded, multi-length candidate pool."""
    if mode not in {"short", "sentence"}:
        raise ValueError("mode must be 'short' or 'sentence'")
    if min_match_words < 2 or max_match_words < min_match_words:
        raise ValueError("require at least a two-word trailing match")
    if min(max_words, shortlist, top_k, model_scored_bytes) < 1:
        raise ValueError("limits must be positive")
    if not prompt or not prompt.strip():
        return RetrievalResult((), True, "prompt_too_short",
                               method="experimental splice-compatible retrieval v2")
    selection = select_compatible_candidates(
        index, prompt, min_match_words=min_match_words,
        max_match_words=max_match_words,
        max_length_gap=max_length_gap, per_length=shortlist,
    )
    if not selection.candidates:
        reason = ("no_two_word_train_match" if not selection.raw_matches
                  else "no_role_compatible_continuation")
        return RetrievalResult((), True, reason,
                               method="experimental splice-compatible retrieval v2")
    candidates = selection.candidates[:shortlist]
    support = Counter(WORD.search(c.continuation).group().casefold()
                      for c in selection.candidates)
    initial = scorer.start(prompt.encode("utf-8"))
    ranked = []
    prompt_words = list(WORD.finditer(prompt))
    for candidate in candidates:
        source = index.sentences[candidate.sentence_index]
        matched_words = candidate.matched_words
        next_word = WORD.search(candidate.continuation).group().casefold()
        count = support[next_word]
        if mode == "sentence":
            continuation, constructed = _sentence_clause(candidate.continuation)
            if len(list(WORD.finditer(continuation))) > 16:
                continue
        else:
            clause, constructed = _sentence_clause(candidate.continuation)
            continuation = _shorten(clause, max_words)
            if continuation != clause:
                constructed = False
        continuation, constructed = _align_terminal(prompt, continuation,
                                                     constructed)
        if not continuation:
            continue
        model_mean, nbytes = _mean_logp(scorer, initial, continuation,
                                        model_scored_bytes)
        prefix_content = {w.group().casefold()
                          for w in prompt_words[:-matched_words]} - STOPWORDS
        tail_content = {w.group().casefold()
                        for w in prompt_words[-matched_words:]} - STOPWORDS
        suffix_words = len(list(WORD.finditer(continuation)))
        topic_overlap = _fixed_topic_overlap(
            index, prompt, source.text, candidate.continuation, matched_words)
        if prefix_content and not tail_content and topic_overlap == 0:
            continue
        if (matched_words == 2 and len(prefix_content) >= 2 and
                topic_overlap == 0 and
                not (count >= 2 and suffix_words <= 5) and
                not (model_mean > -0.65 and suffix_words <= 5)):
            continue
        if (matched_words == 3 and len(prefix_content) >= 3 and
                topic_overlap == 0 and suffix_words > 8):
            continue
        semantic = index.topic.score(prompt, continuation, matched_words)
        score = (model_mean + 0.60 * math.log1p(count) +
                 0.25 * min(topic_overlap, 8.0) -
                 0.025 * suffix_words +
                 0.20 * min(semantic.value, 1.5) +
                 0.12 * matched_words +
                 (0.60 if candidate.context_terms else 0.0))
        ranked.append(RetrievalSuggestion(
            continuation=continuation, full_text=prompt + continuation,
            source_sentence=source.text, source_train_line=source.train_line,
            source_sentence_ordinal=source.sentence_ordinal,
            source_corpus=index.manifest.get("source_train_file"),
            source_url=source.source_url, source_title=source.source_title,
            source_license=source.source_license,
            matched_words=matched_words, collocation_support=count,
            topic_overlap=topic_overlap, topic_signal=semantic.value,
            topic_anchor=semantic.anchor,
            topic_candidate_word=semantic.candidate_word,
            topic_joint_sentences=semantic.joint_sentences,
            constructed_punctuation=constructed,
            model_mean_logp=model_mean, model_scored_bytes=nbytes,
            rank_score=score,
        ))
    ranked.sort(key=lambda s: (-s.rank_score, -s.matched_words,
                               s.source_train_line, s.continuation))
    unique: dict[str, RetrievalSuggestion] = {}
    for item in ranked:
        unique.setdefault(item.continuation, item)
    result = tuple(list(unique.values())[:top_k])
    return RetrievalResult(
        result, not bool(result), "ok" if result else "no_usable_continuation",
        method="experimental splice-compatible retrieval v2",
    )
