"""Experimental topical signal for train-corpus sentence completions.

The retrieval index already stores word -> sentence postings. This helper
uses those postings to measure whether a distinctive prompt noun and the
beginning of a proposed continuation occur together in other train sentences.
The signal is advisory: zero can mean either unrelated text or sparse data.
It is never a language-model score and should not replace MICA ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval_suggest import SentenceIndex


WORD = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
DETERMINERS = frozenset("a an the my your his her our their this that".split())
COMMON_MODIFIERS = frozenset("new old big small little large".split())
FUNCTION_WORDS = frozenset("""
    a an the and or but of on in to for from with at by about as that this
    these those is are was were be been being do does did have has had will
    would can could should may might must i you he she it we they me him her
    them my your his its our their who what when where why how if then so
    not no yes please very just all any some one two into over under again
    there here than also more most before after anywhere like
""".split())


@dataclass(frozen=True)
class TopicSignal:
    value: float
    anchor: str | None
    candidate_word: str | None
    joint_sentences: int


def _tokens(text: str) -> list[re.Match[str]]:
    return list(WORD.finditer(text))


def topic_anchors(prompt: str, matched_words: int) -> tuple[str, ...]:
    """Find the nearest completed noun phrase before the matched tail.

    This limited syntactic cue avoids generic tail verbs overwhelming the
    entity the user is actually talking about. A one-word fallback helps
    prompts without a determiner, such as "It was raining so hard that".
    """
    words = _tokens(prompt)
    if matched_words < 0 or matched_words > len(words):
        raise ValueError("invalid matched_words")
    context = words[:-matched_words] if matched_words else words
    for position in range(len(context) - 2, -1, -1):
        if context[position].group().casefold() not in DETERMINERS:
            continue
        following = []
        for word in context[position + 1:position + 3]:
            between = prompt[context[position].end():word.start()]
            if re.search(r"[,.!?;:]", between):
                break
            token = word.group().casefold()
            if token in FUNCTION_WORDS:
                break
            following.append(token)
        if following:
            head = (following[1] if len(following) > 1 and
                    following[0] in COMMON_MODIFIERS else following[0])
            return (head,)
    for word in reversed(context):
        token = word.group().casefold()
        if token not in FUNCTION_WORDS:
            return (token,)
    return ()


class TopicConsistency:
    """Read-only positive-PMI signal from a loaded train sentence index."""

    def __init__(self, index: SentenceIndex):
        self.index = index
        self.total_sentences = len(index.sentences)

    @lru_cache(maxsize=256)
    def _sentences_with(self, word: str) -> frozenset[int]:
        return frozenset(sentence for sentence, _ in
                         self.index.postings.get(word, ()))

    def _pair(self, anchor: str, candidate: str) -> TopicSignal:
        left = self._sentences_with(anchor)
        right = self._sentences_with(candidate)
        if not left or not right:
            return TopicSignal(0.0, anchor, candidate, 0)
        joint = len(left & right)
        if not joint:
            return TopicSignal(0.0, anchor, candidate, 0)
        lift = ((joint + 0.5) * self.total_sentences /
                ((len(left) + 0.5) * (len(right) + 0.5)))
        value = max(0.0, math.log(lift)) * math.log1p(joint)
        return TopicSignal(value, anchor, candidate, joint)

    def score(self, prompt: str, continuation: str,
              matched_words: int, *, max_continuation_words: int = 2) -> TopicSignal:
        """Return the strongest prompt-topic to early-continuation link."""
        anchors = topic_anchors(prompt, matched_words)
        words = [m.group().casefold() for m in _tokens(continuation)
                 if m.group().casefold() not in FUNCTION_WORDS]
        if not anchors or not words:
            return TopicSignal(0.0, None, None, 0)
        signals = [self._pair(anchor, word) for anchor in anchors
                   for word in words[:max_continuation_words]]
        return max(signals, key=lambda signal: signal.value)
