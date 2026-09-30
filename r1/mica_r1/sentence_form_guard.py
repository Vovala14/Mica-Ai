"""Conservative, opt-in English sentence-form veto for retrieved suggestions.

This is deliberately a small rule set, not a parser or a grammaticality score.
It rejects clear nominal/participial fragments, dependent clauses with no
main clause, and direct questions punctuated as statements. Ambiguous forms
are left alone so a retrieval candidate is not discarded on weak evidence.
Only use it for completed sentence suggestions, never for next-word output.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import TypeVar


WORD = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
TERMINAL = re.compile(r"[.!?][\"'\u201d\u2019)]*$")
DIRECT_QUESTION = re.compile(
    r"^(?:"
    r"what(?:'s|\u2019s| is)\s+(?:my|your|his|her|our|their)\b|"
    r"(?:can|could|would|will|do|does|did|is|are|was|were|have|has|"
    r"should|may)\s+(?:i|you|he|she|it|we|they|the|a|an|this|that|"
    r"these|those|my|your|his|her|their|our)\b|"
    r"(?:what|where|when|why|how|which)\s+"
    r"(?:(?:time|many|much|long|far)\s+)?"
    r"(?:am|is|are|was|were|do|does|did|can|could|would|will|"
    r"have|has|should|may)\b"
    r")",
    re.IGNORECASE,
)

SUBORDINATORS = frozenset({
    "although", "because", "if", "once", "though", "unless", "until",
    "when", "whenever", "whereas", "while",
})
NOMINAL_START = frozenset({
    "a", "an", "the", "some", "any", "several", "many", "few", "both",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "this", "that", "these", "those", "my", "your",
    "his", "her", "its", "our", "their",
})
INITIAL_ADJECTIVES = frozenset({
    "local", "regional", "national", "global", "public", "private",
    "young", "old", "new", "large", "small", "red", "blue", "green",
    "quiet", "beautiful", "several",
})
PREPOSITIONS = frozenset({
    "about", "above", "across", "after", "against", "along", "among",
    "around", "at", "before", "behind", "below", "beneath", "beside",
    "between", "by", "during", "for", "from", "in", "inside", "into",
    "near", "of", "off", "on", "onto", "out", "over", "through", "to",
    "under", "until", "up", "upon", "with", "within", "without",
})
FINITE_AUXILIARIES = frozenset({
    "am", "is", "are", "was", "were", "have", "has", "had", "do",
    "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must",
})
FINITE_IRREGULARS = frozenset({
    "ate", "began", "bent", "bit", "blew", "bought", "broke", "brought",
    "built", "came", "caught", "chose", "drew", "drank", "drove", "fell",
    "felt", "fled", "flew", "forgot", "found", "gave", "got", "grew", "heard",
    "held", "kept", "knew", "laid", "led", "left", "lost", "made",
    "meant", "met", "paid", "ran", "read", "rode", "rose", "said",
    "sang", "sat", "saw", "sent", "slept", "sold", "spoke", "stood",
    "swam", "taught", "thought", "told", "took", "understood", "went",
    "won", "wore", "wrote",
})
# Common bare present-tense verbs. Other plausible verbs are left eligible by
# the conservative nominal-pattern check below; this list is not a dictionary.
BARE_VERBS = frozenset({
    "ask", "begin", "bring", "call", "come", "cut", "dance", "eat", "feel",
    "find", "fly", "get", "give", "go", "help", "hold", "know", "leave",
    "let", "like", "live", "look", "make", "move", "need", "open", "play",
    "put", "read", "run", "say", "see", "sit", "sleep", "speak", "stand",
    "start", "stay", "stop", "take", "talk", "think", "try", "turn",
    "use", "wait", "walk", "want", "watch", "work", "write",
})
NON_VERB_S = frozenset({"this", "his", "its", "ours", "theirs", "yours"})
INFINITIVE_OR_MODIFIER = frozenset({"to", "a", "an", "the", "some", "of"})
KNOWN_PLURAL = frozenset({"people", "children", "men", "women", "mice", "geese"})
NOMINAL_ADVERBS = frozenset({
    "here", "there", "today", "tomorrow", "yesterday", "outside", "inside",
    "nearby",
})
MEASURE_HEADS = frozenset({
    "bunch", "collection", "couple", "group", "lot", "pair", "pile", "set",
})


def _known_s_verb(word: str) -> bool:
    stems = {word[:-1]}
    if word.endswith("ies"):
        stems.add(word[:-3] + "y")
    if word.endswith("es"):
        stems.add(word[:-2])
    return any(stem in BARE_VERBS for stem in stems)


def _words(text: str) -> list[str]:
    return [match.group().casefold().replace("\u2019", "'")
            for match in WORD.finditer(text)]


def _finite_positions(words: list[str]) -> list[int]:
    """Find likely finite verbs; uncertainty means the candidate survives."""
    positions = []
    for i, word in enumerate(words):
        previous = words[i - 1] if i else ""
        if word.endswith(("'m", "'re", "'ve", "'ll", "'d")) or word.endswith("n't"):
            positions.append(i)
        elif word in FINITE_AUXILIARIES and previous != "to":
            positions.append(i)
        elif word in FINITE_IRREGULARS and previous != "to":
            positions.append(i)
        elif word in BARE_VERBS and previous not in INFINITIVE_OR_MODIFIER:
            positions.append(i)
        elif word.endswith("ed") and len(word) > 4 and previous != "to":
            # A bare -ed form can be a past-tense verb or a participial
            # modifier. Accept the ambiguity instead of vetoing it.
            positions.append(i)
        elif (word.endswith("s") and len(word) > 3 and
              word not in NON_VERB_S and
              (i >= 2 or (i == 1 and previous in {"this", "that"})) and
              (previous in {"this", "that"} or previous not in
               NOMINAL_START | INITIAL_ADJECTIVES | PREPOSITIONS or
               previous in PREPOSITIONS and
               any(part.endswith("ing") for part in words[:i - 1]))):
            positions.append(i)
        elif (i >= 2 and words[i - 2] in NOMINAL_START and
              (previous in KNOWN_PLURAL or
               previous.endswith("s") and not previous.endswith("ss")) and
              not any(part in PREPOSITIONS for part in words[:i - 1]) and
              word not in PREPOSITIONS | INITIAL_ADJECTIVES | NOMINAL_ADVERBS and
              not word.endswith("ing")):
            # Preserve possible unfamiliar bare verbs after plural subjects,
            # e.g. "The dogs frolic in the park."
            positions.append(i)
    return positions


def _main_clause_after_separator(text: str) -> bool:
    parts = re.split(r"[,;:]", text, maxsplit=1)
    if len(parts) == 1:
        return False
    tail = _words(parts[1])
    if not tail:
        return False
    if _finite_positions(tail):
        return True
    # An imperative can supply the main clause: "If it rains, go inside."
    return tail[0] in BARE_VERBS or (len(tail) > 1 and tail[0] == "please"
                                     and tail[1] in BARE_VERBS)


def _main_clause_without_separator(words: list[str]) -> bool:
    finite = _finite_positions(words)
    if not finite:
        return False
    subject_starts = NOMINAL_START | frozenset({
        "i", "you", "he", "she", "it", "we", "they", "there",
    })
    for first_finite in finite:
        for i in range(first_finite + 1, len(words) - 1):
            if words[i] in subject_starts and _finite_positions(words[i + 1:]):
                return True
    return False


def _clear_nominal_fragment(words: list[str]) -> bool:
    if not words:
        return False
    if words[0] in {"these", "those"}:
        return False  # The next unknown word may be a bare plural verb.
    if words[0] not in NOMINAL_START and words[0] not in INITIAL_ADJECTIVES:
        return False
    finite = _finite_positions(words)
    # In a measure phrase such as "A bunch of water balloons", an unknown
    # plural-looking final word is more likely the noun ending the "of" phrase
    # than a verb. Known verb inflections still survive ("A group of boys
    # plays.").
    if (finite == [len(words) - 1] and words[-1].endswith("s") and
            not _known_s_verb(words[-1]) and "of" in words[2:] and
            any(head in MEASURE_HEADS for head in words[1:3])):
        finite = []
    # "The book that I read" has a finite relative clause but still lacks a
    # predicate for "book". An additional finite verb can be the main clause.
    relative = next((i for i, word in enumerate(words[2:], 2)
                     if word in {"that", "who", "which", "whose", "where", "why"}),
                    None)
    if relative is not None and finite and all(i > relative for i in finite):
        if len(finite) == 1:
            return True
    if finite:
        return False
    # Clear nonfinite modifier: "A dog jumping over a log."
    if any(word.endswith("ing") and len(word) > 5 for word in words[1:]):
        return True
    # A nominal head followed only by prepositional material is a reliable
    # fragment pattern: "A bunch of people", "Local clubs for years".
    if any(word in PREPOSITIONS for word in words[2:]):
        return True
    # Short, familiar determiner + adjective + noun or determiner + noun.
    if words[0] in NOMINAL_START and (
        len(words) == 2 or
        len(words) == 3 and words[1] in INITIAL_ADJECTIVES
    ):
        return True
    return False


def sentence_form_issue(full_text: str) -> str | None:
    """Return a veto reason, or ``None`` when no clear form error is found.

    ``None`` does not certify grammaticality or topical fit. This guard is
    designed for high-precision rejection of a few common retrieval artifacts.
    """
    text = full_text.strip()
    terminal = TERMINAL.search(text)
    if not text or not terminal:
        return "missing_terminal_punctuation"
    words = _words(text)
    if not words:
        return "missing_main_clause"
    direct_question = bool(DIRECT_QUESTION.match(text.lstrip(" \"'\u201c\u2018(")))
    if terminal.group().startswith(".") and direct_question:
        return "question_terminal_punctuation"
    if words[0] in SUBORDINATORS:
        if words[0] == "when" and direct_question:
            return None
        if not (_main_clause_after_separator(text) or
                _main_clause_without_separator(words)):
            return "missing_main_clause"
    if _clear_nominal_fragment(words):
        return "nominal_fragment"
    return None


_Result = TypeVar("_Result")


def veto_sentence_fragments(result: _Result, *, mode: str = "sentence") -> _Result:
    """Filter a retrieval result's ranked suggestions without changing its API.

    Call after retrieval with an expanded ``top_k`` if replacement candidates
    are wanted. A wholly vetoed result becomes an abstention. Short-mode
    continuations are deliberately untouched.
    """
    if mode != "sentence" or result.abstained:
        return result
    retained = tuple(item for item in result.suggestions
                     if sentence_form_issue(item.full_text) is None)
    if len(retained) == len(result.suggestions):
        return result
    return replace(result, suggestions=retained, abstained=not bool(retained),
                   reason="ok" if retained else "sentence_form_veto")
