"""Vocabulary-constrained text suggestions for a trained MICA model.

This is an optional user-interface decoder.  It does not change the R1
reference decoder in ``decode.py`` or the model's byte-level probabilities.
Every proposed word is followed by a space, punctuation, or EOS in the search,
so a returned suggestion never ends in an unfinished UTF-8 character or word.

The reported confidence is the best sequence's share of the probability mass
among *found* completions.  It is a search heuristic, not a calibrated estimate
of correctness, because the vocabulary and beam discard other continuations.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable, Protocol

import numpy as np


BOS = 256
EOS = 257
BOUNDARIES = (ord(" "), ord("."), ord(","), ord("!"), ord("?"), ord(";"), ord(":"), EOS)
WORD_RE = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)
PREFIX_RE = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*['\u2019-]?$", re.UNICODE)
DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "data/everyday/train.jsonl"


class ByteScorer(Protocol):
    """A branchable byte-level language model; scores are logits in nats."""

    def start(self, prompt: bytes) -> object: ...
    def scores(self, state: object) -> np.ndarray: ...
    def advance(self, state: object, byte: int) -> object: ...


@dataclass
class _Node:
    children: dict[int, _Node] = field(default_factory=dict)
    word: str | None = None


class Vocabulary:
    def __init__(self, words: Iterable[str]):
        self.root = _Node()
        self.size = 0
        for word in sorted(set(words)):
            if not word or WORD_RE.fullmatch(word) is None:
                continue
            raw = word.encode("utf-8")
            node = self.root
            for byte in raw:
                node = node.children.setdefault(byte, _Node())
            if node.word is None:
                node.word = word
                self.size += 1
        if not self.size:
            raise ValueError("vocabulary contains no valid words")

    def at_prefix(self, prefix: str) -> _Node | None:
        node = self.root
        for byte in prefix.encode("utf-8"):
            node = node.children.get(byte)
            if node is None:
                return None
        return node

    @classmethod
    def from_corpus(cls, path: str | Path = DEFAULT_CORPUS,
                    *, min_count: int = 2, max_words: int = 50_000,
                    max_word_bytes: int = 32) -> Vocabulary:
        """Count words in hex-encoded TRAIN records; never read validation/test.

        The small vocabulary cap keeps decoding practical. Invalid UTF-8 at a
        truncated record edge is ignored, rather than becoming a vocabulary
        item. Ties in frequency are settled lexically for repeatability.
        """
        if min_count < 1 or max_words < 1 or max_word_bytes < 1:
            raise ValueError("vocabulary limits must be positive")
        counts: Counter[str] = Counter()
        with Path(path).open("r", encoding="ascii") as stream:
            for line_number, line in enumerate(stream, 1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    text = bytes.fromhex(raw).decode("utf-8", errors="ignore")
                except ValueError as exc:
                    raise ValueError(f"invalid hex record at line {line_number}") from exc
                for match in WORD_RE.finditer(text):
                    word = match.group()
                    if len(word.encode("utf-8")) <= max_word_bytes:
                        counts[word] += 1
        words = [word for word, n in counts.items() if n >= min_count]
        words.sort(key=lambda word: (-counts[word], word))
        return cls(words[:max_words])


class MicaScorer:
    """Adapter to the exact integer engine; import after setting MICA_* geometry."""

    def __init__(self, model):
        from . import spec
        self.model = model
        self.divisor = spec.LOGIT_DIVISOR

    def start(self, prompt: bytes):
        from . import decode, engine
        if not decode.validate_prompt(prompt):
            raise ValueError("prompt is not valid UTF-8")
        session = engine.new_session(self.model)
        decode.feed_prompt(self.model, session, prompt)
        return session

    def scores(self, state) -> np.ndarray:
        from . import engine
        return engine.probe_scores(self.model, state).astype(np.float64) / self.divisor

    def advance(self, state, byte: int):
        from . import engine
        # Branches must not share a mutable field or phase array.
        clone = engine.Session(F=state.F.copy(), phase=state.phase.copy(),
                               position=state.position, ended=state.ended,
                               updates=state.updates, ticks_run=state.ticks_run)
        engine.ingest(self.model, clone, byte)
        return clone


@dataclass(frozen=True)
class Suggestion:
    continuation: str
    full_text: str
    log_probability: float
    confidence: float


@dataclass(frozen=True)
class SuggestionResult:
    suggestions: tuple[Suggestion, ...]
    abstained: bool
    reason: str
    confidence_note: str = "beam-relative, uncalibrated"


@dataclass
class _Partial:
    node: _Node
    state: object
    suffix: bytes
    logp: float


@dataclass
class _WordChoice:
    word: str
    suffix: bytes
    boundary: int
    state: object
    logp: float


@dataclass
class _Phrase:
    state: object
    emitted: bytes
    history: tuple[str, ...]
    logp: float
    words: int
    finished: bool
    partial_first: bool


def _log_probs(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (258,) or not np.all(np.isfinite(values)):
        raise ValueError("scorer must return 258 finite logits")
    values = values.copy()
    values[BOS] = -math.inf  # BOS is ingested at reset, never emitted.
    peak = float(values.max())
    return values - (peak + math.log(float(np.exp(values - peak).sum())))


def _repeats(history: tuple[str, ...], word: str, max_loop: int) -> bool:
    seq = history + (word.casefold(),)
    for period in range(1, max_loop + 1):
        if len(seq) >= 2 * period and seq[-period:] == seq[-2 * period:-period]:
            return True
    return False


def _prompt_context(prompt: str) -> tuple[str, tuple[str, ...], bytes]:
    match = PREFIX_RE.search(prompt)
    prefix = match.group() if match else ""
    earlier = prompt[:match.start()] if match else prompt
    history = tuple(m.group().casefold() for m in WORD_RE.finditer(earlier))
    if prefix or not prompt or prompt[-1].isspace() or prompt[-1] in "([{'\"\u201c\u2018":
        joiner = b""
    else:
        joiner = b" "
    return prefix, history, joiner


def _word_choices(scorer: ByteScorer, state: object, root: _Node,
                  history: tuple[str, ...], *, beam_width: int,
                  max_word_bytes: int, max_loop: int, limit: int,
                  continue_after_space: bool,
                  boundaries: tuple[int, ...] = BOUNDARIES) -> list[_WordChoice]:
    partials = [_Partial(root, state, b"", 0.0)]
    completed: list[_WordChoice] = []
    for depth in range(max_word_bytes + 1):
        next_edges: list[tuple[float, bytes, _Partial, int, _Node]] = []
        for partial in partials:
            probabilities = _log_probs(scorer.scores(partial.state))
            if partial.node.word is not None and not _repeats(
                    history, partial.node.word, max_loop):
                for boundary in boundaries:
                    # Only a space can lead to another word. Punctuation and
                    # EOS finish this phrase, so their branch need not ingest.
                    boundary_state = (scorer.advance(partial.state, boundary)
                                      if boundary == ord(" ") and continue_after_space
                                      else partial.state)
                    completed.append(_WordChoice(
                        partial.node.word, partial.suffix, boundary,
                        boundary_state, partial.logp + float(probabilities[boundary])))
            if depth == max_word_bytes:
                continue
            for byte, child in partial.node.children.items():
                suffix = partial.suffix + bytes((byte,))
                next_edges.append((partial.logp + float(probabilities[byte]),
                                   suffix, partial, byte, child))
        next_edges.sort(key=lambda edge: (-edge[0], edge[1]))
        partials = [_Partial(child, scorer.advance(parent.state, byte),
                             suffix, score)
                    for score, suffix, parent, byte, child in next_edges[:beam_width]]
        if not partials:
            break
    completed.sort(key=lambda c: (-c.logp, c.word, c.boundary))
    return completed[:limit]


def suggest(scorer: ByteScorer, vocabulary: Vocabulary, prompt: str,
            *, mode: str = "beam", beam_width: int = 2, max_words: int = 1,
            min_words: int = 1, require_terminal_punctuation: bool = False,
            max_word_bytes: int = 32, max_loop: int = 3,
            min_confidence: float = 0.55, top_k: int = 3,
            complete_final_word: bool = False) -> SuggestionResult:
    """Suggest a continuation using complete words from ``vocabulary``.

    ``greedy`` keeps one live byte/phrase path; ``beam`` keeps ``beam_width``.
    Search scores actual model byte probabilities, including separators. A
    repeated word or immediately repeated 2- or 3-word phrase is ineligible.
    The confidence is relative to found continuations and must not be treated
    as a calibrated probability or as evidence of factual correctness.
    Set ``complete_final_word`` when a prompt's final word is known to be
    complete (as in a next-word benchmark), so it cannot be extended.
    """
    if mode not in {"greedy", "beam"}:
        raise ValueError("mode must be 'greedy' or 'beam'")
    if beam_width < 1 or max_words < 1 or max_word_bytes < 1 or top_k < 1:
        raise ValueError("beam_width, max_words, max_word_bytes and top_k must be positive")
    if not 1 <= min_words <= max_words:
        raise ValueError("min_words must be between 1 and max_words")
    if max_loop < 1 or not 0 <= min_confidence <= 1:
        raise ValueError("max_loop must be positive and min_confidence in [0, 1]")
    width = 1 if mode == "greedy" else beam_width
    prefix, history, joiner = _prompt_context(prompt)
    root = (vocabulary.at_prefix(prefix)
            if prefix and not complete_final_word else vocabulary.root)
    if root is None:
        return SuggestionResult((), True, "prefix_not_in_vocabulary")
    initial = scorer.start(prompt.encode("utf-8"))
    if joiner:
        logp = float(_log_probs(scorer.scores(initial))[joiner[0]])
        initial = scorer.advance(initial, joiner[0])
    else:
        logp = 0.0
    active = []
    if not (prefix and complete_final_word):
        active.append(_Phrase(initial, joiner, history, logp, 0, False, True))
    if prefix and (complete_final_word or root.word is not None):
        # An exact vocabulary word at the end of a prompt may already be
        # complete. Search the following word as well as longer words sharing
        # the prefix. The space is part of the scored continuation.
        space = ord(" ")
        space_logp = float(_log_probs(scorer.scores(initial))[space])
        active.append(_Phrase(scorer.advance(initial, space), b" ",
                              history + (prefix.casefold(),),
                              logp + space_logp, 0, False, False))
    finished: list[_Phrase] = []
    for step in range(max_words):
        expanded: list[_Phrase] = []
        for phrase in active:
            word_root = root if phrase.partial_first else vocabulary.root
            if step + 1 < min_words:
                boundaries = (ord(" "),)
            elif require_terminal_punctuation:
                terminal = (ord("."), ord("!"), ord("?"))
                boundaries = ((ord(" "),) + terminal
                              if step + 1 < max_words else terminal)
            else:
                boundaries = BOUNDARIES
            choices = _word_choices(scorer, phrase.state, word_root,
                                    phrase.history, beam_width=width,
                                    max_word_bytes=max_word_bytes,
                                    max_loop=max_loop,
                                    limit=max(width * 8, top_k * 4),
                                    continue_after_space=step + 1 < max_words,
                                    boundaries=boundaries)
            for choice in choices:
                if phrase.partial_first and prefix and not choice.suffix:
                    # Adding only punctuation to the final prompt word is not
                    # a word suggestion. The full-word branch above handles
                    # possible continuations after this word.
                    continue
                emitted = phrase.emitted + choice.suffix
                if choice.boundary != EOS:
                    emitted += bytes((choice.boundary,))
                done = choice.boundary != ord(" ") or step + 1 == max_words
                candidate = _Phrase(choice.state, emitted,
                                    phrase.history + (choice.word.casefold(),),
                                    phrase.logp + choice.logp, step + 1, done,
                                    False)
                if done:
                    finished.append(candidate)
                else:
                    expanded.append(candidate)
        expanded.sort(key=lambda p: (-p.logp, p.emitted))
        active = expanded[:width]
        if not active:
            break
    by_text: dict[str, float] = {}
    for phrase in finished:
        try:
            continuation = phrase.emitted.decode("utf-8").rstrip(" ")
        except UnicodeDecodeError:
            continue
        if continuation:
            by_text[continuation] = max(by_text.get(continuation, -math.inf), phrase.logp)
    if not by_text:
        return SuggestionResult((), True, "no_complete_nonrepeating_word")
    ranked = sorted(by_text.items(), key=lambda item: (-item[1], item[0]))
    ranked = ranked[:max(top_k, width * 2)]
    peak = ranked[0][1]
    weights = [math.exp(score - peak) for _, score in ranked]
    total = sum(weights)
    suggestions = tuple(Suggestion(text, prompt + text, score, weight / total)
                        for (text, score), weight in zip(ranked[:top_k], weights[:top_k]))
    abstained = suggestions[0].confidence < min_confidence
    return SuggestionResult(suggestions, abstained,
                            "low_search_confidence" if abstained else "ok")


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="trained run directory with best.mica and run_info.json")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                        help="hex-encoded everyday TRAIN records")
    parser.add_argument("--words-file", type=Path,
                        help="alternative UTF-8 file with one allowed word per line")
    parser.add_argument("--mode", choices=("greedy", "beam"), default="beam")
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=1)
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--require-terminal-punctuation", action="store_true",
                        help="finish with a period, exclamation mark or question mark")
    parser.add_argument("--max-vocab", type=int, default=50_000)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--complete-final-word", action="store_true",
                        help="treat the prompt's final word as complete")
    args = parser.parse_args()

    info = json.loads((args.run / "run_info.json").read_text(encoding="utf-8"))
    os.environ.update({key: str(value) for key, value in info["env"].items()})
    # Import after loading the run's geometry: spec binds MICA_* at import.
    from . import serialize
    model = serialize.load(args.run / "best.mica")
    if args.words_file:
        vocabulary = Vocabulary(args.words_file.read_text(encoding="utf-8").splitlines())
    else:
        vocabulary = Vocabulary.from_corpus(args.corpus, max_words=args.max_vocab)
    result = suggest(MicaScorer(model), vocabulary, args.prompt,
                     mode=args.mode, beam_width=args.beam_width,
                     max_words=args.max_words, min_words=args.min_words,
                     require_terminal_punctuation=args.require_terminal_punctuation,
                     min_confidence=args.min_confidence,
                     complete_final_word=args.complete_final_word)
    print(json.dumps({"abstained": result.abstained, "reason": result.reason,
                      "confidence_note": result.confidence_note,
                      "vocabulary_size": vocabulary.size,
                      "suggestions": [s.__dict__ for s in result.suggestions]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
