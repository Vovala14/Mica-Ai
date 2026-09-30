"""Archived word-count generator used only for comparison.

This learns word transition counts, then constructs a new sentence one token
at a time. It was a diagnostic sidecar and is not part of MICA's learned
integer rules, reference decoder, or release model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Iterable

from mica_r1.suggest import ByteScorer, _log_probs


TOKEN = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*|[.!?,;:]")
START = "<s>"
TERMINAL = frozenset(".!?")
PUNCTUATION = frozenset(".!?,;:")


def _normal(text: str) -> str:
    return " ".join(text.casefold().split())


def _tokens(text: str) -> list[str]:
    return [match.group().casefold() for match in TOKEN.finditer(text)]


def _piece(text: str, token: str) -> str:
    if token in PUNCTUATION:
        return token
    if not text or text[-1].isspace() or text[-1] in "([{'\"":
        return token
    return " " + token


@dataclass(frozen=True)
class GeneratedSentence:
    text: str
    continuation: str
    complete: bool
    exact_index_match: bool
    generated_words: int
    seed: int
    method: str = "learned word transitions + MICA byte scoring"


class WordNgram:
    """Three-gram next-word distribution, backed off to two/one-gram counts."""

    def __init__(self) -> None:
        self.trigram: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.bigram: dict[str, Counter[str]] = defaultdict(Counter)
        self.unigram: Counter[str] = Counter()
        self.forms: dict[str, Counter[str]] = defaultdict(Counter)
        self.source_hashes: set[str] = set()
        self.sentences = 0

    def add(self, sentence: str) -> bool:
        matches = list(TOKEN.finditer(sentence))
        if not matches or matches[-1].group() not in TERMINAL:
            return False
        words = [match.group() for match in matches if match.group() not in PUNCTUATION]
        if not 5 <= len(words) <= 35:
            return False
        items = [match.group().casefold() for match in matches]
        context = [START, START]
        for original, token in zip(matches, items):
            self.trigram[context[-2], context[-1]][token] += 1
            self.bigram[context[-1]][token] += 1
            self.unigram[token] += 1
            if token not in PUNCTUATION:
                self.forms[token][original.group()] += 1
            context.append(token)
        self.source_hashes.add(hashlib.sha256(_normal(sentence).encode()).hexdigest())
        self.sentences += 1
        return True

    @classmethod
    def from_index(cls, index_file: str | Path, *, max_sentences: int | None = None) -> WordNgram:
        model = cls()
        with Path(index_file).open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                model.add(record["text"])
                if max_sentences is not None and model.sentences >= max_sentences:
                    break
        if not model.sentences:
            raise ValueError("no complete training sentences were found")
        return model

    def next_probabilities(self, history: tuple[str, str], *, limit: int = 32) -> list[tuple[str, float]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        tri = self.trigram.get(history, Counter())
        bi = self.bigram.get(history[-1], Counter())
        if tri:
            weights = (0.76, 0.20, 0.04)
        elif bi:
            weights = (0.0, 0.88, 0.12)
        else:
            weights = (0.0, 0.0, 1.0)
        possible = {word for word, _ in tri.most_common(limit)}
        possible.update(word for word, _ in bi.most_common(min(limit, 16)))
        possible.update(word for word, _ in self.unigram.most_common(8))
        t_total = sum(tri.values()) or 1
        b_total = sum(bi.values()) or 1
        u_total = sum(self.unigram.values()) or 1
        ranked = [
            (word, weights[0] * tri[word] / t_total +
             weights[1] * bi[word] / b_total +
             weights[2] * self.unigram[word] / u_total)
            for word in possible
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:limit]

    def _surface(self, token: str, text: str) -> str:
        if token in PUNCTUATION:
            return token
        variants = self.forms.get(token)
        form = variants.most_common(1)[0][0] if variants else token
        if not text.strip() or text.rstrip().endswith(tuple(TERMINAL)):
            return form[:1].upper() + form[1:]
        return form

    def generate(self, scorer: ByteScorer, prompt: str = "", *, seed: int = 0,
                 min_words: int = 5, max_words: int = 18,
                 options: int = 20, temperature: float = 0.8,
                 mica_weight: float = 0.35) -> GeneratedSentence:
        if not 0 < min_words <= max_words:
            raise ValueError("require 0 < min_words <= max_words")
        if temperature <= 0 or options < 1 or mica_weight < 0:
            raise ValueError("invalid generation settings")
        if not self.sentences:
            raise ValueError("word model is empty")
        rng = random.Random(seed)
        text = prompt
        original_prompt = prompt
        prompt_tokens = _tokens(prompt)
        words = [token for token in prompt_tokens if token not in PUNCTUATION]
        history = tuple((([START, START] + prompt_tokens)[-2:]))
        state = scorer.start(prompt.encode("utf-8"))
        generated_words = 0
        complete = False
        for _ in range(max_words * 2 + 4):
            candidates = []
            for token, probability in self.next_probabilities(history, limit=options):
                if token in TERMINAL and generated_words < min_words:
                    continue
                if token in {",", ";", ":"} and (generated_words < 2 or
                        (history[-1] in PUNCTUATION)):
                    continue
                if token not in PUNCTUATION and generated_words >= max_words:
                    continue
                form = self._surface(token, text)
                piece = _piece(text, form)
                branch = state
                total = 0.0
                for byte in piece.encode("utf-8"):
                    total += float(_log_probs(scorer.scores(branch))[byte])
                    branch = scorer.advance(branch, byte)
                mica_mean = total / len(piece.encode("utf-8"))
                logit = math.log(max(probability, 1e-12)) + mica_weight * mica_mean
                if token in TERMINAL and generated_words >= min_words:
                    logit += 0.25 * (generated_words - min_words)
                candidates.append((token, piece, branch, logit))
            if not candidates:
                break
            peak = max(candidate[3] for candidate in candidates)
            weights = [math.exp((candidate[3] - peak) / temperature)
                       for candidate in candidates]
            chosen = rng.choices(candidates, weights=weights, k=1)[0]
            token, piece, state, _ = chosen
            text += piece
            history = (history[-1], token)
            if token not in PUNCTUATION:
                generated_words += 1
            if token in TERMINAL:
                complete = True
                break
        if not complete:
            text = text.rstrip(" ,;:") + "."
        source_hash = hashlib.sha256(_normal(text).encode()).hexdigest()
        return GeneratedSentence(text, text[len(original_prompt):], complete,
                                 source_hash in self.source_hashes,
                                 generated_words, seed)
