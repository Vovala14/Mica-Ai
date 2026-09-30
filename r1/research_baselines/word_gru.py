"""Archived GRU comparison experiment; not part of MICA.

The user's project trains only learned integer-rule MICA models. This file
retains the abandoned comparison code and is never used for MICA training or
its release inference path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch
from torch import nn
import torch.nn.functional as F

from mica_r1.suggest import ByteScorer, _log_probs
from .word_ngram import PUNCTUATION, TERMINAL, _piece, _tokens


class WordGRU(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        # Windows ROCm's MIOpen build currently lacks a dropout header; the
        # fused multi-layer GRU works with dropout disabled.
        self.recurrent = nn.GRU(dim, dim, layers, batch_first=True,
                                dropout=0.0)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, state=None):
        hidden, state = self.recurrent(self.embedding(tokens), state)
        return self.output(self.norm(hidden)) + self.bias, state

    @torch.no_grad()
    def next_log_probs(self, token: int, state=None):
        logits, state = self.forward(torch.tensor([[token]],
                                                  device=self.bias.device), state)
        return F.log_softmax(logits[0, -1], dim=-1), state


@dataclass(frozen=True)
class WordGRUSample:
    text: str
    continuation: str
    complete: bool
    generated_words: int
    seed: int


@torch.no_grad()
def generate(model: WordGRU, vocab: list[str], scorer: ByteScorer,
             prompt: str = "", *, seed: int = 0, min_words: int = 4,
             max_words: int = 16, options: int = 32, temperature: float = 0.8,
             mica_weight: float = 0.35) -> WordGRUSample:
    """Sample word candidates from the local GRU and score each with MICA."""
    if min_words < 1 or max_words < min_words or options < 1:
        raise ValueError("invalid length or candidate limit")
    if temperature <= 0 or mica_weight < 0:
        raise ValueError("temperature and MICA weight must be nonnegative")
    model.eval()
    word_id = {token: i for i, token in enumerate(vocab)}
    prefix = [0] + [word_id.get(token, 2) for token in _tokens(prompt)]
    logits, neural_state = model(torch.tensor([prefix], device=model.bias.device))
    log_probs = F.log_softmax(logits[0, -1], dim=-1)
    mica_state = scorer.start(prompt.encode("utf-8"))
    rng = random.Random(seed)
    text = prompt
    generated_words = 0
    complete = False
    previous = _tokens(prompt)[-1] if _tokens(prompt) else ""
    for _ in range(max_words * 2 + 4):
        candidates = []
        top = torch.topk(log_probs, min(options * 2, len(vocab)))
        for index, neural_logp in zip(top.indices.tolist(), top.values.tolist()):
            token = vocab[index]
            if index < 3 or token in {"<unk>", "<eos>", "<bos>"}:
                continue
            if token in TERMINAL and generated_words < min_words:
                continue
            if token in {",", ";", ":"} and (generated_words < 2 or
                    previous in PUNCTUATION):
                continue
            if token not in PUNCTUATION and generated_words >= max_words:
                continue
            form = token.capitalize() if not text.strip() else token
            piece = _piece(text, form)
            branch = mica_state
            mica_mean = 0.0
            if mica_weight:
                total = 0.0
                for byte in piece.encode("utf-8"):
                    total += float(_log_probs(scorer.scores(branch))[byte])
                    branch = scorer.advance(branch, byte)
                mica_mean = total / len(piece.encode("utf-8"))
            logit = neural_logp + mica_weight * mica_mean
            if token in TERMINAL and generated_words >= min_words:
                logit += 0.2 * (generated_words - min_words)
            candidates.append((index, token, piece, branch, logit))
            if len(candidates) >= options:
                break
        if not candidates:
            break
        peak = max(row[4] for row in candidates)
        weights = [math.exp((row[4] - peak) / temperature) for row in candidates]
        index, token, piece, mica_state, _ = rng.choices(candidates, weights=weights)[0]
        text += piece
        previous = token
        if token not in PUNCTUATION:
            generated_words += 1
        if token in TERMINAL:
            complete = True
            break
        log_probs, neural_state = model.next_log_probs(index, neural_state)
    if not complete:
        text = text.rstrip(" ,;:") + "."
    return WordGRUSample(text, text[len(prompt):], complete, generated_words, seed)
