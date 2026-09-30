"""Greedy reference decoder and UTF-8 validator. Sections 8 and 9.

"No temperature, top-k, repetition penalty, sampling seed, or exponential
 lookup is part of the reference decoder."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spec
from .engine import Model, Session, ingest, probe_scores, new_session
from .spec import BOS, EOS

EOS_STOP, LIMIT_STOP, INCOMPLETE_LIMIT = "EOS_STOP", "LIMIT_STOP", "INCOMPLETE_LIMIT"
INVALID_UTF8, NO_VALID_OUTPUT = "INVALID_UTF8", "NO_VALID_OUTPUT"


@dataclass
class Utf8State:
    """Remaining continuation count and the allowed range for the next byte."""
    remaining: int = 0
    lo: int = 0x00
    hi: int = 0x7F

    def at_boundary(self) -> bool:
        return self.remaining == 0


def lead_spec(b: int):
    """Section 9 table: (next range, continuations) for an accepted lead byte."""
    if 0xC2 <= b <= 0xDF:
        return (0x80, 0xBF), 1
    if b == 0xE0:
        return (0xA0, 0xBF), 2
    if b == 0xED:
        return (0x80, 0x9F), 2
    if 0xE1 <= b <= 0xEC or 0xEE <= b <= 0xEF:
        return (0x80, 0xBF), 2
    if b == 0xF0:
        return (0x90, 0xBF), 3
    if 0xF1 <= b <= 0xF3:
        return (0x80, 0xBF), 3
    if b == 0xF4:
        return (0x80, 0x8F), 3
    return None


def eligible_mask(st: Utf8State, text_mode: bool) -> np.ndarray:
    """Which of the 258 symbols may be chosen. BOS is always ineligible."""
    mask = np.zeros(spec.N_SYMBOLS, bool)
    if not text_mode:
        mask[0:spec.BOS] = True
        mask[EOS] = True
        return mask
    if st.at_boundary():
        mask[0x00:0x80] = True
        for b in range(0xC2, 0xF5):
            if lead_spec(b):
                mask[b] = True
        mask[EOS] = True                      # only at a character boundary
    else:
        mask[st.lo:st.hi + 1] = True
    return mask


def advance(st: Utf8State, byte: int) -> Utf8State:
    if st.at_boundary():
        if byte < 0x80:
            return Utf8State(0, 0x00, 0x7F)
        ls = lead_spec(byte)
        if ls is None:
            raise ValueError(f"invalid lead byte {byte:#04x}")
        (lo, hi), n = ls
        return Utf8State(n, lo, hi)
    rem = st.remaining - 1
    return Utf8State(0, 0x00, 0x7F) if rem == 0 else Utf8State(rem, 0x80, 0xBF)


def validate_prompt(data: bytes) -> bool:
    """Section 9: validate the whole prompt before mutating a session."""
    st = Utf8State()
    for b in data:
        if st.at_boundary():
            if b >= 0x80 and lead_spec(b) is None:
                return False
        elif not (st.lo <= b <= st.hi):
            return False
        try:
            st = advance(st, b)
        except ValueError:
            return False
    return st.at_boundary()


def feed_prompt(model: Model, s: Session, data: bytes, text_mode: bool = True):
    """Section 2: ingest the complete prompt silently; emit only in generation."""
    if text_mode and not validate_prompt(data):
        return INVALID_UTF8
    for b in data:
        ingest(model, s, b)
    return None


def generate(model: Model, s: Session, byte_limit: int, text_mode: bool = True):
    """Section 8 greedy reference decoder. Returns (bytes, status)."""
    out = bytearray()
    st = Utf8State()
    pending = bytearray()
    while len(out) + len(pending) < byte_limit:
        scores = probe_scores(model, s)
        mask = eligible_mask(st, text_mode)
        if not mask.any():
            return bytes(out), NO_VALID_OUTPUT
        masked = np.where(mask, scores, np.iinfo(np.int64).min)
        symbol = int(masked.argmax())          # ties -> smaller symbol id
        if symbol == EOS:
            s.ended = True
            return bytes(out), EOS_STOP
        if text_mode:
            pending.append(symbol)
            st = advance(st, symbol)
            if st.at_boundary():
                out.extend(pending)
                pending.clear()
        else:
            out.append(symbol)
        ingest(model, s, symbol)
    if text_mode and pending:
        # "discard the pending display buffer, return INCOMPLETE_LIMIT"
        return bytes(out), INCOMPLETE_LIMIT
    return bytes(out), LIMIT_STOP


def logits(model: Model, s: Session) -> np.ndarray:
    """Section 12: "divide each integer probe score by 16 to obtain a logit"."""
    return probe_scores(model, s).astype(np.float64) / spec.LOGIT_DIVISOR
