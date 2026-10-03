"""Flame-W memory readout (Claude, 2026-10-03).

The cellular automaton's rules reach only a few words back. This adds an integer readout
over the last 64 tokens, on top of the automaton's own integer scores:

  recall       every earlier occurrence of word w (lag 1..64) adds R[class(w)][lag bucket]
  association  M = sum over the window of lam[lag bucket] * A[u]   (16 int8 channels)
               every word w gets (Bm[w] . M) * MUL[w] >> 30

All quantities are integers (int8 tables, int64 accumulation); the bonus is in the engine's
1/1024 logit units and is added before the softmax. BOS, EOS and <sep> are never read.
The tables are in memory.npz (no pickle).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

WINDOW = 64
LAG_BUCKET = np.array([0, 0, 1, 2, 3, 3, 3] + [4] * 6 + [5] * 12 + [6] * 41, np.int64)  # index = lag
SHIFT = 30


class Memory:
    def __init__(self, path: str | Path):
        d = np.load(Path(path), allow_pickle=False)
        self.R = d["R_int"].astype(np.int64)            # (5 classes, 7 lag buckets)
        self.cls = d["class_of_id"].astype(np.int64)     # (16384,)
        self.skip = set(int(x) for x in d["never_read"])
        self.assoc = "A_int" in d.files
        if self.assoc:
            self.A = d["A_int"].astype(np.int64)         # (16384, k) int8 values
            self.lam = d["lam_int"].astype(np.int64)     # (7,)
            self.Bm = d["Bm_int"].astype(np.int64)       # (16384, k) int8 values
            self.MUL = d["MUL"].astype(np.int64)         # (16384,)

    def bonus(self, hist: list[int]) -> np.ndarray:
        """Integer bonus for every id, given the token history (oldest first)."""
        out = np.zeros(len(self.cls), np.int64)
        win = hist[-WINDOW:][::-1]
        ids = np.array([t for t in win if t not in self.skip], np.int64)
        if not len(ids):
            return out
        lags = np.array([j + 1 for j, t in enumerate(win) if t not in self.skip], np.int64)
        lb = LAG_BUCKET[lags]
        np.add.at(out, ids, self.R[self.cls[ids], lb])
        if self.assoc:
            M = (self.lam[lb][:, None] * self.A[ids]).sum(0)
            out += ((self.Bm @ M) * self.MUL) >> SHIFT
        return out


class MemState:
    def __init__(self, session, hist):
        self.s, self.hist = session, hist


class MemoryMicaWord:
    """word_eval.MicaWord plus the memory readout; same start/feed/fork/scores/logp API."""

    def __init__(self, base, memory: Memory):
        self.base, self.memory = base, memory
        self.div, self.elig = base.div, base.elig
        self.info = dict(base.info, memory=True)

    def start(self):
        return MemState(self.base.start(), [])

    def feed(self, st, i: int):
        st.s = self.base.feed(st.s, i)
        st.hist.append(int(i))
        return st

    def fork(self, st):
        return MemState(self.base.fork(st.s), list(st.hist))

    def scores(self, st) -> np.ndarray:
        return self.base.scores(st.s).astype(np.int64) + self.memory.bonus(st.hist)

    def logp(self, st, ids) -> np.ndarray:
        z = self.scores(st).astype(np.float64) / self.div
        z = np.where(self.elig, z, -np.inf)
        z -= z.max()
        return z[np.asarray(ids)] - np.log(np.exp(z[self.elig]).sum())
