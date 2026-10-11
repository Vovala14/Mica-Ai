"""Flame-W 0.3.6 context banks (Claude, 2026-10-12): two integer readouts of the recent words, added to 0.3.5's scores.

  PrevWordBank  the previous fed word p (not <sep>/BOS/EOS) gives every word y the bonus
                ((Bm[y] . A[p]) * MUL[y]) >> 30 (16 int8 values per word), and p itself R[class(p)]  (bank.npz)
  PhraseCopy    if the last fed token q1 (not <sep>/BOS/EOS) occurred earlier among the last 256 fed tokens, at lag L,
                the token f that followed it there (not <sep>/BOS/EOS) gets I1[class(q1)][class(f)][bucket(L)], plus
                I2[class(f)][bucket(L)] when the token before q1 also matches the token before that occurrence; every
                occurrence counts; buckets of L: 2-3, 4-7, 8-15, 16-31, 32-63, 64-127, 128-256  (ind.npz, 210 integers)

All quantities are integers (int64 accumulation) in the engine's 1/1024 logit units, added before the softmax; the
automaton, the three memory banks, answer mode and the belief memory are unchanged from 0.3.5. No neural layers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from blend_memory import QuarterBlendMicaWord

IND_WIN = 256
BUCKET = np.array([min(max(L.bit_length() - 2, 0), 6) for L in range(IND_WIN + 2)], np.int64)   # index = lag L


class PrevWordBank:
    def __init__(self, path: str | Path):
        d = np.load(Path(path), allow_pickle=False)
        self.A = d["A_int"].astype(np.int64)            # (16384, 16) int8 values
        self.Bm = d["Bm_int"].astype(np.int64)          # (16384, 16) int8 values
        self.MUL = d["MUL"].astype(np.int64)            # (16384,)
        self.R = d["R_int"].astype(np.int64)            # (5,) copy value per word class
        self.cls = d["class_of_id"].astype(np.int64)
        self.skip = set(int(x) for x in d["never_read"])

    def bonus(self, hist):
        p = next((int(t) for t in reversed(hist) if t not in self.skip), -1)
        if p < 0:
            return 0
        v = ((self.Bm @ self.A[p]) * self.MUL) >> 30
        v[p] += int(self.R[self.cls[p]])
        return v


class PhraseCopy:
    def __init__(self, path: str | Path):
        d = np.load(Path(path), allow_pickle=False)
        self.I1 = d["I1_int"].astype(np.int64)          # (5, 5, 7): class of q1, class of f, lag bucket
        self.I2 = d["I2_int"].astype(np.int64)          # (5, 7): class of f, lag bucket (two-word match)
        self.cls = d["class_of_id"].astype(np.int64)
        never = sorted(int(x) for x in d["never_read"])
        self.skip, self.skip_arr = set(never), np.array(never, np.int64)
        assert int(d["ind_win"]) == IND_WIN and np.array_equal(d["bucket_of_lag"], BUCKET)

    def bonus(self, hist):
        out = np.zeros(len(self.cls), np.int64)
        n = len(hist)
        if n < 2:
            return out
        q1 = int(hist[-1])
        if q1 in self.skip:
            return out
        w = min(n, IND_WIN)
        h = np.asarray(hist[n - w:], np.int64)
        i = np.flatnonzero(h[:w - 1] == q1)             # earlier occurrences of q1, at lag w - i >= 2
        if not len(i):
            return out
        f = h[i + 1]                                    # the token that followed each one
        keep = ~np.isin(f, self.skip_arr)
        i, f = i[keep], f[keep]
        if not len(i):
            return out
        b = BUCKET[w - i]
        cf = self.cls[f]
        v = self.I1[self.cls[q1], cf, b]
        q2 = int(hist[-2])
        if q2 not in self.skip:
            t = i >= 1                                  # two-word match inside the window
            t[t] = h[i[t] - 1] == q2
            v = v + self.I2[cf, b] * t
        np.add.at(out, f, v)
        return out


class ContextBlendMicaWord(QuarterBlendMicaWord):
    """0.3.5's model (automaton + integer memory blend) plus the previous-word bank and the phrase-copy table."""

    def __init__(self, base, released, fitted, long, bank: PrevWordBank, phrase: PhraseCopy):
        super().__init__(base, released, fitted, long)
        self.bank, self.phrase = bank, phrase
        self.info["context_banks_036"] = True

    def scores(self, st):
        return super().scores(st) + self.bank.bonus(st.hist) + self.phrase.bonus(st.hist)


def load(pkg: str | Path, base, answer_mode: bool = False) -> ContextBlendMicaWord:
    """The Flame-W 0.3.6 model from a package folder; answer_mode=True for multiple-choice scoring."""
    from memory import Memory
    pkg = Path(pkg)
    mode = pkg / "answer_mode.npz" if answer_mode else None
    return ContextBlendMicaWord(base, *(Memory(pkg / n, mode) for n in ("memory.npz", "memory_fitted.npz", "memory_long.npz")),
                                PrevWordBank(pkg / "bank.npz"), PhraseCopy(pkg / "ind.npz"))
