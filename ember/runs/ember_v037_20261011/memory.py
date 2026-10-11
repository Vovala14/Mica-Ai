"""Standalone byte adaptation of Flame-W 0.3.1's integer memory readout.

The frozen automaton still consumes bytes. This sidecar adds recall and a rank-16
association over 64 previous bytes, in the base model's integer logit units.
Unlike the word model, byte zero is real data; only BOS/EOS are skipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

V, BOS, EOS, WINDOW, SHIFT, RANK = 258, 256, 257, 64, 30, 16
CLASS_NAMES = ("whitespace", "lowercase", "uppercase", "digits", "punctuation", "high_or_control")
NC, NB = len(CLASS_NAMES), 7
LAG_BUCKET = np.array([0, 0, 1, 2, 3, 3, 3] + [4] * 6 + [5] * 12 + [6] * 40, np.int64)
LAG_BUCKET.flags.writeable = False
I64_MAX = int(np.iinfo(np.int64).max)


def byte_classes() -> np.ndarray:
    """Fixed ASCII byte classes; no vocabulary or held-out data is consulted."""
    out = np.full(V, 5, np.uint8)
    out[33:127] = 4
    out[list(b" \t\n\r\v\f")] = 0
    out[ord("a"):ord("z") + 1] = 1
    out[ord("A"):ord("Z") + 1] = 2
    out[ord("0"):ord("9") + 1] = 3
    return out


def _readonly(value, dtype=None):
    a = np.asarray(value, dtype=dtype)
    # A bytes-backed array cannot be made writable again via setflags.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _integer(value, name, shape, low, high, dtype):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array with shape {shape}")
    if np.any(a < low) or np.any(a > high):
        raise ValueError(f"{name} values must be in [{low}, {high}]")
    return _readonly(a, dtype)


def _float(value, name, shape):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return a


def quantize(A=None, Bm=None, lam=None, R=None, *, divisor=1024.0, class_of_id=None):
    """Quantize float parameters measured in nats, using deterministic rounding.

Pass R alone for recall-only storage. A, Bm and lam must be provided together.
R: [6,7]; A/Bm: [258,16]; lam: [7]. No pickle or torch dependency is used.
"""
    divisor = float(divisor)
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("divisor must be finite and positive")
    if R is None:
        raise ValueError("R is required")
    rq = np.rint(_float(R, "R", (NC, NB)) * divisor)
    if not np.isfinite(rq).all() or np.any(np.abs(rq) > np.iinfo(np.int32).max):
        raise ValueError("quantized R exceeds int32 range")
    rdtype = np.int16 if np.max(np.abs(rq)) <= np.iinfo(np.int16).max else np.int32
    out = {
        "R_int": rq.astype(rdtype),
        "class_of_id": byte_classes() if class_of_id is None else np.asarray(class_of_id),
        "class_names": np.asarray(CLASS_NAMES),
        "never_read": np.asarray([BOS, EOS], np.int16),
        "window": np.asarray(WINDOW, np.int16),
        "shift": np.asarray(SHIFT, np.int16),
        "divisor": np.asarray(divisor, np.float64),
        "version": np.asarray(1, np.int16),
    }
    supplied = [x is not None for x in (A, Bm, lam)]
    if any(supplied) and not all(supplied):
        raise ValueError("A, Bm and lam must be supplied together")
    if all(supplied):
        A = _float(A, "A", (V, RANK))
        Bm = _float(Bm, "Bm", (V, RANK))
        lam = _float(lam, "lam", (NB,))
        sa = max(float(np.abs(A).max()) / 127.0, np.finfo(np.float64).tiny)
        sl = max(float(np.abs(lam).max()) / 127.0, np.finfo(np.float64).tiny)
        sb = np.maximum(np.abs(Bm).max(axis=1) / 127.0, np.finfo(np.float64).tiny)
        mul = np.rint(divisor * sa * sl * sb * 2.0 ** SHIFT)
        # Float64 rounds int64's upper endpoint to 2**63, which is not valid.
        if not np.isfinite(mul).all() or np.any(mul < 0) or np.any(mul >= 2.0 ** 63):
            raise ValueError("quantized association multiplier exceeds int64 range")
        out.update(A_int=np.rint(A / sa).astype(np.int8),
                   Bm_int=np.rint(Bm / sb[:, None]).astype(np.int8),
                   lam_int=np.rint(lam / sl).astype(np.int8), MUL=mul.astype(np.int64))
    # Validate arithmetic bounds before publishing parameters.
    return dict(Memory(out).tables)


def save_memory(path: str | Path, tables: Mapping) -> None:
    """Write validated compact arrays; loading always uses allow_pickle=False."""
    checked = Memory(tables)
    np.savez(Path(path), **checked.tables)


class Memory:
    def __init__(self, path_or_mapping: str | Path | Mapping, recall_only=False):
        if isinstance(path_or_mapping, Mapping):
            d = dict(path_or_mapping)
        else:
            with np.load(Path(path_or_mapping), allow_pickle=False) as archive:
                d = {k: archive[k] for k in archive.files}
        for key, expected in (("window", WINDOW), ("shift", SHIFT), ("version", 1)):
            a = np.asarray(d[key])
            if a.shape != () or a.dtype.kind not in "iu" or int(a) != expected:
                raise ValueError(f"{key} must equal {expected}")
        a = np.asarray(d["divisor"])
        if a.shape != () or not np.isfinite(float(a)) or float(a) <= 0:
            raise ValueError("divisor must be a finite positive scalar")
        self.divisor = float(a)
        names = np.asarray(d["class_names"])
        if names.shape != (NC,) or tuple(names.tolist()) != CLASS_NAMES:
            raise ValueError("class_names must describe the fixed byte classes")
        cls = _integer(d["class_of_id"], "class_of_id", (V,), 0, NC - 1, np.uint8)
        if not np.array_equal(cls, byte_classes()):
            raise ValueError("class_of_id does not match the fixed byte classes")
        never = _integer(d["never_read"], "never_read", (2,), BOS, EOS, np.int16)
        if tuple(never) != (BOS, EOS):
            raise ValueError("never_read must be exactly [BOS, EOS]")
        raw_r = np.asarray(d["R_int"])
        if raw_r.dtype not in (np.dtype("int16"), np.dtype("int32")):
            raise ValueError("R_int must use int16 or int32 storage")
        r = _integer(raw_r, "R_int", (NC, NB), -2 ** 31, 2 ** 31 - 1, raw_r.dtype)
        compact = {k: _readonly(d[k]) for k in ("window", "shift", "version", "divisor", "class_names")}
        compact.update(R_int=r, class_of_id=cls, never_read=never)
        self.R, self.cls = _readonly(r, np.int64), _readonly(cls, np.int64)
        assoc_keys = ("A_int", "Bm_int", "lam_int", "MUL")
        present = [key in d for key in assoc_keys]
        if any(present) and not all(present):
            raise ValueError("incomplete association tables")
        self.assoc = all(present) and not recall_only
        self.max_bonus_abs = WINDOW * int(np.abs(self.R).max())
        if all(present):
            for key, shape in (("A_int", (V, RANK)), ("Bm_int", (V, RANK)), ("lam_int", (NB,))):
                if np.asarray(d[key]).dtype != np.dtype("int8"):
                    raise ValueError(f"{key} must use int8 storage")
                compact[key] = _integer(d[key], key, shape, -127, 127, np.int8)
            compact["MUL"] = _integer(d["MUL"], "MUL", (V,), 0, I64_MAX, np.int64)
            self.A = _readonly(compact["A_int"], np.int64)
            self.Bm = _readonly(compact["Bm_int"], np.int64)
            self.lam = _readonly(compact["lam_int"], np.int64)
            self.MUL = compact["MUL"]
            # Use Python ints to bound every possible 64-byte history without
            # overflowing during the validation itself. Specials cannot enter M.
            lag_mass = sum(abs(int(self.lam[b])) for b in LAG_BUCKET[1:])
            mbound = [lag_mass * max(abs(int(x)) for x in self.A[:256, j]) for j in range(RANK)]
            largest = 0
            for row, mul in zip(self.Bm, self.MUL):
                dot_bound = sum(abs(int(b)) * m for b, m in zip(row, mbound))
                product = dot_bound * int(mul)
                if dot_bound > I64_MAX or product > I64_MAX:
                    raise ValueError("association accumulation can overflow int64")
                largest = max(largest, (product + (1 << SHIFT) - 1) >> SHIFT)
            if self.assoc:
                self.max_bonus_abs += largest
        if self.max_bonus_abs > I64_MAX:
            raise ValueError("memory bonus can overflow int64")
        self.tables = MappingProxyType(compact)

    @staticmethod
    def _history(hist, ndim):
        a = np.asarray(hist)
        if a.size == 0 and a.dtype.kind not in "iu":
            a = a.astype(np.int64)
        if a.ndim != ndim or a.dtype.kind not in "iu" or np.any(a < -1) or np.any(a >= V):
            raise ValueError(f"history must be a {ndim}-dimensional integer array in [-1,257]")
        return a.astype(np.int64, copy=False)

    def bonus(self, hist) -> np.ndarray:
        """One history, oldest first. Special/padded ids keep their lag slots."""
        h = self._history(hist, 1)[-WINDOW:][::-1]
        out = np.zeros(V, np.int64)
        ids, buckets = [], []
        for lag, token in enumerate(h, 1):
            if 0 <= token < BOS:
                bucket = int(LAG_BUCKET[lag])
                out[token] += self.R[self.cls[token], bucket]
                ids.append(int(token)); buckets.append(bucket)
        if self.assoc and ids:
            M = (self.A[ids] * self.lam[buckets, None]).sum(axis=0, dtype=np.int64)
            out += ((self.Bm @ M) * self.MUL) >> SHIFT
        return out

    def bonus_batch(self, histories) -> np.ndarray:
        """Histories [n, <=64], newest first, -1 padding; returns [n,258]."""
        H = self._history(histories, 2)
        if H.shape[1] > WINDOW:
            raise ValueError("batched histories must have at most 64 lag columns")
        n, width = H.shape
        out = np.zeros((n, V), np.int64)
        valid = (H >= 0) & (H < BOS)
        ids = np.where(valid, H, 0)
        lb = LAG_BUCKET[1:width + 1]
        values = self.R[self.cls[ids], lb] * valid
        np.add.at(out, (np.arange(n)[:, None], ids), values)
        if self.assoc and width:
            w = self.lam[lb][None, :] * valid
            M = (self.A[ids] * w[:, :, None]).sum(axis=1, dtype=np.int64)
            out += ((M @ self.Bm.T) * self.MUL[None, :]) >> SHIFT
        return out


@dataclass
class MemState:
    s: object
    hist: tuple[int, ...]


class MemoryMicaByte:
    """Wrap an integer base model with start/feed/fork/scores/logp methods."""
    def __init__(self, base, memory: Memory):
        if float(base.div) != memory.divisor:
            raise ValueError("memory and base model logit divisors differ")
        if np.asarray(base.elig).shape != (V,):
            raise ValueError("base model must have 258 symbols")
        self.base, self.memory = base, memory
        self.div, self.elig = base.div, base.elig
        self.info = dict(getattr(base, "info", {}), memory=True, memory_window_bytes=WINDOW)

    def start(self):
        return MemState(self.base.start(), ())

    def feed(self, st, i: int):
        if not isinstance(i, (int, np.integer)) or not 0 <= i < V:
            raise ValueError("symbol must be an integer in [0,257]")
        st.s = self.base.feed(st.s, int(i))
        st.hist = (st.hist + (int(i),))[-WINDOW:]
        return st

    def fork(self, st):
        return MemState(self.base.fork(st.s), st.hist)

    def scores(self, st):
        raw = np.asarray(self.base.scores(st.s))
        if raw.shape != (V,) or raw.dtype.kind not in "iu":
            raise ValueError("base scores must be 258 integer logits")
        margin = self.memory.max_bonus_abs
        if np.any(raw > I64_MAX - margin) or np.any(raw < -I64_MAX + margin):
            raise ValueError("base plus memory score can overflow int64")
        return raw.astype(np.int64) + self.memory.bonus(st.hist)

    def logp(self, st, ids):
        z = self.scores(st).astype(np.float64) / self.div
        z = np.where(self.elig, z, -np.inf)
        z -= z.max()
        return z[np.asarray(ids, dtype=np.int64)] - np.log(np.exp(z[self.elig]).sum())
