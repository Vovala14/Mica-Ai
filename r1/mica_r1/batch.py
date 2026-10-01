"""Batched evaluator: many sessions under many models, in one vectorised pass.

Section 14 prices R1's own search at "over 11 billion target evaluations" and
notes it "can be prohibitively slow". A scalar interpreter makes the search
impossible rather than merely slow, so the trainer evaluates the incumbent and
all 16 children on all 32 records simultaneously: 544 independent sessions
stepped together.

The semantics are unchanged. Each session keeps its own F, phase, position and
activity, and every read still comes from the old F of that session.
"""

from __future__ import annotations

import numpy as np

from . import spec
from .engine import Model
from .spec import (N_CELLS, N_CHANNELS, N_PAGES, N_CANDIDATES, N_SCORE_TERMS,
                   N_INJECT, N_PROBE, MAX_TICKS, N_SYMBOLS, OFFSETS,
                   ROUTING_CHANNELS, N_PHASE, PAGE_STRIDE, BOS, EOS)

OFF = np.array(OFFSETS, dtype=np.int32)
CHUNK = 16384          # active-cell rows scored at once, to bound peak memory
# verification hook (r1/checks/verify_exact.py): a list -> every tick appends
# (backend, sessions, cells, pages, winners, top score, candidates at the top)
TRACE = None


class ModelStack:
    """K models with a leading axis, so a child's rules are one index away."""

    FIELDS = ("inj_cell", "inj_chan", "inj_delta", "sc_nb", "sc_ch", "sc_co",
              "sc_bias", "op_code", "op_d", "op_n", "op_c", "op_a", "op_b",
              "op_u", "pr_cell", "pr_chan", "pr_co", "pr_bias")

    def __init__(self, models: list[Model], score_backend: str = "numpy"):
        if spec.TOPIC_CHANNELS:
            raise NotImplementedError("this batch machine has no topic register; "
                                      "use engine.py for topic models")
        if score_backend not in ("numpy", "c"):
            raise ValueError(score_backend)
        self.K = len(models)
        self.scorer = None
        for f in self.FIELDS:
            setattr(self, f, np.stack([getattr(m, f) for m in models]))
        self.op_v = (np.stack([m.op_v for m in models])
                     if spec.VSET_WIDTH > 1 else None)
        if score_backend == "c":
            # Exact same arithmetic, merged duplicate reads, zero terms dropped.
            from .c_score import CompiledScorer
            self.scorer = CompiledScorer(self)

    def __getitem__(self, k: int) -> Model:
        m = Model(**{f: getattr(self, f)[k].copy() for f in self.FIELDS})
        if self.op_v is not None:
            m.op_v = self.op_v[k].copy()
        return m


class BatchSession:
    def __init__(self, n: int, model_of: np.ndarray):
        self.n = n
        self.mi = model_of.astype(np.int32)            # (n,) model index per session
        self.F = np.zeros((n, N_CELLS, N_CHANNELS), np.int32)
        self.phase = np.zeros((n, N_CELLS), np.int32)
        self.position = np.zeros(n, np.int32)
        self.updates = np.zeros(n, np.int64)


def _run_ticks(ms: ModelStack, st: BatchSession, active: np.ndarray) -> None:
    from .engine import FULL_FIELD
    for _ in range(MAX_TICKS):
        if FULL_FIELD:
            active = np.ones_like(active)
        bs, cs = np.nonzero(active)
        if bs.size == 0:
            return
        next_active = np.zeros_like(active)
        _update(ms, st, bs, cs, next_active)
        active = next_active


def _run_window(ms: ModelStack, st: BatchSession, head: np.ndarray) -> None:
    """spec.WINDOW: exactly MAX_TICKS updates of the W cells behind each
    session's write head. Same machine as engine.run_window."""
    W = spec.WINDOW
    bs = np.repeat(np.arange(st.n), W)
    cs = ((head[:, None] - np.arange(W)[None, :]) % N_CELLS).reshape(-1)
    for _ in range(MAX_TICKS):
        _update(ms, st, bs, cs, None)


def _update(ms: ModelStack, st: BatchSession, bs: np.ndarray, cs: np.ndarray,
            next_active) -> None:
    """One synchronous tick of the (session, cell) pairs (bs, cs). Fills
    next_active with the change-triggered activation when it is given."""
    from .engine import PHASE_ROUTING
    st.updates += np.bincount(bs, minlength=st.n)

    F = st.F
    G = F.copy()
    phase_next = st.phase.copy()

    for start in range(0, bs.size, CHUNK):
        b = bs[start:start + CHUNK]
        i = cs[start:start + CHUNK]
        k = st.mi[b]

        from .engine import _routing_bits_pair
        bits = _routing_bits_pair(F, b, i)
        pid = ((PAGE_STRIDE * st.phase[b, i] + bits)
               if PHASE_ROUTING else bits)

        if ms.scorer is not None:
            win = ms.scorer.winners(F, b, i, k, pid)
            if TRACE is not None:
                TRACE.append(("c", b.copy(), i.copy(), pid.copy(), win.copy(),
                              None, None))
        else:
            nb = ms.sc_nb[k, pid]                              # (m,32,6)
            ch = ms.sc_ch[k, pid]
            co = ms.sc_co[k, pid].astype(np.int32)
            src = (i[:, None, None] + OFF[nb]) % N_CELLS
            vals = F[b[:, None, None], src, ch]
            scores = ms.sc_bias[k, pid].astype(np.int32) + (co * vals).sum(2)
            win = scores.argmax(1)                             # ties -> lowest
            if TRACE is not None:
                top = scores.max(1)
                TRACE.append(("numpy", b.copy(), i.copy(), pid.copy(),
                              win.copy(), top,
                              (scores == top[:, None]).sum(1)))

        opc = ms.op_code[k, pid, win].astype(np.int32)
        d = ms.op_d[k, pid, win].astype(np.int32)
        n_ = ms.op_n[k, pid, win].astype(np.int32)
        c_ = ms.op_c[k, pid, win].astype(np.int32)
        a = ms.op_a[k, pid, win].astype(np.int32)
        imm = ms.op_b[k, pid, win].astype(np.int32)
        u = ms.op_u[k, pid, win].astype(np.int32)

        v = F[b, (i + OFF[n_]) % N_CELLS, c_]
        own_d = F[b, i, d]
        own_u = F[b, i, u]
        cl = lambda x: np.clip(x, spec.SAT_MIN, spec.SAT_MAX)

        s = opc == spec.ADD
        if s.any():
            G[b[s], i[s], d[s]] = cl(own_d[s] + a[s] * v[s] + imm[s])
        s = opc == spec.SET
        if s.any():
            G[b[s], i[s], d[s]] = cl(a[s] * v[s] + imm[s])
        s = (opc == spec.SWAP) & (d != u)
        if s.any():
            G[b[s], i[s], d[s]] = own_u[s]
            G[b[s], i[s], u[s]] = own_d[s]
        s = opc == spec.DECAY
        if s.any():
            G[b[s], i[s], d[s]] = own_d[s] - np.sign(own_d[s])
        s = opc == spec.TURN
        if s.any():
            G[b[s], i[s], d[s]] = -own_d[s]
        if spec.VSET_WIDTH:
            s = np.flatnonzero(opc == spec.VSET)
            if s.size:
                vals = [imm[s]]
                if ms.op_v is not None:
                    ov = ms.op_v[k[s], pid[s], win[s]].astype(np.int32)
                    vals += [ov[:, j] for j in range(ov.shape[1])]
                for j, v in enumerate(vals):
                    ch = d[s] + j
                    ok = ch < N_CHANNELS
                    G[b[s][ok], i[s][ok], ch[ok]] = v[ok]

        phase_next[b, i] = (st.phase[b, i] + 1) % N_PHASE

        s = opc == spec.PULSE
        if next_active is not None and s.any():
            next_active[b[s], (i[s] + OFF[n_[s]]) % N_CELLS] = True

    if next_active is not None:
        changed_b, changed_i = np.nonzero((G != F).any(axis=2))
        if changed_b.size:
            next_active[changed_b, changed_i] = True
            for off in OFFSETS[1:]:
                next_active[changed_b, (changed_i + off) % N_CELLS] = True

    st.F, st.phase = G, phase_next


def ingest_batch(ms: ModelStack, st: BatchSession, symbols: np.ndarray) -> None:
    """One symbol per session. Entry order matters, so the 12 writes loop."""
    active = np.zeros((st.n, N_CELLS), bool)
    rows = np.arange(st.n)
    k = st.mi
    first = 0
    if spec.TAPE_CHANNELS:
        # the head cell starts a new life: cleared, phase 0, code SET on the
        # tape channels (engine.ingest has the reasoning)
        K = first = spec.TAPE_CHANNELS
        head = st.position
        st.F[rows, head, :] = 0
        st.F[rows, head, :K] = ms.inj_delta[k, symbols, :K].astype(np.int32)
        st.phase[rows, head] = 0
        active[rows, head] = True
    for e in range(first, N_INJECT):
        cell = (ms.inj_cell[k, symbols, e].astype(np.int32) + st.position) % N_CELLS
        chan = ms.inj_chan[k, symbols, e].astype(np.int32)
        delta = ms.inj_delta[k, symbols, e].astype(np.int32)
        cur = st.F[rows, cell, chan]
        st.F[rows, cell, chan] = np.clip(cur + delta, spec.SAT_MIN, spec.SAT_MAX)
        active[rows, cell] = True
    if spec.WINDOW:
        _run_window(ms, st, st.position)
    else:
        _run_ticks(ms, st, active)
    st.position = (st.position + 1) % N_CELLS


def probe_batch(ms: ModelStack, st: BatchSession) -> np.ndarray:
    """(n, 258) int32 scores."""
    k = st.mi
    cell = ms.pr_cell[k].astype(np.int32)        # (n,258,8)
    if spec.ROLLING_READOUT:
        cell = (cell + st.position[:, None, None]) % N_CELLS
    chan = ms.pr_chan[k].astype(np.int32)
    co = ms.pr_co[k].astype(np.int32)
    vals = st.F[np.arange(st.n)[:, None, None], cell, chan]
    return ms.pr_bias[k].astype(np.int32) + (co * vals).sum(2)


# ---------------------------------------------------------------------------
# section 12 objective
# ---------------------------------------------------------------------------
_ELIGIBLE = np.zeros(N_SYMBOLS, bool)
_ELIGIBLE[0:spec.BOS] = True
_ELIGIBLE[EOS] = True
_ELIG_IDX = np.flatnonzero(_ELIGIBLE)


def evaluate(ms: ModelStack, records: list[bytes], model_of_row=None,
             per_record: bool = False):
    """Mean natural-log target loss and mean active updates, per (model, record).

    Section 12: reset (which ingests BOS once), predict each next true byte
    before ingesting it, and finally predict EOS. BOS is excluded from the
    normalisation. Scores are divided by 16 to obtain logits, and the negative
    log probability is accumulated in float64 with a stable log-sum-exp.

    Returns (loss (K,), updates (K,), targets (K,)).
    """
    K, R = ms.K, len(records)
    mi = np.repeat(np.arange(K), R) if model_of_row is None else model_of_row
    st = BatchSession(K * R, mi)

    # reset: ingest BOS once for every session
    ingest_batch(ms, st, np.full(st.n, BOS, np.int32))

    maxlen = max(len(r) for r in records)
    padded = np.full((R, maxlen), -1, np.int32)
    for j, r in enumerate(records):
        padded[j, :len(r)] = (np.frombuffer(r, np.uint8) if N_SYMBOLS == 258
                              else np.asarray(r, dtype=np.int32))
    lengths = np.array([len(r) for r in records])

    tiled = np.tile(padded, (K, 1))                      # (K*R, maxlen)
    tiled_len = np.tile(lengths, K)

    nats = np.zeros(st.n)
    counts = np.zeros(st.n, np.int64)

    for t in range(maxlen + 1):
        scores = probe_batch(ms, st).astype(np.float64) / spec.LOGIT_DIVISOR
        target = np.where(t < tiled_len, tiled[:, min(t, maxlen - 1)], EOS)
        alive = t <= tiled_len                           # the EOS step is included
        elig = scores[:, _ELIG_IDX]
        mx = elig.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(elig - mx).sum(axis=1))
        nats += np.where(alive, lse - scores[np.arange(st.n), target], 0.0)
        counts += alive

        if t >= maxlen:
            break
        feed = np.where(t < tiled_len, tiled[:, t], BOS)  # BOS is inert padding
        ingest_batch(ms, st, feed.astype(np.int32))

    n_kr = nats.reshape(K, R); c_kr = counts.reshape(K, R)
    u_kr = st.updates.reshape(K, R)
    tgt = c_kr.sum(1)
    loss = n_kr.sum(1) / tgt
    # Mean ACTIVE UPDATES PER SYMBOL, so the section 12 penalty divides by the
    # 2,304 per-symbol bound and lands in [0, 1].
    updates = u_kr.sum(1) / tgt
    if per_record:
        # Per (model, record) rates. Record counts are identical across
        # models, so a paired comparison across records is well defined.
        return loss, updates, tgt, n_kr / c_kr, u_kr / c_kr
    return loss, updates, tgt


def objective(loss: np.ndarray, updates: np.ndarray, weight: float = 0.01):
    """J = mean natural-log target loss + 0.01 * mean active updates / 2304."""
    return loss + weight * updates / spec.MAX_UPDATES_PER_SYMBOL
