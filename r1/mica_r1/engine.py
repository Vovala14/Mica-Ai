"""MICA R1 reference engine — the exact integer state transition.

This is the normative implementation: section 3's arithmetic contract, section
4's ingest, section 5's page addressing and candidate scoring, section 6's
synchronous single-writer tick, and section 8's readout.

It is vectorised with numpy for speed, but every operation is integer and the
semantics are the scalar ones. Specifically:

  * all arithmetic promotes to int32 before evaluation (section 3)
  * saturation clamps to [-127, 127]; -128 is forbidden (section 3)
  * candidate selection is argmax with ties to the lowest index (section 5);
    numpy's argmax returns the first maximum, which is that rule
  * a tick reads only the old F and old phase, and only cell i writes row i
    (section 6), so the result cannot depend on visitation order
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field as dc_field

import numpy as np

from . import spec
import numpy as _np
from .spec import (N_CELLS, N_CHANNELS, N_PAGES, N_CANDIDATES, N_SCORE_TERMS,
                   N_INJECT, N_PROBE, MAX_TICKS, N_SYMBOLS, BOS, EOS,
                   OFFSETS, N_PHASE, ROUTING_CHANNELS, PAGE_STRIDE,
                   ROUTING_PAIR_OFFSET, MAX_TICKS as TICKS)

OFF = np.array(OFFSETS, dtype=np.int32)
# cell addresses need two bytes once the field passes 256 cells
CELL_DT = np.uint16 if spec.WIDE_CELLS else np.uint8


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
@dataclass
class Model:
    """Learned data only. Dimensions and routing live in spec.py."""
    # injection: 258 symbols x 12 entries
    inj_cell: np.ndarray      # uint8   [258, 12]  base_cell 0..191
    inj_chan: np.ndarray      # uint8   [258, 12]  channel 0..23
    inj_delta: np.ndarray     # int8    [258, 12]  -127..127

    # candidate scoring: 64 pages x 32 candidates x 6 triples
    sc_nb: np.ndarray         # uint8   [64, 32, 6]  selector 0..6
    sc_ch: np.ndarray         # uint8   [64, 32, 6]  channel 0..23
    sc_co: np.ndarray         # int8    [64, 32, 6]  -1, 0, +1
    sc_bias: np.ndarray       # int16   [64, 32]

    # candidate operation arguments
    op_code: np.ndarray       # uint8   [64, 32]  0..7
    op_d: np.ndarray          # uint8   [64, 32]  destination channel
    op_n: np.ndarray          # uint8   [64, 32]  source neighbour selector
    op_c: np.ndarray          # uint8   [64, 32]  source channel
    op_a: np.ndarray          # int8    [64, 32]  coefficient
    op_b: np.ndarray          # int8    [64, 32]  immediate
    op_u: np.ndarray          # uint8   [64, 32]  auxiliary channel

    # readout: 258 symbols x 8 entries, absolute cells
    pr_cell: np.ndarray       # uint8   [258, 8]
    pr_chan: np.ndarray       # uint8   [258, 8]
    pr_co: np.ndarray         # int8    [258, 8]
    pr_bias: np.ndarray       # int16   [258]
    # tape extension (spec.VSET_WIDTH): VSET's extra immediates, stored in the
    # candidate record's reserved bytes. None when VSET is off.
    op_v: np.ndarray = None   # int8    [pages, candidates, VSET_WIDTH-1]

    def validate(self) -> None:
        """Loader requirements, section 11."""
        def rng(a, lo, hi, name):
            if a.size and (int(a.min()) < lo or int(a.max()) > hi):
                raise ValueError(f"{name} out of range [{lo},{hi}]")
        rng(self.inj_cell, 0, N_CELLS - 1, "inj_cell")
        rng(self.inj_chan, 0, N_CHANNELS - 1, "inj_chan")
        rng(self.inj_delta, -127, 127, "inj_delta")     # -128 forbidden
        rng(self.sc_nb, 0, len(OFFSETS) - 1, "sc_nb")
        rng(self.sc_ch, 0, N_CHANNELS - 1, "sc_ch")
        rng(self.sc_co, -1, 1, "sc_co")
        rng(self.op_code, 0, 8 if spec.VSET_WIDTH else 7, "op_code")
        if spec.VSET_WIDTH > 1:
            if self.op_v is None or self.op_v.shape != (N_PAGES, N_CANDIDATES,
                                                        spec.VSET_WIDTH - 1):
                raise ValueError("VSET needs op_v [pages, candidates, width-1]")
            rng(self.op_v, -127, 127, "op_v")
        for a, name in ((self.op_d, "op_d"), (self.op_c, "op_c"),
                        (self.op_u, "op_u")):
            rng(a, 0, N_CHANNELS - 1, name)
        rng(self.op_n, 0, len(OFFSETS) - 1, "op_n")
        rng(self.pr_cell, 0, N_CELLS - 1, "pr_cell")
        rng(self.pr_chan, 0, N_CHANNELS - 1, "pr_chan")
        rng(self.pr_co, spec.PROBE_CO_MIN, spec.PROBE_CO_MAX, "pr_co")
        K = spec.TAPE_CHANNELS
        if K:
            # tape entries are addressed by position, not by the table: the
            # file stores cell 0 (the head) and channel e so it reads plainly
            if (self.inj_cell[:, :K] != 0).any() or \
                    (self.inj_chan[:, :K] != np.arange(K)).any():
                raise ValueError("tape injection entries must be (head, e)")
            rng(self.inj_chan[:, K:], K, N_CHANNELS - 1, "inj_chan (work)")
            rng(self.op_d, K, N_CHANNELS - 1, "op_d (no tape writes)")
            rng(self.op_u, K, N_CHANNELS - 1, "op_u (no tape writes)")
        T = spec.TOPIC_CHANNELS
        if T:
            # topic entries, like tape entries, are (head, topic channel t);
            # nothing else may write the register
            if (self.inj_cell[:, K:K + T] != 0).any() or \
                    (self.inj_chan[:, K:K + T] != spec.TOPIC_AT + np.arange(T)).any():
                raise ValueError("topic injection entries must be (head, topic channel)")
            rng(self.inj_chan[:, K + T:], K, spec.TOPIC_AT - 1,
                "inj_chan (work, below the topic register)")
            last = self.op_d.astype(np.int64) + np.where(
                self.op_code == spec.VSET, max(1, spec.VSET_WIDTH), 1) - 1
            rng(last, K, spec.TOPIC_AT - 1, "op_d (no topic writes)")
            rng(self.op_u, K, spec.TOPIC_AT - 1, "op_u (no topic writes)")
        if spec.PROBE_WINDOW:
            rng(self.pr_cell, N_CELLS - spec.PROBE_WINDOW, N_CELLS - 1,
                "pr_cell (probe window)")
        if spec.WINDOW and K and self.inj_chan.shape[1] > K:
            # a work injection outside the window would edit frozen history
            lag = (N_CELLS - self.inj_cell[:, K:].astype(np.int64)) % N_CELLS
            if (lag >= spec.WINDOW).any():
                raise ValueError("work injection outside the rule window")

    def copy(self) -> "Model":
        m = Model(**{k: (None if getattr(self, k) is None else
                         getattr(self, k).copy())
                     for k in self.__dataclass_fields__})
        for name in ("topic_codes", "topic_w"):
            value = getattr(self, name, None)
            if value is not None:
                setattr(m, name, np.array(value, copy=True))
        for name in ("topic_shift", "topic_mul", "topic_rshift"):
            if hasattr(self, name):
                setattr(m, name, getattr(self, name))
        return m


def random_model(seed: int) -> Model:
    """Section 13 initialisation.

    "sample addresses uniformly in their valid ranges and coefficients
     uniformly from -1,0,+1. Initialize signature deltas uniformly from -3
     through +3, all biases to zero, all opcodes uniformly from 0 through 7,
     immediates uniformly from -2 through +2"
    """
    r = np.random.default_rng(seed)
    m = Model(
        inj_cell=r.integers(0, N_CELLS, (N_SYMBOLS, N_INJECT)).astype(CELL_DT),
        inj_chan=r.integers(0, N_CHANNELS, (N_SYMBOLS, N_INJECT), dtype=np.uint8),
        inj_delta=r.integers(-3, 4, (N_SYMBOLS, N_INJECT)).astype(np.int8),
        sc_nb=r.integers(0, len(OFFSETS), (N_PAGES, N_CANDIDATES, N_SCORE_TERMS),
                         dtype=np.uint8),
        sc_ch=r.integers(0, N_CHANNELS, (N_PAGES, N_CANDIDATES, N_SCORE_TERMS),
                         dtype=np.uint8),
        sc_co=r.integers(-1, 2, (N_PAGES, N_CANDIDATES, N_SCORE_TERMS)).astype(np.int8),
        sc_bias=np.zeros((N_PAGES, N_CANDIDATES), dtype=np.int16),
        op_code=r.integers(0, 8, (N_PAGES, N_CANDIDATES), dtype=np.uint8),
        op_d=r.integers(0, N_CHANNELS, (N_PAGES, N_CANDIDATES), dtype=np.uint8),
        op_n=r.integers(0, len(OFFSETS), (N_PAGES, N_CANDIDATES), dtype=np.uint8),
        op_c=r.integers(0, N_CHANNELS, (N_PAGES, N_CANDIDATES), dtype=np.uint8),
        op_a=r.integers(-1, 2, (N_PAGES, N_CANDIDATES)).astype(np.int8),
        op_b=r.integers(-2, 3, (N_PAGES, N_CANDIDATES)).astype(np.int8),
        op_u=r.integers(0, N_CHANNELS, (N_PAGES, N_CANDIDATES), dtype=np.uint8),
        pr_cell=r.integers(0, N_CELLS, (N_SYMBOLS, N_PROBE)).astype(CELL_DT),
        pr_chan=r.integers(0, N_CHANNELS, (N_SYMBOLS, N_PROBE), dtype=np.uint8),
        pr_co=r.integers(-1, 2, (N_SYMBOLS, N_PROBE)).astype(np.int8),
        pr_bias=np.zeros(N_SYMBOLS, dtype=np.int16),
    )
    if spec.VSET_WIDTH > 1:
        m.op_v = r.integers(-8, 9, (N_PAGES, N_CANDIDATES,
                                    spec.VSET_WIDTH - 1)).astype(np.int8)
        m.op_code = r.integers(0, 9, (N_PAGES, N_CANDIDATES), dtype=np.uint8)
    return conform(m, r) if spec.EXTENDED else m


def conform(m: Model, r=None) -> Model:
    """Bend a model into the tape extensions' address rules (spec.TAPE_CHANNELS,
    WINDOW, PROBE_WINDOW): tape entries at (head, e), no rule or work
    injection writing a tape channel, work injections and probes inside their
    windows. Used for random models; trained ones are born conforming."""
    r = r or np.random.default_rng(0)
    K = spec.TAPE_CHANNELS
    if K:
        m.inj_cell[:, :K] = 0
        m.inj_chan[:, :K] = np.arange(K, dtype=np.uint8)
        work = m.inj_chan[:, K:]
        m.inj_chan[:, K:] = np.where(work < K, r.integers(K, N_CHANNELS, work.shape),
                                     work).astype(np.uint8)
        for f in ("op_d", "op_u"):
            a = getattr(m, f)
            setattr(m, f, np.where(a < K, r.integers(K, N_CHANNELS, a.shape), a)
                    .astype(np.uint8))
        if spec.WINDOW:
            lag = r.integers(0, spec.WINDOW, m.inj_cell[:, K:].shape)
            m.inj_cell[:, K:] = ((N_CELLS - lag) % N_CELLS).astype(m.inj_cell.dtype)
        T = spec.TOPIC_CHANNELS
        if T:
            top = spec.TOPIC_AT
            m.inj_cell[:, K:K + T] = 0
            m.inj_chan[:, K:K + T] = top + np.arange(T, dtype=np.uint8)
            work = m.inj_chan[:, K + T:]
            m.inj_chan[:, K + T:] = np.where(
                work >= top, r.integers(K, top, work.shape), work).astype(np.uint8)
            width = np.where(m.op_code == spec.VSET, max(1, spec.VSET_WIDTH), 1)
            m.op_d = np.minimum(m.op_d, top - width).astype(np.uint8)
            m.op_u = np.where(m.op_u >= top, r.integers(K, top, m.op_u.shape),
                              m.op_u).astype(np.uint8)
    if spec.PROBE_WINDOW:
        lag = r.integers(1, spec.PROBE_WINDOW + 1, m.pr_cell.shape)
        m.pr_cell[:] = (N_CELLS - lag).astype(m.pr_cell.dtype)
    m.validate()
    return m


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------
@dataclass
class Session:
    """Persistent state, section 3 table.

    F is held as int32 for arithmetic convenience; its values are always a
    valid int8 in [-127, 127], which `check` asserts.
    """
    F: np.ndarray = dc_field(default_factory=lambda: np.zeros((N_CELLS, N_CHANNELS), np.int32))
    phase: np.ndarray = dc_field(default_factory=lambda: np.zeros(N_CELLS, np.uint8))
    position: int = 0
    ended: bool = False
    updates: int = 0          # active cell updates, for the section 12 penalty
    ticks_run: int = 0
    # Sidecar topic register (readout-only). None unless a topic readout is
    # bound. It is not a field channel: rules never read or write it, and it
    # is not part of the model file.
    topic: np.ndarray = None

    def check(self) -> None:
        assert self.F.min() >= -127 and self.F.max() <= 127
        assert self.phase.max() < N_PHASE
        if self.topic is not None:
            assert self.topic.min() >= -127 and self.topic.max() <= 127

    def snapshot(self):
        topic = None if self.topic is None else self.topic.copy()
        return self.F.copy(), self.phase.copy(), self.position, topic

    def restore(self, snap):
        if len(snap) == 3:
            F, ph, pos = snap
            topic = None
        else:
            F, ph, pos, topic = snap
        self.F, self.phase, self.position = F.copy(), ph.copy(), pos
        self.topic = None if topic is None else np.asarray(topic, np.int32).copy()


def new_session(model: Model) -> Session:
    """"new_session and reset both ingest BOS once after zeroing." (section 11)"""
    s = Session()
    ingest(model, s, BOS)
    return s


# ---------------------------------------------------------------------------
# the tick
# ---------------------------------------------------------------------------
# Routing variants. "r1" is the specification; the others exist only so the
# degeneracy measured in docs/r1-findings.md can be tested rather than argued.
ROUTING_MODE = os.environ.get("MICA_ROUTING", "r1")

# Section 16 ablation switches. Defaults are R1; each is turned on only by an
# ablation run, and the run records which one it used.
PHASE_ROUTING = os.environ.get("MICA_NO_PHASE_ROUTING", "0") != "1"
FULL_FIELD = os.environ.get("MICA_FULL_FIELD", "0") == "1"


def _routing_bits(F: np.ndarray, cells: np.ndarray, mode: str) -> np.ndarray:
    bits = np.zeros(cells.shape, dtype=np.int32)
    if mode == "r1":
        # "bits = sum((1 if F[i][c] >= 0 else 0) << c for c in range(4))"
        for c in ROUTING_CHANNELS:
            bits |= (F[cells, c] >= 0).astype(np.int32) << c
    elif mode == "positive":
        for c in ROUTING_CHANNELS:
            bits |= (F[cells, c] > 0).astype(np.int32) << c
    elif mode == "pairdiff":
        # Compare two channels instead of testing one against zero: an
        # untouched cell no longer has a privileged bit pattern.
        for k, c in enumerate(ROUTING_CHANNELS):
            bits |= (F[cells, c] >= F[cells, c + ROUTING_PAIR_OFFSET]).astype(np.int32) << k
    elif mode == "parity":
        for k, c in enumerate(ROUTING_CHANNELS):
            bits |= (F[cells, c] & 1).astype(np.int32) << k
    else:
        raise ValueError(mode)
    return bits


def _routing_bits_pair(F: np.ndarray, b: np.ndarray, i: np.ndarray) -> np.ndarray:
    """Routing bits for a batched (session, cell) index pair."""
    bits = np.zeros(b.shape, dtype=np.int32)
    mode = ROUTING_MODE
    if mode == "r1":
        for c in ROUTING_CHANNELS:
            bits |= (F[b, i, c] >= 0).astype(np.int32) << c
    elif mode == "positive":
        for c in ROUTING_CHANNELS:
            bits |= (F[b, i, c] > 0).astype(np.int32) << c
    elif mode == "pairdiff":
        for k, c in enumerate(ROUTING_CHANNELS):
            bits |= (F[b, i, c] >= F[b, i, c + ROUTING_PAIR_OFFSET]).astype(np.int32) << k
    elif mode == "parity":
        for k, c in enumerate(ROUTING_CHANNELS):
            bits |= (F[b, i, c] & 1) << k
    return bits


def _page_ids(F: np.ndarray, phase: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """page_id = 16 * phase[i] + bits, bits from the sign of channels 0..3.

    "Zero counts as nonnegative."
    """
    bits = _routing_bits(F, cells, ROUTING_MODE)
    if not PHASE_ROUTING:
        # ablation: the page index ignores phase, so 64 pages collapse to 16
        return bits
    return PAGE_STRIDE * phase[cells].astype(np.int32) + bits


def run_ticks(model: Model, s: Session, active: np.ndarray,
              max_ticks: int = MAX_TICKS) -> np.ndarray:
    """Section 6. Synchronous, single-writer, change-triggered, capped.

    Returns the activity set left over when it stops, which is empty in normal
    operation ("Clear all residual activity on return") and is returned only so
    that the section 7 single-tick fixture can inspect it.
    """
    m = model
    for _tick in range(max_ticks):
        if FULL_FIELD:
            # ablation: replace event activation with full-field updates
            active = np.ones(N_CELLS, dtype=np.uint8)
        cells = np.flatnonzero(active)
        if cells.size == 0:
            return active
        G, phase_next, opc, n = _update(m, s, cells)

        # ---- activation: change-triggered, plus PULSE targets --------------
        F = s.F
        next_active = np.zeros(N_CELLS, dtype=np.uint8)
        changed = cells[(G[cells] != F[cells]).any(axis=1)]
        if changed.size:
            next_active[changed] = 1
            for off in OFFSETS[1:]:
                next_active[(changed + off) % N_CELLS] = 1
        sel = opc == spec.PULSE
        if sel.any():
            next_active[(cells[sel] + OFF[n[sel]]) % N_CELLS] = 1

        s.F, s.phase = G, phase_next
        active = next_active
    return active


def run_window(model: Model, s: Session, head: int,
               ticks: int = None) -> None:
    """spec.WINDOW: exactly `ticks` synchronous updates of the W cells behind
    the write head (head, head-1, ...). No activation bookkeeping: the work per
    symbol is fixed, and the soft model trains this very machine."""
    cells = (head - np.arange(spec.WINDOW)) % N_CELLS
    for _tick in range(TICKS if ticks is None else ticks):
        G, phase_next, _opc, _n = _update(model, s, cells)
        s.F, s.phase = G, phase_next


def _update(m: Model, s: Session, cells: np.ndarray):
    """One synchronous tick of `cells`: every read comes from the old F.
    Returns the new field and phases, and each cell's opcode and neighbour
    selector (the activation rule needs them)."""
    s.ticks_run += 1
    s.updates += cells.size

    F = s.F
    G = F.copy()
    phase_next = s.phase.copy()

    pid = _page_ids(F, s.phase, cells)                      # (k,)

    # ---- candidate scoring: six triples per candidate ------------------
    nb = m.sc_nb[pid]                                       # (k,32,6)
    ch = m.sc_ch[pid]
    co = m.sc_co[pid].astype(np.int32)
    src = (cells[:, None, None] + OFF[nb]) % N_CELLS
    vals = F[src, ch]                                       # (k,32,6) int32
    scores = m.sc_bias[pid].astype(np.int32) + (co * vals).sum(axis=2)
    winner = scores.argmax(axis=1)                          # ties -> lowest

    # ---- operation arguments of the winning candidate ------------------
    w = (pid, winner)
    opc = m.op_code[w].astype(np.int32)
    d = m.op_d[w].astype(np.int32)
    n = m.op_n[w].astype(np.int32)
    c = m.op_c[w].astype(np.int32)
    a = m.op_a[w].astype(np.int32)
    b = m.op_b[w].astype(np.int32)
    u = m.op_u[w].astype(np.int32)

    v = F[(cells + OFF[n]) % N_CELLS, c].astype(np.int32)
    own_d = F[cells, d].astype(np.int32)
    own_u = F[cells, u].astype(np.int32)

    def clamp(x):
        return np.clip(x, spec.SAT_MIN, spec.SAT_MAX)

    # ADD
    sel = opc == spec.ADD
    if sel.any():
        G[cells[sel], d[sel]] = clamp(own_d[sel] + a[sel] * v[sel] + b[sel])
    # SET
    sel = opc == spec.SET
    if sel.any():
        G[cells[sel], d[sel]] = clamp(a[sel] * v[sel] + b[sel])
    # SWAP (d == u is a no-op, and assigning both ways handles it anyway)
    sel = (opc == spec.SWAP) & (d != u)
    if sel.any():
        G[cells[sel], d[sel]] = own_u[sel]
        G[cells[sel], u[sel]] = own_d[sel]
    # DECAY: one step toward zero
    sel = opc == spec.DECAY
    if sel.any():
        G[cells[sel], d[sel]] = own_d[sel] - np.sign(own_d[sel])
    # TURN: negate (safe: -127 is representable, -128 cannot occur)
    sel = opc == spec.TURN
    if sel.any():
        G[cells[sel], d[sel]] = -own_d[sel]
    # VSET (tape extension): channels d.. d+L-1 get b, v_1, ..., v_{L-1};
    # channels past the last one are not written
    if spec.VSET_WIDTH:
        sel = np.flatnonzero(opc == spec.VSET)
        if sel.size:
            vals = np.concatenate([b[sel, None],
                                   (m.op_v[w][sel].astype(np.int32)
                                    if spec.VSET_WIDTH > 1 else
                                    np.zeros((sel.size, 0), np.int32))], 1)
            for j in range(spec.VSET_WIDTH):
                ch = d[sel] + j
                ok = ch < N_CHANNELS
                G[cells[sel][ok], ch[ok]] = vals[ok, j]
    # HOLD, PULSE, QUIET write no field value.

    # ---- phase advances for every active cell, even HOLD and QUIET -----
    phase_next[cells] = (s.phase[cells].astype(np.int32) + 1) % N_PHASE
    return G, phase_next, opc, n


def topic_step(prev: np.ndarray, code: np.ndarray, shift: int = None) -> np.ndarray:
    """One symbol's update of the topic register.

    Integer and symmetric: |prev| >> shift is removed toward zero, the code
    is added, and the sum saturates. ``shift`` defaults to spec.TOPIC_SHIFT
    (3, so a code fades by 1/8 per symbol). No floating point.
    """
    if shift is None:
        shift = spec.TOPIC_SHIFT
    prev = np.asarray(prev, np.int32)
    code = np.asarray(code, np.int32)
    sign = np.where(prev > 0, np.int32(1),
                    np.where(prev < 0, np.int32(-1), np.int32(0))).astype(np.int32)
    decayed = prev - sign * (np.abs(prev) >> np.int32(shift))
    return np.clip(decayed + code, spec.SAT_MIN, spec.SAT_MAX).astype(np.int32)


def bind_topic_readout(model: Model, codes: np.ndarray, weights: np.ndarray,
                       shift: int = None, mul: int = 1, rshift: int = 0) -> Model:
    """Attach an int8 word×topic bias without touching the automaton.

    ``codes`` and ``weights`` are int8 ``(N_SYMBOLS, T)``. The register is a
    side vector on the session, updated by ``topic_step`` as symbols arrive.
    Rules, probes, injection and the ``.mica`` bytes stay as they were: this
    is a third readout, not a retargeting of the existing probes and not the
    phase-8–15 rule rewiring. Scores gain

        ((weights · register) * mul) >> rshift

    in the engine's integer logit units. With ``mul == 1`` and ``rshift == 0``
    that is the bare dot product. Call this before ``new_session``.
    """
    if shift is None:
        shift = spec.TOPIC_SHIFT
    codes = np.asarray(codes)
    weights = np.asarray(weights)
    if codes.dtype != np.int8 or weights.dtype != np.int8:
        raise ValueError("topic codes and weights must be int8")
    if codes.ndim != 2 or codes.shape != weights.shape or codes.shape[0] != N_SYMBOLS:
        raise ValueError(f"topic codes and weights must be int8 {(N_SYMBOLS, 'T')}")
    if codes.shape[1] < 1:
        raise ValueError("topic readout needs at least one channel")
    if (codes.view(np.uint8) == 128).any() or (weights.view(np.uint8) == 128).any():
        raise ValueError("forbidden -128 in topic codes or weights")
    if not 0 <= int(shift) <= 7 or int(mul) < 1 or int(rshift) < 0:
        raise ValueError("topic shift must be 0..7, mul >= 1, rshift >= 0")
    model.topic_codes = codes.copy()
    model.topic_w = weights.copy()
    model.topic_shift = int(shift)
    model.topic_mul = int(mul)
    model.topic_rshift = int(rshift)
    return model


def save_topic_readout(path, model: Model) -> None:
    """Write the sidecar table. The automaton file is not involved."""
    if getattr(model, "topic_w", None) is None:
        raise ValueError("no topic readout is bound")
    np.savez(path, codes=model.topic_codes, W=model.topic_w,
             shift=np.int64(model.topic_shift), mul=np.int64(model.topic_mul),
             rshift=np.int64(model.topic_rshift))


def load_topic_readout(path, model: Model) -> Model:
    """Bind a sidecar written by ``save_topic_readout``."""
    d = np.load(path, allow_pickle=False)
    return bind_topic_readout(model, d["codes"], d["W"], int(d["shift"]),
                              int(d["mul"]), int(d["rshift"]))


def topic_bonus(model: Model, session: Session):
    """Integer word×topic bonus, or None when no readout is bound.

    Runtime path: int8 weights, int32 register, int64 dot product. No
    floating point. None leaves ``probe_scores`` exactly as it was.
    """
    weights = getattr(model, "topic_w", None)
    if weights is None:
        return None
    width = weights.shape[1]
    reg = session.topic
    if reg is None:
        reg = np.zeros(width, np.int32)
    dot = weights.astype(np.int64) @ np.asarray(reg, np.int64)
    mul = int(getattr(model, "topic_mul", 1))
    rshift = int(getattr(model, "topic_rshift", 0))
    if mul != 1 or rshift:
        dot = (dot * np.int64(mul)) >> np.int64(rshift)
    lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
    if int(dot.min()) < lo or int(dot.max()) > hi:
        raise OverflowError("topic bonus does not fit int32")
    return dot.astype(np.int32)


def _with_topic(model: Model, session: Session, scores: np.ndarray) -> np.ndarray:
    bonus = topic_bonus(model, session)
    if bonus is None:
        return scores
    return scores + bonus


def record_symbols(record) -> np.ndarray:
    """Symbols of one training record, matching fit.contexts."""
    if N_SYMBOLS == 258:
        raw = record if isinstance(record, (bytes, bytearray)) else bytes(record)
        return np.frombuffer(raw, np.uint8).astype(np.int64)
    return np.asarray(record, np.int64)


def topic_register_rows(codes: np.ndarray, symbols, shift: int = None) -> np.ndarray:
    """Register after each ingested symbol. Row 0 is after BOS.

    ``symbols`` is the record without BOS or EOS. The returned array has
    one row per target (each record symbol, then the EOS that follows),
    and row k is the register that scores that target: lag 1, every channel.
    """
    if shift is None:
        shift = spec.TOPIC_SHIFT
    codes = np.asarray(codes)
    symbols = np.asarray(symbols, np.int64)
    width = codes.shape[1]
    rows = np.zeros((len(symbols) + 1, width), np.int32)
    state = topic_step(np.zeros(width, np.int32), codes[BOS], shift)
    rows[0] = state
    for i, sym in enumerate(symbols):
        state = topic_step(state, codes[int(sym)], shift)
        rows[i + 1] = state
    return rows


def topic_registers(records, codes: np.ndarray, shift: int = None) -> np.ndarray:
    """Stack ``topic_register_rows`` in record order (the sidecar features)."""
    if shift is None:
        shift = spec.TOPIC_SHIFT
    codes = np.asarray(codes)
    parts = [topic_register_rows(codes, record_symbols(r), shift) for r in records]
    if not parts:
        return np.zeros((0, codes.shape[1]), np.int32)
    return np.concatenate(parts, 0)


def topic_probe_features(records, codes: np.ndarray, probes, shift: int = None) -> np.ndarray:
    """Features for probes that read the topic register.

    ``probes`` is ``(probe_index, lag, channel)`` with ``channel`` at or above
    ``spec.TOPIC_AT``. The value is the simulated register, not a tape code
    and not a rule immediate. Lag 1 is the register after the last ingested
    symbol; a lag that reaches before BOS reads 0. Row order matches
    ``fit.contexts`` (record symbols, then EOS).
    """
    if shift is None:
        shift = spec.TOPIC_SHIFT
    codes = np.asarray(codes)
    lags = np.array([lag for _, lag, _ in probes], np.int64)
    ch = np.array([c - spec.TOPIC_AT for _, _, c in probes], np.int64)
    parts = []
    for record in records:
        reg = topic_register_rows(codes, record_symbols(record), shift)
        n = reg.shape[0]
        idx = np.arange(n)[:, None] + 1 - lags[None, :]
        valid = idx >= 0
        safe = np.clip(idx, 0, n - 1)
        vals = reg[safe, ch[None, :]]
        parts.append(np.where(valid, vals, 0).astype(np.int32))
    if not parts:
        return np.zeros((0, len(probes)), np.int32)
    return np.concatenate(parts, 0)


def _update_topic_register(model: Model, s: Session, symbol: int) -> None:
    codes = getattr(model, "topic_codes", None)
    if codes is None:
        return
    prev = s.topic if s.topic is not None else np.zeros(codes.shape[1], np.int32)
    s.topic = topic_step(prev, codes[int(symbol)], getattr(model, "topic_shift", None))


def ingest(model: Model, s: Session, symbol: int) -> None:
    """Section 4 exact ingest preparation."""
    active = np.zeros(N_CELLS, dtype=np.uint8)
    cells = model.inj_cell[symbol].astype(np.int32)
    chans = model.inj_chan[symbol].astype(np.int32)
    deltas = model.inj_delta[symbol].astype(np.int32)
    pos = s.position
    first = 0
    if spec.TAPE_CHANNELS:
        # spec.TAPE_CHANNELS: the head cell starts a new life. Whatever it held
        # N_CELLS symbols ago is cleared (and its phase with it), then the
        # symbol's code is SET on the tape channels.
        K = first = spec.TAPE_CHANNELS
        T = spec.TOPIC_CHANNELS
        if T:
            # read the previous head's register before this cell is cleared
            prev = s.F[(pos - 1) % N_CELLS, spec.TOPIC_AT:].astype(np.int32)
        s.F[pos, :] = 0
        s.F[pos, :K] = deltas[:K]
        if T:
            # spec.TOPIC_CHANNELS: decay one step toward zero, add the code
            s.F[pos, spec.TOPIC_AT:] = topic_step(prev, deltas[K:K + T])
            first = K + T
        s.phase[pos] = 0
        active[pos] = 1
    # "Each injection saturates immediately in entry order", so this loop is
    # sequential on purpose: two opposite deltas on a saturated channel need
    # not cancel.
    for k in range(first, N_INJECT):
        i = (int(cells[k]) + pos) % N_CELLS
        ch = int(chans[k])
        s.F[i, ch] = spec.sat(int(s.F[i, ch]) + int(deltas[k]))
        active[i] = 1
    if spec.WINDOW:
        run_window(model, s, pos)
    else:
        run_ticks(model, s, active)
    s.position = (pos + 1) % N_CELLS
    # Sidecar register only. Absent when no readout is bound, so the field
    # and the rules are untouched.
    _update_topic_register(model, s, symbol)


def probe_scores(model: Model, s: Session) -> np.ndarray:
    """Section 8. int32 accumulation, absolute cell addresses, no rotation --
    unless spec.ROLLING_READOUT, when each probe cell is relative to the write
    head (cell + position), the same frame section 4 uses for injection."""
    if N_SYMBOLS > 258:
        # Fit mode fixes the probe wiring for every word. Gather the field
        # once, then apply each word's learned integer coefficients. The
        # fallback below remains the exact path for other rule books.
        cache = getattr(model, "_shared_word_probes", None)
        if cache is None:
            shared = (np.all(model.pr_cell == model.pr_cell[0]) and
                      np.all(model.pr_chan == model.pr_chan[0]))
            cache = (model.pr_cell[0].astype(np.int32),
                     model.pr_chan[0].astype(np.int32),
                     model.pr_co.astype(np.int32)) if shared else False
            model._shared_word_probes = cache
        if cache is not False:
            cells, chans, coefficients = cache
            if spec.ROLLING_READOUT:
                cells = (cells + s.position) % N_CELLS
            values = s.F[cells, chans].astype(np.int32)
            return _with_topic(model, s, model.pr_bias.astype(np.int32) + coefficients @ values)
    cells = model.pr_cell.astype(np.int32)
    if spec.ROLLING_READOUT:
        cells = (cells + s.position) % N_CELLS
    vals = s.F[cells, model.pr_chan.astype(np.int32)]
    return _with_topic(
        model, s,
        model.pr_bias.astype(np.int32) + (model.pr_co.astype(np.int32) * vals).sum(axis=1))
