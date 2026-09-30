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
        if spec.PROBE_WINDOW:
            rng(self.pr_cell, N_CELLS - spec.PROBE_WINDOW, N_CELLS - 1,
                "pr_cell (probe window)")
        if spec.WINDOW and K and self.inj_chan.shape[1] > K:
            # a work injection outside the window would edit frozen history
            lag = (N_CELLS - self.inj_cell[:, K:].astype(np.int64)) % N_CELLS
            if (lag >= spec.WINDOW).any():
                raise ValueError("work injection outside the rule window")

    def copy(self) -> "Model":
        return Model(**{k: (None if getattr(self, k) is None else
                            getattr(self, k).copy())
                        for k in self.__dataclass_fields__})


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

    def check(self) -> None:
        assert self.F.min() >= -127 and self.F.max() <= 127
        assert self.phase.max() < N_PHASE

    def snapshot(self):
        return self.F.copy(), self.phase.copy(), self.position

    def restore(self, snap):
        F, ph, pos = snap
        self.F, self.phase, self.position = F.copy(), ph.copy(), pos


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
        s.F[pos, :] = 0
        s.F[pos, :K] = deltas[:K]
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
            return model.pr_bias.astype(np.int32) + coefficients @ values
    cells = model.pr_cell.astype(np.int32)
    if spec.ROLLING_READOUT:
        cells = (cells + s.position) % N_CELLS
    vals = s.F[cells, model.pr_chan.astype(np.int32)]
    return model.pr_bias.astype(np.int32) + (model.pr_co.astype(np.int32) * vals).sum(axis=1)
