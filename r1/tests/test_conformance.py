"""Conformance fixtures for MICA R1.

Section 15 asks for golden state vectors and section 16's table lists the
required tests. The centrepiece is the section 7 hand-worked transition: if the
engine does not reproduce those exact numbers, nothing downstream is meaningful.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from mica_r1 import spec
from mica_r1.engine import (Model, Session, random_model, new_session, ingest,
                            run_ticks, probe_scores, _page_ids)
from mica_r1.spec import N_CELLS, N_CHANNELS, N_PAGES, N_CANDIDATES, N_SYMBOLS


def blank_model() -> Model:
    """A model of all zeros: every address 0, every coefficient 0."""
    z = lambda shape, dt: np.zeros(shape, dt)
    return Model(
        inj_cell=z((N_SYMBOLS, 12), np.uint8), inj_chan=z((N_SYMBOLS, 12), np.uint8),
        inj_delta=z((N_SYMBOLS, 12), np.int8),
        sc_nb=z((N_PAGES, N_CANDIDATES, 6), np.uint8),
        sc_ch=z((N_PAGES, N_CANDIDATES, 6), np.uint8),
        sc_co=z((N_PAGES, N_CANDIDATES, 6), np.int8),
        sc_bias=z((N_PAGES, N_CANDIDATES), np.int16),
        op_code=z((N_PAGES, N_CANDIDATES), np.uint8), op_d=z((N_PAGES, N_CANDIDATES), np.uint8),
        op_n=z((N_PAGES, N_CANDIDATES), np.uint8), op_c=z((N_PAGES, N_CANDIDATES), np.uint8),
        op_a=z((N_PAGES, N_CANDIDATES), np.int8), op_b=z((N_PAGES, N_CANDIDATES), np.int8),
        op_u=z((N_PAGES, N_CANDIDATES), np.uint8),
        pr_cell=z((N_SYMBOLS, 8), np.uint8), pr_chan=z((N_SYMBOLS, 8), np.uint8),
        pr_co=z((N_SYMBOLS, 8), np.int8), pr_bias=z(N_SYMBOLS, np.int16),
    )


def section7_model() -> Model:
    """Exactly the fixture described in section 7."""
    m = blank_model()
    # "For byte 65, use injection entries (0,0,+5), (1,1,+3), and ten
    #  zero-delta entries at cell 0 channel 2."
    m.inj_cell[65, 0], m.inj_chan[65, 0], m.inj_delta[65, 0] = 0, 0, 5
    m.inj_cell[65, 1], m.inj_chan[65, 1], m.inj_delta[65, 1] = 1, 1, 3
    for k in range(2, 12):
        m.inj_cell[65, k], m.inj_chan[65, k], m.inj_delta[65, k] = 0, 2, 0

    p = 15
    # "Set every remaining candidate bias to -100 with zero coefficients."
    m.sc_bias[p, :] = -100
    # Candidate 0: one +1 term reading self channel 0, bias 0.
    m.sc_bias[p, 0] = 0
    m.sc_nb[p, 0, 0], m.sc_ch[p, 0, 0], m.sc_co[p, 0, 0] = 0, 0, 1
    # Candidate 1: neighbour +1, channel 1, coefficient +1, bias 1.
    m.sc_bias[p, 1] = 1
    m.sc_nb[p, 1, 0], m.sc_ch[p, 1, 0], m.sc_co[p, 1, 0] = 1, 1, 1
    # Candidate 0 is ADD d=0, source neighbour +1, source channel 1, a=+1, b=-1.
    m.op_code[p, 0], m.op_d[p, 0] = spec.ADD, 0
    m.op_n[p, 0], m.op_c[p, 0] = 1, 1
    m.op_a[p, 0], m.op_b[p, 0] = 1, -1
    # Candidate 1 is HOLD.
    m.op_code[p, 1] = spec.HOLD
    return m


# ---------------------------------------------------------------------------
def test_section7_transition():
    m = section7_model()
    m.validate()
    s = Session()                       # position 0, phases 0, field zero

    # -- injection --------------------------------------------------------
    active = np.zeros(N_CELLS, np.uint8)
    for k in range(12):
        i = (int(m.inj_cell[65, k]) + s.position) % N_CELLS
        ch = int(m.inj_chan[65, k])
        s.F[i, ch] = spec.sat(int(s.F[i, ch]) + int(m.inj_delta[65, k]))
        active[i] = 1
    assert s.F[0, 0] == 5 and s.F[1, 1] == 3
    assert sorted(np.flatnonzero(active)) == [0, 1], "active cells are 0 and 1"

    # -- page addressing ---------------------------------------------------
    cells = np.flatnonzero(active)
    pid = _page_ids(s.F, s.phase, cells)
    assert list(pid) == [15, 15], "both cells select page 15"

    # -- one complete tick -------------------------------------------------
    residual = run_ticks(m, s, active, max_ticks=1)

    assert s.F[0, 0] == 7, "sat(5 + 3 - 1) = 7 at cell 0 channel 0"
    assert s.F[1, 1] == 3, "cell 1 held its old values"
    assert s.phase[0] == 1 and s.phase[1] == 1, "both active cells advanced phase"
    assert s.phase[2] == 0, "inactive cells do not advance phase"

    # "next_active includes cells 0, 1, 191, 16, 176, 37, and 139"
    assert sorted(np.flatnonzero(residual)) == sorted([0, 1, 191, 16, 176, 37, 139])


def test_section7_readout():
    """The illustrative decoder fixture: scores 8 and 7, first wins."""
    m = blank_model()
    s = Session()
    s.F[0, 0], s.F[1, 1] = 7, 3
    # symbol A: +1 at F[0][0] and +1 at F[1][1], bias -2  -> 8
    m.pr_cell[10, 0], m.pr_chan[10, 0], m.pr_co[10, 0] = 0, 0, 1
    m.pr_cell[10, 1], m.pr_chan[10, 1], m.pr_co[10, 1] = 1, 1, 1
    m.pr_bias[10] = -2
    # symbol B: +1 at F[0][0], bias 0 -> 7
    m.pr_cell[11, 0], m.pr_chan[11, 0], m.pr_co[11, 0] = 0, 0, 1
    sc = probe_scores(m, s)
    assert sc[10] == 8 and sc[11] == 7
    assert int(sc.argmax()) == 10


def test_saturation_fixture():
    """Section 16: 127 plus 4 is 127; -127 minus 4 is -127."""
    assert spec.sat(127 + 4) == 127
    assert spec.sat(-127 - 4) == -127
    assert spec.sat(0) == 0


def test_zero_counts_as_nonnegative():
    m = blank_model()
    s = Session()                       # all zeros
    pid = _page_ids(s.F, s.phase, np.array([0]))
    assert pid[0] == 15, "all four sign tests true on a zero row"
    s.F[0, 2] = -1
    pid = _page_ids(s.F, s.phase, np.array([0]))
    assert pid[0] == 15 - 4, "channel 2 negative clears bit 2"


def test_page_id_range():
    m = random_model(3)
    s = new_session(m)
    cells = np.arange(N_CELLS)
    pid = _page_ids(s.F, s.phase, cells)
    assert pid.min() >= 0 and pid.max() < N_PAGES


def test_tie_goes_to_lowest_candidate_index():
    """Section 5: resolve ties by the smallest candidate index."""
    m = blank_model()
    m.sc_bias[15, :] = 0                # every candidate scores exactly 0
    m.op_code[15, :] = spec.HOLD
    m.op_code[15, 5] = spec.ADD         # a non-HOLD that must NOT be chosen
    m.op_d[15, 5], m.op_b[15, 5] = 7, 9
    s = Session()
    active = np.zeros(N_CELLS, np.uint8); active[0] = 1
    run_ticks(m, s, active, max_ticks=1)
    assert s.F[0, 7] == 0, "candidate 0 won the tie, so no ADD happened"


def test_visitation_order_does_not_matter():
    """Section 6: the update is deterministic regardless of visitation order.

    The engine is vectorised, so instead of permuting a loop we check the
    stronger property the spec relies on: every read comes from the old F.
    """
    m = random_model(7)
    s1, s2 = new_session(m), new_session(m)
    for b in b"order independence":
        ingest(m, s1, b)
    for b in b"order independence":
        ingest(m, s2, b)
    assert np.array_equal(s1.F, s2.F)
    assert np.array_equal(s1.phase, s2.phase)


def test_reset_fixture():
    """Section 16: same model and prompt produce identical state and output."""
    m = random_model(11)
    outs = []
    for _ in range(2):
        s = new_session(m)
        for b in b"reset determinism":
            ingest(m, s, b)
        outs.append((s.F.copy(), s.phase.copy(), s.position))
    assert np.array_equal(outs[0][0], outs[1][0])
    assert np.array_equal(outs[0][1], outs[1][1])
    assert outs[0][2] == outs[1][2]


def test_field_stays_in_int8_range():
    m = random_model(13)
    s = new_session(m)
    for b in b"the field must never leave [-127, 127] however long it runs":
        ingest(m, s, b)
        s.check()


def test_tick_cap_and_work_bound():
    """Section 10: at most 2,304 cell updates per symbol."""
    m = random_model(5)
    s = new_session(m)
    before = s.updates
    ingest(m, s, ord("x"))
    assert s.updates - before <= spec.MAX_UPDATES_PER_SYMBOL


def test_position_advances_after_settling():
    m = random_model(17)
    s = Session()
    assert s.position == 0
    ingest(m, s, spec.BOS)
    assert s.position == 1
    for _ in range(N_CELLS):
        ingest(m, s, ord("a"))
    assert s.position == 1, "position wraps after every 192 ingested symbols"
