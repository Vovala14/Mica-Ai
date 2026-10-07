"""Cases for test_topic_readout.py. Each runs in its own process because the
geometry is fixed when mica_r1.spec is imported.

    python _topic_readout_cases.py CASE
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _topic_cases as C  # noqa: E402


def _frozen(model):
    from mica_r1 import serialize
    return serialize.dumps(model), {name: getattr(model, name).copy()
                                    for name in ("sc_nb", "sc_ch", "sc_co",
                                                 "op_b", "op_d", "pr_cell",
                                                 "pr_chan", "pr_co", "pr_bias",
                                                 "inj_delta")}


def _still_frozen(model, blob, fields):
    from mica_r1 import serialize
    assert serialize.dumps(model) == blob, "the automaton file changed"
    for name, before in fields.items():
        assert np.array_equal(getattr(model, name), before), name


def case_sidecar() -> None:
    """Readout-only bias: file and rule wiring stay put; scores move by the
    integer dot product; the register matches topic_step, including BOS."""
    from mica_r1 import engine, spec
    m = C.structured(4)
    m.validate()
    blob, fields = _frozen(m)
    r = np.random.default_rng(7)
    T = 4
    codes = r.integers(-20, 21, (spec.N_SYMBOLS, T)).astype(np.int8)
    codes[spec.BOS] = np.array([3, -5, 0, 8], np.int8)
    weights = r.integers(-12, 13, (spec.N_SYMBOLS, T)).astype(np.int8)
    # a known row so the scale check does not depend on the draw
    weights[4] = np.array([8, -3, 0, 5], np.int8)
    engine.bind_topic_readout(m, codes, weights, shift=2, mul=5, rshift=2)
    _still_frozen(m, blob, fields)
    s = engine.new_session(m)
    assert s.topic is not None
    want = engine.topic_register_rows(codes, [], shift=2)[0]
    assert np.array_equal(s.topic, want)
    seq = [1, 4, 9, 4, 2, spec.BOS % 50]
    got_rows = [s.topic.copy()]
    for sym in seq:
        plain_model = m.copy()
        # copy keeps the readout; drop it to read the frozen scores
        plain_model.topic_w = None
        plain_model.topic_codes = None
        base = engine.probe_scores(plain_model, s)
        bonus = engine.topic_bonus(m, s)
        full = engine.probe_scores(m, s)
        assert np.array_equal(full, base + bonus)
        dot = weights.astype(np.int64) @ s.topic.astype(np.int64)
        assert np.array_equal(bonus, ((dot * 5) >> 2).astype(np.int32))
        engine.ingest(m, s, int(sym))
        got_rows.append(s.topic.copy())
    rows = engine.topic_register_rows(codes, seq, shift=2)
    assert np.array_equal(np.stack(got_rows), rows)
    # zero weights add nothing, including on the fast shared-probe path
    z = m.copy()
    z.topic_w = np.zeros_like(z.topic_w)
    sz = engine.new_session(z)
    bare = _drop(m)
    for sym in seq:
        # same field; zero weights must not move a single score
        assert np.array_equal(engine.probe_scores(z, sz), engine.probe_scores(bare, sz))
        engine.ingest(z, sz, int(sym))
    # fork keeps the register
    snap = s.snapshot()
    fork = engine.Session()
    fork.restore(snap)
    assert np.array_equal(fork.topic, s.topic)
    engine.ingest(m, fork, 6)
    assert not np.array_equal(fork.topic, s.topic)
    fork.restore(snap)
    assert np.array_equal(fork.topic, s.topic)
    _still_frozen(m, blob, fields)
    # round trip of the sidecar does not enter the mica bytes
    path = HERE / "_topic_readout_tmp.npz"
    try:
        engine.save_topic_readout(path, m)
        fresh = C.structured(4)
        fresh.validate()
        from mica_r1 import serialize
        assert serialize.dumps(fresh) == blob
        engine.load_topic_readout(path, fresh)
        assert serialize.dumps(fresh) == blob
        assert np.array_equal(fresh.topic_w, m.topic_w)
    finally:
        path.unlink(missing_ok=True)
    print("ok")


def _drop(model):
    bare = model.copy()
    bare.topic_w = None
    bare.topic_codes = None
    return bare


def case_register_matches_field() -> None:
    """Probes aimed at the register read the simulated value. Rules are not
    rewired (no add_topic_register)."""
    import torch
    from mica_r1 import engine, fit, spec
    m = C.structured(9)
    T, TOP, K = spec.TOPIC_CHANNELS, spec.TOPIC_AT, spec.TAPE_CHANNELS
    rules = m.sc_ch.copy(), m.sc_nb.copy(), m.sc_co.copy()
    r = np.random.default_rng(11)
    codes = r.integers(-15, 16, (spec.N_SYMBOLS, T)).astype(np.int8)
    codes[:8] = 0
    codes[spec.BOS] = np.array([4, -6, 2], np.int8)[:T]
    m.inj_delta[:, K:K + T] = codes
    # two probes read the register at different lags; the rest stay work/tape
    probes = [(0, 1, TOP), (1, 2, TOP + 1), (2, 3, TOP + (T - 1))]
    for p, lag, ch in probes:
        m.pr_cell[:, p] = (spec.N_CELLS - lag) % spec.N_CELLS
        m.pr_chan[:, p] = ch
        m.pr_co[:, p] = 1
    m.validate()
    assert np.array_equal(m.sc_ch, rules[0]) and np.array_equal(m.sc_nb, rules[1])
    assert np.array_equal(m.sc_co, rules[2])
    seq = [10, 11, 12, 10, 40, 41, 12, 8]
    rec = np.asarray(seq, np.uint16)
    feat = engine.topic_probe_features([rec], codes, probes, spec.TOPIC_SHIFT)
    _ctx, tgt = fit.contexts([rec], [1], torch.device("cpu"))
    assert feat.shape == (int(tgt.shape[0]), len(probes))
    tape, topic, work = fit.split_probe_layout(
        [(lag, ch) for _p, lag, ch in probes] + [(1, K)])
    assert [p for p, _, _ in topic] == [0, 1, 2]
    assert [p for p, _, _ in work] == [3]
    assert tape == []
    s = engine.new_session(m)
    got = []
    pending = list(seq) + [None]          # None = the EOS target, already ingested
    for sym in pending:
        row = []
        for _p, lag, ch in probes:
            cell = (s.position - lag) % spec.N_CELLS
            row.append(int(s.F[cell, ch]))
        got.append(row)
        if sym is not None:
            engine.ingest(m, s, int(sym))
    assert np.array_equal(np.asarray(got, np.int32), feat), (got, feat)
    # the same codes, bound as a sidecar, track the head cell's register
    engine.bind_topic_readout(m, codes, np.zeros_like(codes), shift=spec.TOPIC_SHIFT)
    s = engine.new_session(m)
    assert np.array_equal(s.topic, s.F[(s.position - 1) % spec.N_CELLS, TOP:])
    for sym in seq:
        engine.ingest(m, s, int(sym))
        assert np.array_equal(s.topic, s.F[(s.position - 1) % spec.N_CELLS, TOP:])
    # zero sidecar does not change scores relative to the probes alone
    bare = _drop(m)
    sb = engine.new_session(bare)
    ss = engine.new_session(m)
    assert np.array_equal(engine.probe_scores(m, ss), engine.probe_scores(bare, sb))
    print("ok")


if __name__ == "__main__":
    name = sys.argv[1]
    globals()["case_" + name]()
