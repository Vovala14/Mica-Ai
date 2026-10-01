"""Cases for test_topic_register.py. Each runs in its own process, because
the geometry (and the topic register) is fixed when mica_r1.spec is imported.

    python _topic_cases.py CASE [ARGS]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = {"MICA_CELLS": "64", "MICA_CHANNELS": "12", "MICA_PAGES": "0",
        "MICA_CANDIDATES": "8", "MICA_INJECT": "8", "MICA_PROBE": "6",
        "MICA_TICKS": "4", "MICA_PHASES": "4", "MICA_TAPE": "4", "MICA_WINDOW": "1",
        "MICA_VSET": "2", "MICA_ROUTING": "pairdiff", "MICA_ROUTING_CHANNELS": "0,1",
        "MICA_OFFSETS": "-1,-2,-3", "MICA_PROBE_COEF": "int8", "MICA_PROBE_WINDOW": "16",
        "MICA_ROLLING_READOUT": "1", "MICA_LOGIT_DIVISOR": "64", "MICA_SYMBOLS": "300"}
TOPIC = dict(BASE, MICA_CHANNELS="15", MICA_TOPIC="3", MICA_TOPIC_SHIFT="2")


def structured(seed: int):
    """A random VSET rule book under the current geometry (what fit mode
    trains), with the same probe wiring for every symbol."""
    from mica_r1 import engine, spec
    r = np.random.default_rng(seed)
    m = engine.random_model(seed)
    m.op_code[:] = spec.VSET
    K, VW = spec.TAPE_CHANNELS, spec.VSET_WIDTH
    phase = np.arange(spec.N_PAGES) // spec.PAGE_STRIDE
    m.op_d[:] = (K + phase[:, None] * VW).astype(np.uint8)
    m.op_u[:] = K
    m.sc_ch[:] = r.integers(0, K, m.sc_ch.shape)
    m.sc_bias[:] = r.integers(-5, 6, m.sc_bias.shape)
    m.pr_cell[:] = m.pr_cell[:1]
    m.pr_chan[:] = m.pr_chan[:1]
    m.inj_chan[:] = m.inj_chan[:1]          # one wiring for every symbol (compact)
    m.inj_delta[:, :K] = r.integers(-40, 41, (spec.N_SYMBOLS, K))
    m.inj_delta[:, K:] = 0                       # unused work entries, as in Flame-W B
    return m


def case_make_base(out: str) -> None:
    from mica_r1 import serialize
    m = structured(3)
    m.validate()
    serialize.save(m, out)


def case_topic(base_path: str, out: str) -> None:
    """Convert the base model, then check engine == the fitter's simulator,
    the register's arithmetic, and that nothing else changed."""
    import torch
    from mica_r1 import engine, fit, serialize, spec
    T, TOP, K = spec.TOPIC_CHANNELS, spec.TOPIC_AT, spec.TAPE_CHANNELS
    base = serialize.load_other_channels(base_path)
    before = base.copy()
    r = np.random.default_rng(5)
    codes = r.integers(-30, 31, (spec.N_SYMBOLS, T)).astype(np.int8)
    codes[:50] = 0                               # function words
    info = fit.add_topic_register(base, codes, phases=[2, 3], terms=2, seed=1)
    m = base
    assert info["phases"] == [2, 3] and info["content_symbols"] == spec.N_SYMBOLS - 50 - \
        int(((codes[50:] == 0).all(1)).sum())
    # phases 0 and 1 are untouched; 2 and 3 changed only in the last two terms
    S = spec.PAGE_STRIDE
    for f in ("sc_nb", "sc_ch", "sc_co"):
        a, b = getattr(m, f), getattr(before, f)
        assert np.array_equal(a[:2 * S], b[:2 * S]), f
        assert np.array_equal(a[2 * S:, :, :-2], b[2 * S:, :, :-2]), f
    assert (m.sc_ch[2 * S:, :, -2:] >= TOP).all() and (m.sc_nb[2 * S:, :, -2:] == 0).all()
    for f in ("op_b", "op_v", "op_d", "pr_co", "pr_bias", "sc_bias"):
        assert np.array_equal(getattr(m, f), getattr(before, f)), f
    serialize.save(m, out)
    m = serialize.load(out)                      # round trip under the topic geometry

    seq = list(r.integers(0, spec.N_SYMBOLS - 2, 40)) + [spec.BOS]
    s = engine.new_session(m)
    want = np.zeros(T, np.int64)
    want = np.clip(want - np.sign(want) * (np.abs(want) >> spec.TOPIC_SHIFT)
                   + codes[spec.BOS], -127, 127)
    fields = []
    for sym in seq:
        prev_head = s.position
        engine.ingest(m, s, int(sym))
        want = np.clip(want - np.sign(want) * (np.abs(want) >> spec.TOPIC_SHIFT)
                       + codes[sym], -127, 127)
        assert np.array_equal(s.F[prev_head, TOP:], want), (s.F[prev_head, TOP:], want)
        fields.append(s.F.copy())
    # the register really steers rules: with zero codes the field differs
    z = m.copy()
    z.inj_delta[:, K:K + T] = 0
    sz = engine.new_session(z)
    for sym in seq:
        engine.ingest(z, sz, int(sym))
    assert not np.array_equal(sz.F[:, :TOP], s.F[:, :TOP]), "topic terms never changed a winner"

    # the fitter's simulator runs the same machine
    sm = fit.from_integer(m, spec.MAX_TICKS)
    from mica_r1.discretise import to_integer
    assert serialize.dumps(to_integer(sm)) == serialize.dumps(m)
    got = []

    def on_read(i, part, t, prov, F, pos):
        if t >= 1:
            got.append(F[0].cpu().numpy().copy())
    rec = np.asarray(seq[:-1], np.uint16)
    fit._simulate(sm, [rec], torch.device("cpu"), spec.MAX_TICKS, on_read=on_read)
    assert len(got) == len(seq) - 1 >= 40, len(got)
    changed = sum(not np.array_equal(a, b) for a, b in zip(fields[1:], fields[:-1]))
    assert changed == len(fields) - 1, "the field should move every symbol"
    for t, (a, b) in enumerate(zip(got, fields)):
        assert np.array_equal(a, b), f"simulator differs from the engine after symbol {t}"
    print("ok")


def case_refuse(path: str) -> None:
    """A topic file must not load as a plain model, and vice versa."""
    from mica_r1 import serialize
    try:
        serialize.load(path)
    except serialize.FormatError:
        print("refused")
        return
    raise AssertionError("loaded a file of the other machine")


if __name__ == "__main__":
    name, *args = sys.argv[1:]
    globals()["case_" + name](*args)
