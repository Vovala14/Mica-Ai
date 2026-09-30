"""Focused preflight for explicit tape probe lags and the exported model."""

import os
import subprocess
import sys
from pathlib import Path


def test_explicit_tape_lags_survive_export_and_read_old_context(tmp_path):
    # spec binds its geometry at import, so keep this small experiment in its
    # own process instead of changing the geometry of the other tests.
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update({
        "MICA_CELLS": "64",
        "MICA_CHANNELS": "8",
        "MICA_CANDIDATES": "1",
        "MICA_INJECT": "4",
        "MICA_PROBE": "4",
        "MICA_TICKS": "1",
        "MICA_PHASES": "1",
        "MICA_TAPE": "2",
        "MICA_WINDOW": "1",
        "MICA_PROBE_WINDOW": "24",
        "MICA_PROBE_COEF": "int8",
        "MICA_ROLLING_READOUT": "1",
        "MICA_ROUTING_CHANNELS": "0,1",
        "MICA_OFFSETS": "-1",
    })
    script = r"""
import sys
import numpy as np
import torch
from train_soft import _parse_tape_lags, same_run, tape_init_ext
from mica_r1 import engine, fit, serialize
from mica_r1.discretise import to_integer
from mica_r1.soft import SoftMica

for bad, count, limit in (("1:3", 4, 24), ("25:4", 4, 24),
                          ("1:0,16:4", 4, 24), ("1:2,16:3", 4, 24)):
    try:
        _parse_tape_lags(bad, count, limit)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted invalid layout {bad}")
assert not same_run([1, 2], [1, 2, "tape_lags=16:1"])

sm = SoftMica(ticks=1, hard=True, sel_init=0.05, sel_tau=1.0)
layout = tape_init_ext(sm, 0.05, work_probes=0,
                       tape_lags="1:2,16:1,24:1")
assert layout == [(1, "tape"), (1, "tape"),
                  (16, "tape"), (24, "tape")]
assert fit.probe_layout(sm) == [(1, 0), (1, 1), (16, 0), (24, 0)]
with torch.no_grad():
    sm.inj_delta[ord("A"), 0] = 50
    sm.inj_delta[ord("Z"), 0] = -50
    sm.pr_w.zero_()
    sm.pr_w[ord("x"), 2] = 10  # one exported probe reads lag 16

path = sys.argv[1]
serialize.save(to_integer(sm), path)
model = serialize.load(path)
assert int(model.pr_cell[ord("x"), 2]) == 64 - 16
assert int(model.pr_co[ord("x"), 2]) == 10
engine.ROUTING_MODE = "pairdiff"

def score(prompt):
    session = engine.new_session(model)
    for byte in prompt:
        engine.ingest(model, session, byte)
    return engine.probe_scores(model, session)

near_a = score(b"A" + b"s" * 15)
near_z = score(b"Z" + b"s" * 15)
assert near_a[ord("x")] != near_z[ord("x")]
far_a = score(b"A" + b"s" * 24)
far_z = score(b"Z" + b"s" * 24)
assert np.array_equal(far_a, far_z)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "lag.mica")],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
