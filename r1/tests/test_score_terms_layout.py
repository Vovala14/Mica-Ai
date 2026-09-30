"""Candidate records follow the scoring-term count: six terms keep the
original layout, other counts round-trip exactly (with VSET immediates)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

R1 = Path(__file__).resolve().parents[1]

SCRIPT = """
import numpy as np
from mica_r1 import engine, serialize, spec
m = engine.random_model(3)
m.sc_bias = np.random.default_rng(1).integers(-300, 300, m.sc_bias.shape).astype(np.int16)
m2 = serialize.loads(serialize.dumps(m))
fields = ("inj_cell", "inj_chan", "inj_delta", "sc_nb", "sc_ch", "sc_co", "sc_bias",
          "op_code", "op_d", "op_n", "op_c", "op_a", "op_b", "op_u",
          "pr_cell", "pr_chan", "pr_co", "pr_bias", "op_v")
for f in fields:
    a, b = getattr(m, f), getattr(m2, f)
    assert (a is None) == (b is None), f
    assert a is None or np.array_equal(a, b), f
print(spec.N_SCORE_TERMS, spec.CAND_BIAS_AT, spec.CAND_OPS_AT, spec.CAND_VSET_AT,
      spec.CANDIDATE_BYTES, spec.TOTAL_BYTES)
"""

BASE = {"MICA_CELLS": "64", "MICA_CHANNELS": "64", "MICA_CANDIDATES": "16",
        "MICA_INJECT": "24", "MICA_PROBE": "32", "MICA_TICKS": "8", "MICA_PHASES": "8",
        "MICA_TAPE": "16", "MICA_WINDOW": "1", "MICA_ROLLING_READOUT": "1",
        "MICA_PROBE_COEF": "int8", "MICA_VSET": "6", "MICA_OFFSETS": "-1,-2,-3,-4,-6,-8",
        "MICA_ROUTING": "pairdiff", "MICA_ROUTING_CHANNELS": "0,1,2,3,4",
        "MICA_LOGIT_DIVISOR": "1024", "MICA_PROBE_WINDOW": "32"}


def _run(terms: int) -> list[int]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update(BASE, MICA_SCORE_TERMS=str(terms))
    out = subprocess.run([sys.executable, "-c", SCRIPT], cwd=R1, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    return [int(x) for x in out.stdout.split()]


def test_six_terms_keep_the_original_layout():
    assert _run(6)[1:5] == [18, 20, 27, 32]


def test_twelve_terms_round_trip_with_vset():
    terms, bias_at, ops_at, vset_at, cand_bytes, _ = _run(12)
    assert (terms, bias_at, ops_at, vset_at) == (12, 36, 38, 45)
    assert cand_bytes == 52          # 45 + 5 VSET immediates, padded to 4
