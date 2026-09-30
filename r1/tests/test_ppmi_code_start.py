"""FLAMEW_PPMI_CODES is off unless that env var names a code table."""

import os
import subprocess
import sys
from pathlib import Path


def test_ppmi_code_start_is_a_no_op_until_the_env_is_set(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.pop("FLAMEW_PPMI_CODES", None)
    env.update({
        "MICA_CELLS": "32",
        "MICA_CHANNELS": "8",
        "MICA_CANDIDATES": "1",
        "MICA_INJECT": "4",
        "MICA_PROBE": "4",
        "MICA_TICKS": "1",
        "MICA_PHASES": "1",
        "MICA_TAPE": "2",
        "MICA_WINDOW": "1",
        "MICA_PROBE_WINDOW": "8",
        "MICA_PROBE_COEF": "int8",
        "MICA_ROLLING_READOUT": "1",
        "MICA_ROUTING_CHANNELS": "0,1",
        "MICA_OFFSETS": "-1",
    })
    script = r"""
import os, sys
import numpy as np
import torch
from train_soft import apply_optional_ppmi_codes, tape_init_ext
from mica_r1.soft import SoftMica
from mica_r1 import spec

os.environ.pop("FLAMEW_PPMI_CODES", None)
sm = SoftMica(ticks=1, hard=True, sel_init=0.05, sel_tau=1.0)
tape_init_ext(sm, 0.05, work_probes=0, tape_lags="1:4")
before = sm.inj_delta.detach().clone()
assert apply_optional_ppmi_codes(sm) == ""
assert torch.equal(sm.inj_delta, before)

k = int(spec.TAPE_CHANNELS)
codes = np.zeros((int(spec.N_SYMBOLS), k), np.int8)
codes[:, 0] = 40
codes[3, 1] = -20
path = sys.argv[1]
np.savez(path, codes=codes)
os.environ["FLAMEW_PPMI_CODES"] = path
assert apply_optional_ppmi_codes(sm) == path
got = sm.inj_delta.detach().cpu().numpy()
assert np.array_equal(np.rint(got[:, :k]).astype(np.int8), codes)
assert np.allclose(got[:, k:], before[:, k:].cpu().numpy())

bad = np.zeros((int(spec.N_SYMBOLS), k + 1), np.int8)
bad_path = sys.argv[2]
np.savez(bad_path, codes=bad)
os.environ["FLAMEW_PPMI_CODES"] = bad_path
try:
    apply_optional_ppmi_codes(sm)
except SystemExit:
    pass
else:
    raise AssertionError("wrong shape was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script,
         str(tmp_path / "codes.npz"), str(tmp_path / "bad.npz")],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
