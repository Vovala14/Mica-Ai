"""FLAMEW_FREEZE_TAPE_CODES leaves the named tape channels at their start."""

import os
import subprocess
import sys
from pathlib import Path


def test_freeze_keeps_the_requested_tape_channels(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.pop("FLAMEW_FREEZE_TAPE_CODES", None)
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
        "FLAMEW_FREEZE_TAPE_CODES": "1",
    })
    script = r"""
import os
import numpy as np
import torch
from train_soft import fit_tape_readout, tape_init_ext
from mica_r1.soft import SoftMica

os.environ["FLAMEW_FREEZE_TAPE_CODES"] = "1"
sm = SoftMica(ticks=1, hard=True, sel_init=0.05, sel_tau=1.0)
lags = tape_init_ext(sm, 0.05, work_probes=0, tape_lags="1:4")
with torch.no_grad():
    sm.inj_delta[:, 0] = 40
    sm.inj_delta[:, 1] = -20
before = sm.inj_delta.detach().clone()
recs = [bytes([i % 50, (i * 3) % 50, 7, 8, 9]) for i in range(40)]
fit_tape_readout(sm, recs, lags, torch.device("cpu"), steps=30,
                 batch=8, max_records=40, log=lambda m: None)
got = sm.inj_delta.detach().cpu()
assert torch.equal(got[:, :1].round(), before[:, :1].round())
assert not torch.equal(got[:, 1:2].round(), before[:, 1:2].round())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
