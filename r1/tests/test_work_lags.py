"""Long-lag work probes must survive layout and integer export."""

import os
import subprocess
import sys
from pathlib import Path


def test_explicit_work_lags_keep_complete_groups_and_export(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update({
        "MICA_CELLS": "64",
        "MICA_CHANNELS": "112",
        "MICA_PAGES": "512",
        "MICA_CANDIDATES": "2",
        "MICA_INJECT": "16",
        "MICA_PROBE": "240",
        "MICA_TICKS": "16",
        "MICA_PHASES": "16",
        "MICA_TAPE": "16",
        "MICA_WINDOW": "1",
        "MICA_PROBE_WINDOW": "32",
        "MICA_PROBE_COEF": "int8",
        "MICA_ROLLING_READOUT": "1",
        "MICA_VSET": "6",
        "MICA_ROUTING_CHANNELS": "0,1,2,3,4",
        "MICA_OFFSETS": "-1,-2,-3,-4,-6,-8",
    })
    script = r"""
from collections import Counter
from mica_r1 import fit, serialize
from mica_r1.discretise import to_integer
from mica_r1.soft import SoftMica
from train_soft import tape_init_ext, same_run
import sys

for bad in ('1:16,2:8,8:4', '1:16,2:8,8:4,33:4',
            '1:16,1:8,8:4,16:4', '1:16,2:8,8:4,16:5'):
    try:
        fit.structured_work_probes(192, ticks=16, work_lags=bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'accepted {bad}')

layout = fit.structured_work_probes(
    192, ticks=16, work_lags='1:16,2:8,8:4,16:4')
assert Counter(lag for lag, _ in layout) == {1: 96, 2: 48, 8: 24, 16: 24}
for lag in (1, 2, 8, 16):
    channels = {channel for l, channel in layout if l == lag}
    if lag == 1:
        assert channels == set(range(16, 112))
    for group in {((channel - 16) // 6) for channel in channels}:
        assert set(range(16 + group * 6, 16 + (group + 1) * 6)) <= channels

sm = SoftMica(ticks=16, hard=True, sel_init=0.05, sel_tau=1.0)
tape_init_ext(sm, 0.05, work_probes=192, work_layout=layout)
assert fit.probe_layout(sm)[-192:] == layout
serialize.save(to_integer(sm), sys.argv[1])
integer = serialize.load(sys.argv[1])
assert [(64 - int(c)) % 64 for c in integer.pr_cell[0, -192:]] == [l for l, _ in layout]
assert [int(c) for c in integer.pr_chan[0, -192:]] == [c for _, c in layout]
assert not same_run([1, 2], [1, 2, 'work_lags=1:16,2:8,8:4,16:4'])
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "work.mica")],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
