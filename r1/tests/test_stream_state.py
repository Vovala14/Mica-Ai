"""The optional stream state keeps a machine state across word separators."""

import os
import subprocess
import sys
from pathlib import Path


def test_stream_state_does_not_reset_at_spaces(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update({
        "MICA_CELLS": "64", "MICA_CHANNELS": "20",
        "MICA_CANDIDATES": "8", "MICA_INJECT": "16",
        "MICA_PROBE": "4", "MICA_TICKS": "2", "MICA_PHASES": "2",
        "MICA_TAPE": "16", "MICA_VSET": "2", "MICA_WINDOW": "1",
        "MICA_PROBE_WINDOW": "32", "MICA_PROBE_COEF": "int8",
        "MICA_ROLLING_READOUT": "1", "MICA_ROUTING": "pairdiff",
        "MICA_ROUTING_CHANNELS": "0,1,2,3,4,5,6,7",
        "MICA_OFFSETS": "-1,-2,-3,-4,-6,-8",
    })
    script = r"""
import sys
import torch
from train_soft import tape_init_ext
from mica_r1 import fit, serialize, spec
from mica_r1.discretise import to_integer
from mica_r1.soft import SoftMica

model = SoftMica(ticks=2, hard=True, sel_init=.05, sel_tau=1.0)
tape_init_ext(model, .05, work_probes=2)
fit.structured_rules(model, .05, max_back=(1, 2))
info = fit.word_state_rules(model, .05, reset_on_nonword=False)
assert info['reset_pages'] == 0
assert info['word_pages'] == 256
with torch.no_grad():
    # The phase-0 page for a space must have live candidate scores. A frozen
    # reset page would have all but candidate 0 at -30000.
    page = int(fit.byte_patterns(model)[ord(' ')])
    assert (model.sc_bias[page] > -20000).all()
    phase0_b = model.op_b[:spec.PAGE_STRIDE].clone()
    phase0_v = model.op_v[:spec.PAGE_STRIDE].clone()
fit.fit_readout_and_rules(model, [b"the cat sat.", b"a dog ran."],
                          torch.device("cpu"), ticks=2, steps=2, batch=32,
                          max_records=2, check_every=1,
                          freeze_phase0_state=True, word_start_weight=3,
                          log=lambda _: None)
assert torch.equal(model.op_b[:spec.PAGE_STRIDE], phase0_b.round().clamp(-127, 127))
assert torch.equal(model.op_v[:spec.PAGE_STRIDE], phase0_v.round().clamp(-127, 127))
serialize.save(to_integer(model), sys.argv[1])
serialize.load(sys.argv[1])
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "stream.mica")],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
