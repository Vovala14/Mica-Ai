"""A reserved route must survive tape fitting and the integer file."""

import os
import subprocess
import sys
from pathlib import Path


def test_word_separator_routes_are_disjoint_after_fit_and_export(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update({
        "MICA_CELLS": "64", "MICA_CHANNELS": "16", "MICA_CANDIDATES": "4",
        "MICA_INJECT": "8", "MICA_PROBE": "40", "MICA_TICKS": "4",
        "MICA_PHASES": "4", "MICA_TAPE": "8", "MICA_WINDOW": "1",
        "MICA_PROBE_WINDOW": "16", "MICA_PROBE_COEF": "int8",
        "MICA_ROLLING_READOUT": "1", "MICA_VSET": "2",
        "MICA_ROUTING_CHANNELS": "0,1", "MICA_OFFSETS": "-1,-2",
    })
    script = r"""
import sys
import torch
from mica_r1 import fit, serialize, spec
from mica_r1.discretise import to_integer
from mica_r1.soft import SoftMica
from train_soft import tape_init_ext, fit_tape_readout

torch.set_num_threads(1)
model = SoftMica(ticks=4, hard=True, sel_init=0.05, sel_tau=1)
lags = tape_init_ext(model, 0.05, work_probes=8)
fit_tape_readout(model, [b'the dog ran home.', b'please bring the book.',
                         b'a cat sat nearby.', b'good morning to you.'],
                 lags, torch.device('cpu'), steps=2, batch=16,
                 reserve_word_route=True, log=lambda _: None)

def patterns(code):
    rc = spec.ROUTING_CHANNELS
    offset = spec.ROUTING_PAIR_OFFSET
    return [sum(int(code[b, c] >= code[b, c + offset]) << i
                for i, c in enumerate(rc)) for b in range(spec.N_SYMBOLS)]

p = patterns(model.inj_delta.detach().round())
word = {p[b] for b in fit.WORD_BYTES}
separator = {p[b] for b in range(256) if b not in fit.WORD_BYTES}
separator.update({p[spec.BOS], p[spec.EOS]})
assert word.isdisjoint(separator)
assert p[ord('a')] != p[ord(' ')]

fit.structured_rules(model, 0.05, max_back=(1, 1, 1, 1))
info = fit.word_state_rules(model, 0.05, reset_on_nonword=True)
assert info == {'word_pages': 2, 'reset_pages': 2}
serialize.save(to_integer(model), sys.argv[1])
saved = serialize.load(sys.argv[1])
assert patterns(saved.inj_delta) == p
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "word.mica")],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
