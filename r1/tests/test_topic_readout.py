"""Readout-only topic bias: a third probe over the simulated register.

The automaton file and the rule wiring stay frozen. With no readout bound,
scores are the probe scores alone. Tiny Theory-of-Mind rows are not a fit set.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _topic_cases as C  # noqa: E402


def run(env: dict, case: str) -> str:
    e = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    e.update(env)
    r = subprocess.run([sys.executable, str(HERE / "_topic_readout_cases.py"), case],
                       env=e, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip().splitlines()[-1]


def test_sidecar_does_not_touch_the_automaton():
    assert run(C.BASE, "sidecar") == "ok"


def test_topic_probes_read_the_simulated_register():
    assert run(C.TOPIC, "register_matches_field") == "ok"


def test_split_is_tape_or_work_when_topic_is_off():
    from mica_r1 import spec
    from mica_r1.fit import split_probe_layout
    assert spec.TOPIC_CHANNELS == 0
    layout = [(1, 0), (2, 3), (4, spec.N_CHANNELS - 1)]
    tape, topic, work = split_probe_layout(layout)
    assert topic == []
    assert tape == [(p, lag, ch) for p, (lag, ch) in enumerate(layout)
                    if ch < spec.TAPE_CHANNELS]
    assert work == [(p, lag, ch) for p, (lag, ch) in enumerate(layout)
                    if ch >= spec.TAPE_CHANNELS]


def test_fit_topic_bias_recovers_a_planted_register():
    """Frozen scores plus an int8 word×topic table. Not Tiny ToM."""
    import torch
    from mica_r1.fit import fit_topic_bias
    torch.manual_seed(0)
    n, vocab, width = 80, 6, 2
    kind = np.arange(n) % 2
    registers = np.zeros((n, width), np.int64)
    registers[kind == 0, 0] = 40
    registers[kind == 1, 1] = 40
    targets = np.where(kind == 0, 2, 3).astype(np.int64)
    base = np.zeros((n, vocab), np.int64)
    out = fit_topic_bias(registers, base, targets, steps=400, lr=0.5,
                         l2=1e-6, divisor=1024, seed=0, log=lambda *_a, **_k: None)
    W = out["W"]
    assert W.dtype == np.int8 and W.shape == (vocab, width)
    assert out["heldout_bits_best"] < out["heldout_bits_start"]
    assert int(W[2, 0]) > 0 and int(W[3, 1]) > 0
    assert int(W[2, 0]) > int(W[3, 0])
    bonus = registers @ W.astype(np.int64).T
    # integer replay of one row: channel 0 is live, so word 2 is ahead of word 3
    assert bonus[0, 2] > bonus[0, 3]
    assert bonus[1, 3] > bonus[1, 2]


def test_unbound_probe_scores_skip_the_sidecar():
    """Importing the engine with MICA_TOPIC unset leaves scores unchanged."""
    from mica_r1 import engine, spec
    assert spec.TOPIC_CHANNELS == 0
    m = engine.random_model(1)
    m.validate()
    s = engine.new_session(m)
    a = engine.probe_scores(m, s)
    assert engine.topic_bonus(m, s) is None
    assert a.shape == (spec.N_SYMBOLS,)
