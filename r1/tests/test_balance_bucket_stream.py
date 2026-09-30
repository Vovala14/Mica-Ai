"""Bucket balancing phase by phase gives exactly the biases of the joint
dual ascent over all phases' score rows (the pre-streaming algorithm)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

R1 = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import random, torch
from mica_r1 import spec, fit
from mica_r1.soft import SoftMica
torch.manual_seed(1)
model = SoftMica(ticks=spec.MAX_TICKS, tau=0.5, hard=True, sel_init=0.05, sel_tau=1.0)
fit.structured_rules(model, 0.05, max_back=(1, 2, 3, 4), self_terms=1)
rng = random.Random(0)
words = "the a cat dog sat on mat and ran to see you me we it is was".split()
recs = [" ".join(rng.choice(words) for _ in range(rng.randint(3, 12))).encode()
        for _ in range(400)]
dev = torch.device("cpu")
P, C = spec.N_PAGES, spec.N_CANDIDATES
beta = model.sc_bias.detach().round().int().clone()
eligible = beta > -20000
tie = torch.arange(C, dtype=torch.int32)
max_records = max(500, 3000 * 4 // spec.N_PHASE)
sample = [r for r in recs[::max(1, len(recs) // max_records)] if len(r)]
pages, S = fit._page_score_rows(model, sample, dev)

def winners():
    return ((S + beta[pages]).long() * (C + 1) - tie).argmax(-1)

for it in range(80):                     # the joint ascent, as before
    cnt = torch.bincount(pages * C + winners(), minlength=P * C).reshape(P, C)
    target = cnt.sum(1, keepdim=True).float() / eligible.sum(1, keepdim=True).clamp(min=1)
    step = max(1, 64 >> (it // 10))
    beta -= (step * torch.sign(cnt.float() - target)).int() * \
        eligible.int() * (cnt.sum(1, keepdim=True) > 0).int()
fit.balance_buckets(model, recs, dev, log=lambda m: None)
got = model.sc_bias.detach().round().int()
assert (got != 0).any()
assert torch.equal(got, beta.clamp(-32767, 32767)), int((got != beta).sum())
print("identical")
'''

GEOMETRY = {"MICA_CANDIDATES": "32", "MICA_CELLS": "64", "MICA_CHANNELS": "40",
            "MICA_INJECT": "24", "MICA_LOGIT_DIVISOR": "1024",
            "MICA_OFFSETS": "-1,-2,-3,-4,-6,-8", "MICA_PHASES": "4",
            "MICA_PROBE": "64", "MICA_PROBE_COEF": "int8", "MICA_PROBE_WINDOW": "64",
            "MICA_ROLLING_READOUT": "1", "MICA_ROUTING": "pairdiff",
            "MICA_ROUTING_CHANNELS": "0,1,2,3,4", "MICA_TAPE": "16",
            "MICA_TICKS": "4", "MICA_VSET": "6", "MICA_WINDOW": "1"}


def test_phase_by_phase_balance_matches_joint_ascent():
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update(GEOMETRY)
    env["PYTHONPATH"] = str(R1) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", SCRIPT], env=env, cwd=R1,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "identical" in out.stdout
