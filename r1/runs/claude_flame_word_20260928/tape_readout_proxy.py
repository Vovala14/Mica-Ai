#!/usr/bin/env python3
"""Tape-readout proxy for Flame-W.

Flame-W's readout has 48 tape probes: they read word codes (16 channels per
word) at lags 1:16, 2:8, 3:4, 4:4, 8:4, 16:4, 32:4, 64:4, taking channels
0..n-1 at each lag, as tape_init_ext lays them out. fit_tape_readout fits the
codes, probe coefficients and biases to the data before the rules are built.
This proxy fits exactly that additive model in float on word tokens (no rules,
no work probes, no integer rounding) to compare code starts, and reports what
the tape part alone predicts.

    python tape_readout_proxy.py --init random|ppmi --out F.json

Training: the first --records records of the mix A train shards. Evaluation:
bits per token on the dev token files, and next-word top-1 at the letter
metric's fixed dev positions (word_eval.letter_positions).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import word_eval as E  # noqa: E402
from word_eval import W  # noqa: E402

V = W.N_SYMBOLS
ZERO = V                      # an empty cell (before the record start)
LAGS = [1, 2, 3, 4, 8, 16, 32, 64]
NPROBE = [16, 8, 4, 4, 4, 4, 4, 4]
SHARDS = Path("/mnt/user-data/uploads/PycharmProjects/mica/r1/data/word/v02a")
DATA = Path("/home/claude/mica/wordw/data/word")
DEV = Path("/home/claude/mica/baselines/data/dev")


def read_records(n: int) -> list[np.ndarray]:
    out = []
    for p in sorted(SHARDS.glob("train.part*.u16.gz")):
        data = gzip.open(p, "rb").read()
        i = 0
        while i < len(data) and len(out) < n:
            k = struct.unpack_from("<H", data, i)[0]
            out.append(np.frombuffer(data, "<u2", k // 2, i + 2).astype(np.int64))
            i += 2 + k
        if len(out) >= n:
            break
    return out


def positions(recs: list) -> tuple[np.ndarray, np.ndarray]:
    """Lag ids [N, 8] and targets [N] for every target of every record
    (tokens then EOS), with BOS before the record and ZERO before that."""
    lens = np.array([len(r) for r in recs])
    seq = np.concatenate([np.concatenate([[W.BOS], r]) for r in recs])
    start = np.concatenate([[0], np.cumsum(lens + 1)[:-1]])
    rec = np.repeat(np.arange(len(recs)), lens + 1)
    t = np.arange(len(rec)) - np.repeat(start, lens + 1)       # target index in record
    tgt = np.where(t < lens[rec], seq[np.minimum(start[rec] + t + 1, len(seq) - 1)], W.EOS)
    lag_ids = np.empty((len(rec), len(LAGS)), np.int64)
    for j, L in enumerate(LAGS):
        k = t + 1 - L
        lag_ids[:, j] = np.where(k >= 0, seq[start[rec] + np.maximum(k, 0)], ZERO)
    return lag_ids, tgt


class TapeReadout(torch.nn.Module):
    def __init__(self, codes0: np.ndarray, unigram: np.ndarray):
        super().__init__()
        self.codes = torch.nn.Parameter(torch.tensor(codes0, dtype=torch.float32))
        self.w = torch.nn.Parameter(torch.zeros(sum(NPROBE), V))
        self.b = torch.nn.Parameter(torch.tensor(np.log(unigram), dtype=torch.float32))
        mask = torch.zeros(V)
        mask[W.BOS] = -1e9
        self.register_buffer("mask", mask)

    def features(self, lag_ids):
        full = torch.cat([self.codes, torch.zeros(1, self.codes.shape[1])])
        return torch.cat([full[lag_ids[:, j], :n] for j, n in enumerate(NPROBE)], 1)

    def forward(self, lag_ids):
        return self.b + self.features(lag_ids) @ self.w + self.mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["random", "ppmi"], required=True)
    ap.add_argument("--records", type=int, default=300_000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    t0 = time.time()
    recs = read_records(a.records)
    X, Y = positions(recs)
    uni = np.bincount(Y, minlength=V).astype(np.float64) + 0.5
    uni /= uni.sum()
    g = np.random.default_rng(1)
    if a.init == "random":                       # tape_init_ext: random sign x 24..64
        codes0 = np.where(g.random((V, 16)) < 0.5, -1, 1) * (24 + 40 * g.random((V, 16)))
    else:
        codes0 = np.load("/home/claude/mica/wordw/results/ppmi_codes16_balanced_v2.npz")["codes"]
    codes0 = codes0.astype(np.float32) / 64.0
    model = TapeReadout(codes0, uni)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    print(f"[proxy] {len(recs)} records, {len(Y):,} positions ({time.time() - t0:.0f}s)", flush=True)
    for step in range(a.steps):
        i = torch.randint(0, len(Yt), (a.batch,))
        loss = torch.nn.functional.cross_entropy(model(Xt[i]), Yt[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 250 == 0:
            print(f"  step {step + 1} train {loss.item() / math.log(2):.3f} bits ({time.time() - t0:.0f}s)",
                  flush=True)
    res = {"init": a.init, "records": len(recs), "positions": int(len(Y)), "steps": a.steps,
           "batch": a.batch, "lr": a.lr}
    vocab = W.Vocab.load(DATA / "vocab.json")
    cand = torch.tensor(E.word_ids(vocab))
    with torch.no_grad():
        for name in ("chat_dev1000", "everyday_dev_fresh1000"):
            x, y = positions(E.token_records(DATA / "eval" / f"{name}.jsonl"))
            lp = torch.log_softmax(model(torch.tensor(x)), -1)
            res[f"bits_{name}"] = float(-lp[torch.arange(len(y)), torch.tensor(y)].mean()) / math.log(2)
            pos, brecs = E.letter_positions(DEV / f"{name}.jsonl")
            ctx = [E.prefix_ids(vocab, brecs[i][:off]) for i, off, _ in pos]
            xs = np.stack([positions([np.array(c + [0], np.int64)])[0][len(c)] for c in ctx])
            top = cand[model(torch.tensor(xs))[:, cand].argmax(-1)]
            words = [vocab.tokens[int(j) - W.FIRST_WORD] for j in top]
            res[f"top1_{name}"] = float(np.mean([w == p[2] for w, p in zip(words, pos)]))
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
