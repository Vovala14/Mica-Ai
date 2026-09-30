#!/usr/bin/env python3
"""Byte n-gram baselines on MICA's exact evaluation protocol.

Records are trimmed to 256 bytes like MICA's; each record is predicted from
BOS: every byte, then EOS (257 possible targets). Contexts before the start
of a record are BOS. Scores are pooled bits per target, like
r1/checks/eval_int.py, on the same record sets (checks/common.py).

  order 1  byte frequencies (maximum likelihood, +0.5 per symbol)
  order 2  previous-byte context       } interpolated Kneser-Ney with
  order 3  two-byte context            } discounts D = n1 / (n1 + 2 n2)
  order 5  four-byte context           } per order (Chen & Goodman)

Tables are fitted on the TRAIN split only (a uniform random sample of
records, seeded). Sizes are reported as stored entries: distinct (context,
next symbol) pairs over all orders a model uses -- these tables are not
size-matched to MICA.

    python r1/checks/ngram_baselines.py --sets val1000,test2000 \
        --max-train-records 200000 --out r1/runs/checks/ngram.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import utf8_stdout, R1, record_set, beat

BOS_C, EOS_T, V = 256, 256, 257      # context-only BOS; target index of EOS
BITS = 9                             # bits per symbol in a packed key


def positions(recs, max_ctx):
    """(contexts (N, max_ctx) nearest-first, targets (N,)) for all targets."""
    ctxs, tgts = [], []
    for r in recs:
        a = np.frombuffer(r, np.uint8).astype(np.int64)
        seq = np.concatenate([np.full(max_ctx, BOS_C), a])
        n = len(a) + 1                                    # bytes, then EOS
        idx = np.arange(n)[:, None] + max_ctx - 1 - np.arange(max_ctx)[None, :]
        ctxs.append(seq[idx].astype(np.int16))
        tgts.append(np.concatenate([a, [EOS_T]]))
    return np.concatenate(ctxs), np.concatenate(tgts)


def pack(ctx, tgt, m):
    """Key of (the m nearest context symbols, target)."""
    if not 0 <= m <= 6:
        raise ValueError("packed n-gram keys support context lengths 0..6")
    k = np.zeros(len(tgt), np.int64)
    for j in range(m - 1, -1, -1):                        # farthest first
        k = (k << BITS) | ctx[:, j].astype(np.int64)
    return (k << BITS) | tgt


class KN:
    """Interpolated Kneser-Ney over contexts of length 0..M."""

    def __init__(self, ctx, tgt, M):
        self.M = M
        key = pack(ctx, tgt, M)
        uk, cnt = np.unique(key, return_counts=True)
        self.tables = {}
        for m in range(M, -1, -1):
            if m < M:
                # continuation counts: distinct left extensions of (h, w)
                uk, cnt = np.unique(uk & ((1 << (BITS * (m + 1))) - 1),
                                    return_counts=True)
            n1, n2 = int((cnt == 1).sum()), int((cnt == 2).sum())
            # The count-of-counts estimate can be zero on small or dense
            # corpora. Keep a backoff mass so unseen targets remain scoreable.
            D = n1 / (n1 + 2 * n2) if n1 else 0.5
            hk = uk >> BITS
            uh, inv = np.unique(hk, return_inverse=True)
            ch = np.bincount(inv, weights=cnt)
            nh = np.bincount(inv).astype(np.float64)
            self.tables[m] = (uk, cnt.astype(np.float64), uh, ch, nh, D)
        self.entries = int(sum(len(t[0]) for t in self.tables.values()))

    def prob(self, ctx, tgt):
        p = np.full(len(tgt), 1.0 / V)
        for m in range(0, self.M + 1):
            uk, cnt, uh, ch, nh, D = self.tables[m]
            key = pack(ctx, tgt, m)
            hk = key >> BITS
            i = np.clip(np.searchsorted(uh, hk), 0, len(uh) - 1)
            found = uh[i] == hk
            j = np.clip(np.searchsorted(uk, key), 0, len(uk) - 1)
            c = np.where(uk[j] == key, cnt[j], 0.0)
            c_h, n_h = ch[i], nh[i]
            p = np.where(found, (np.maximum(c - D, 0.0) + D * n_h * p) /
                         np.where(found, c_h, 1.0), p)
        return p


def unigram(tgt):
    c = np.bincount(tgt, minlength=V).astype(np.float64) + 0.5
    return c / c.sum()


def main() -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(R1 / "data/bulk/train.jsonl"))
    ap.add_argument("--max-train-records", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orders", default="1,2,3,5")
    ap.add_argument("--sets", default="val1000")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--val-file", default=None,
                    help="override the validation JSONL (for another corpus)")
    ap.add_argument("--test-file", default=None,
                    help="override the test JSONL")
    ap.add_argument("--val-skip", type=int, default=64,
                    help="records reserved for checkpoint selection (default 64)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    orders = [int(x) for x in a.orders.split(",")]
    if not orders or any(o < 1 or o > 7 for o in orders):
        ap.error("orders must be between 1 and 7 (int64 packed-key limit)")
    from make_records import load_records
    t0 = time.time()
    beat("ngram: loading")
    train = load_records(a.train)
    rng = np.random.default_rng(a.seed)
    if len(train) > a.max_train_records:
        pick = np.sort(rng.choice(len(train), a.max_train_records, replace=False))
        train = [train[i] for i in pick]
    train = [r[:256] for r in train if len(r)]
    M = max(orders) - 1
    ctx, tgt = positions(train, max(M, 1))
    print(f"[ngram] {len(train):,} train records, {len(tgt):,} targets "
          f"({time.time() - t0:.0f}s)", flush=True)
    models = {}
    for o in orders:
        beat(f"ngram: order {o}")
        if o == 1:
            models[o] = ("frequencies", unigram(tgt), V)
        else:
            km = KN(ctx, tgt, o - 1)
            models[o] = ("kneser-ney", km, km.entries)
        print(f"[ngram] order {o} fitted ({models[o][2]:,} entries, "
              f"{time.time() - t0:.0f}s)", flush=True)
    del ctx, tgt
    out = {"train_records": len(train), "train_file": a.train,
           "val_file": a.val_file, "test_file": a.test_file,
           "val_skip": a.val_skip, "seed": a.seed, "results": {}}
    for set_name in a.sets.split(","):
        recs = record_set(set_name, a.data_root, val_file=a.val_file,
                          test_file=a.test_file, val_skip=a.val_skip)
        c, t = positions(recs, max(M, 1))
        for o in orders:
            kind, mdl, entries = models[o]
            p = mdl[t] if kind == "frequencies" else mdl.prob(c, t)
            bits = float(-np.log2(p).sum() / len(t))
            out["results"][f"{set_name}/order{o}"] = {
                "set": set_name, "order": o, "context_bytes": o - 1,
                "kind": kind, "entries": entries, "targets": int(len(t)),
                "bits": bits}
            print(f"[ngram] {set_name:9s} order {o} ({o - 1}-byte context, "
                  f"{kind}): {bits:.6f} bits/target, {entries:,} entries",
                  flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
