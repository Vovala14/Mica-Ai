#!/usr/bin/env python3
"""Exact integer MICA use(k) on the two disjoint S1 validation probes."""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import R1, run_geometry, worker_env, paired_bootstrap
from run_mica_page_block_refit import digest, write_json


M = 24
KS = (2, 4, 8, 16, 24, 32, 48)
PROBES = {
    "chat_long600": (R1 / "data/ctxprobe/chat_long600.jsonl",
                     "7c23055bac517d4b7c004420efb1b4ed1ab520532f07defe71cb1960226c5158"),
    "everyday_long600": (R1 / "data/ctxprobe/everyday_long600.jsonl",
                         "72f3036297eb7f79778cd511378faa9a6c7e3787624bb1adccdbd0629fa50451"),
}


def paired_records(path: Path) -> list[tuple[bytes, bytes]]:
    rows = [bytes.fromhex(line.strip()) for line in path.open(encoding="ascii")
            if line.strip()]
    rows = [r for r in rows if len(r) >= 64]
    if len(rows) != 600:
        raise RuntimeError("context probe must contain 600 long records")
    rng = random.Random(20260928)
    pairs = []
    for original in rows:
        donor = original
        while donor == original:
            donor = rng.choice(rows)
        pairs.append((original, donor[:M] + original[M:]))
    return pairs


def worker(args) -> None:
    from mica_r1 import batch, serialize, spec

    model = serialize.load(args.run / "best.mica")
    batch.MAX_TICKS = spec.MAX_TICKS
    pairs = paired_records(PROBES[args.probe][0])
    stack = batch.ModelStack([model], score_backend="c")
    nats_true = {k: np.zeros(len(pairs)) for k in KS}
    nats_spliced = {k: np.zeros(len(pairs)) for k in KS}
    counts = {k: np.zeros(len(pairs), dtype=np.int64) for k in KS}
    eligible = np.r_[np.arange(256), spec.EOS]
    for start in range(0, len(pairs), args.chunk):
        part = pairs[start:start + args.chunk]
        records = [r for pair in part for r in pair]
        lengths = np.asarray([len(r) for r in records], dtype=np.int32)
        maxlen = int(lengths.max())
        padded = np.full((len(records), maxlen), -1, dtype=np.int32)
        for i, record in enumerate(records):
            padded[i, :len(record)] = np.frombuffer(record, dtype=np.uint8)
        st = batch.BatchSession(len(records), np.zeros(len(records), dtype=np.int32))
        batch.ingest_batch(stack, st, np.full(len(records), spec.BOS, np.int32))
        local = {k: np.zeros(len(records)) for k in KS}
        local_counts = {k: np.zeros(len(records), dtype=np.int64) for k in KS}
        for position in range(maxlen + 1):
            scores = batch.probe_batch(stack, st).astype(np.float64) / spec.LOGIT_DIVISOR
            logits = scores[:, eligible]
            maximum = logits.max(axis=1)
            lse = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
            target = np.where(position < lengths,
                              padded[:, min(position, maxlen - 1)], spec.EOS)
            loss = lse - scores[np.arange(len(records)), target]
            alive = position <= lengths
            for k in KS:
                keep = alive & (position >= M + k)
                local[k] += np.where(keep, loss, 0.0)
                local_counts[k] += keep
            if position < maxlen:
                feed = np.where(position < lengths, padded[:, position], spec.BOS)
                batch.ingest_batch(stack, st, feed.astype(np.int32))
        ids = np.arange(start, start + len(part))
        for k in KS:
            if not np.array_equal(local_counts[k][0::2], local_counts[k][1::2]):
                raise RuntimeError("paired target counts do not align")
            nats_true[k][ids] = local[k][0::2]
            nats_spliced[k][ids] = local[k][1::2]
            counts[k][ids] = local_counts[k][0::2]
        print(f"{start + len(part)}/{len(pairs)}", flush=True)
    result = {
        "model": str(args.run / "best.mica"),
        "model_sha256": digest(args.run / "best.mica"),
        "probe": args.probe,
        "probe_sha256": PROBES[args.probe][1],
        "protocol": "exact exported integer C engine, first 24 bytes replaced by fixed donor; use(k) is spliced minus true bits/target for targets p>=24+k, 5000 paired record resamples",
        "records": len(pairs), "k": {},
    }
    for k in KS:
        intervals = paired_bootstrap(nats_spliced[k], nats_true[k],
                                     counts[k], n_boot=5000, seed=20260928)
        result["k"][str(k)] = {
            "positions": int(counts[k].sum()),
            "use_bits": intervals[0],
            "ci95": list(intervals[1:]),
            "bits_true": nats_true[k].sum() / counts[k].sum() / math.log(2),
        }
    write_json(args.out, result)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--probe", choices=tuple(PROBES), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()
    if digest(PROBES[args.probe][0]) != PROBES[args.probe][1]:
        raise RuntimeError("context probe changed")
    if not (args.run / "FINISHED").is_file():
        raise RuntimeError("model fit has not finished")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    cmd = [sys.executable, __file__, "--worker", "--run", str(args.run),
           "--probe", args.probe, "--out", str(args.out),
           "--chunk", str(args.chunk)]
    raise SystemExit(subprocess.call(cmd, env=worker_env(run_geometry(args.run))))


if __name__ == "__main__":
    main()
