#!/usr/bin/env python3
"""Two-minute benchmark: what will a full search round cost on this machine?

Run this before committing the machine to a long schedule. It times one
full-specification round (17 models, 32 records, 256 bytes) on each available
backend and projects the section 13 schedule from the measurement.
"""

from __future__ import annotations

import argparse, platform, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

import numpy as np
from make_records import load_records
from mica_r1.engine import random_model
from mica_r1.search import XorShift32, make_child


def project(sec_per_round: float, restarts: int, rounds: int, concurrent: bool):
    total = sec_per_round * rounds * (1 if concurrent else restarts)
    for unit, div in (("seconds", 1), ("minutes", 60), ("hours", 3600), ("days", 86400)):
        if total < 60 * div or unit == "days":
            return f"{total/div:.1f} {unit}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="")
    ap.add_argument("--record-bytes", type=int, default=256)
    ap.add_argument("--batch-records", type=int, default=32)
    ap.add_argument("--children", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    print(f"host: {platform.platform()} / {platform.machine()}")
    # Use whichever record set is present; the packaged build ships 'full'.
    candidates = ([args.records] if args.records else
                  ["r1/data/full/train.jsonl", "r1/data/records/train.jsonl",
                   "r1/data/mydata/train.jsonl"])
    path = next((c for c in candidates if Path(c).exists()), None)
    if path is None:
        print("No record file found. Looked for:")
        for c in candidates:
            print(f"  {c}")
        print("Build one with:  python r1/data/ingest.py --input <folder> "
              "--out r1/data/mydata")
        return 1
    print(f"records: {path}")
    pool = [r for r in load_records(path)[:2000]
            if len(r) >= args.record_bytes][:args.batch_records]
    if not pool:
        print(f"No records of at least {args.record_bytes} bytes in {path}.")
        return 1
    pool = [r[:args.record_bytes] for r in pool]
    print(f"batch: {args.children+1} models x {len(pool)} records "
          f"x {args.record_bytes} bytes\n")

    m = random_model(1)
    rng = XorShift32(1)
    models = [m] + [make_child(m, rng) for _ in range(args.children)]

    results = {}

    from mica_r1.batch import ModelStack, evaluate as np_eval
    t = []
    for _ in range(args.repeats):
        t0 = time.time(); np_eval(ModelStack(models), pool); t.append(time.time() - t0)
    results["numpy/cpu (sparse)"] = min(t)

    try:
        import torch
        from mica_r1.torch_batch import TorchModelStack, evaluate as t_eval
        devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dname in devs:
            dev = torch.device(dname)
            label = f"torch/{dname} (dense)"
            if dname == "cuda":
                label = f"torch/{torch.cuda.get_device_name(0)} (dense)"
            ms = TorchModelStack(models, dev)
            t_eval(ms, pool[:2])                       # warm up kernels
            t = []
            for _ in range(args.repeats):
                if dname == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time(); t_eval(ms, pool)
                if dname == "cuda":
                    torch.cuda.synchronize()
                t.append(time.time() - t0)
            results[label] = min(t)
    except ImportError:
        print("(torch not installed; skipping tensor backends)")

    print(f"{'backend':<44}{'per round':>12}{'8x10,000 rounds':>20}")
    print("-" * 76)
    for name, sec in sorted(results.items(), key=lambda kv: kv[1]):
        concurrent = "cuda" in name.lower() or "radeon" in name.lower() or "nvidia" in name.lower()
        print(f"{name:<44}{sec:>10.1f}s{project(sec, 8, 10000, False):>20}")
    print("\nReference: 313.0s/round measured on 2 cloud CPU cores -> 290 days.")
    fastest = min(results.values())
    print(f"This machine's fastest backend is {313.0/fastest:.1f}x that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
