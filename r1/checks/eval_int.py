#!/usr/bin/env python3
"""Score exported integer models (.mica) on a fixed record set with the exact
integer evaluator (mica_r1.batch.evaluate, compiled scorer).

    python r1/checks/eval_int.py --set val1000 --out r1/runs/checks/val1000.json \
        --run p16=r1/runs/sweep/p16_tiefix --run p16_self1=r1/runs/sweep/p16_self1

Each model runs in its own process under the geometry in its run_info.json
(the file header refuses a mismatch). Output: per model, pooled bits per
target, the trainer's mean-of-record-means, and per-record nats and counts
so any two models can be compared record by record.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (R1, record_set, run_geometry, worker_env, pooled_bits,
                    mean_record_bits, beat, utf8_stdout)


def worker(a) -> int:
    from mica_r1 import spec, serialize, batch
    m = serialize.load(a.mica)
    batch.MAX_TICKS = spec.MAX_TICKS
    recs = record_set(a.set, a.data_root, val_file=a.val_file,
                      test_file=a.test_file, val_skip=a.val_skip)
    t0 = time.time()
    nats, counts = [], []
    ms = batch.ModelStack([m], score_backend="c")
    for i in range(0, len(recs), a.chunk):
        part = recs[i:i + a.chunk]
        beat(f"eval {a.set} {i}")
        loss, _u, tgt, per_rec, _ur = batch.evaluate(ms, part, per_record=True)
        n_rec = [len(r) + 1 for r in part]
        nats += [float(x) * c for x, c in zip(per_rec[0], n_rec)]
        counts += n_rec
    out = {"mica": str(a.mica), "bytes": Path(a.mica).stat().st_size,
           "set": a.set, "records": len(recs), "targets": int(sum(counts)),
           "val_file": a.val_file, "test_file": a.test_file,
           "val_skip": a.val_skip,
           "bits": pooled_bits(nats, counts),
           "mean_record_bits": mean_record_bits(nats, counts),
           "nats": nats, "counts": counts,
           "seconds": round(time.time() - t0, 1),
           "geometry": spec.describe() if hasattr(spec, "describe") else None}
    Path(a.out).write_text(json.dumps(out))
    return 0


def main() -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[],
                    help="name=run_dir (containing best.mica and run_info.json)")
    ap.add_argument("--set", default="val1000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--val-file", default=None,
                    help="override the validation JSONL (for another corpus)")
    ap.add_argument("--test-file", default=None,
                    help="override the test JSONL")
    ap.add_argument("--val-skip", type=int, default=64,
                    help="records reserved for checkpoint selection (default 64)")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mica")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for spec_ in a.run:
        name, run_dir = spec_.split("=", 1)
        run_dir = Path(run_dir)
        tmp = out_path.with_name(out_path.stem + f".{name}.part.json")
        cmd = [sys.executable, __file__, "--worker", "--mica",
               str(run_dir / "best.mica"), "--set", a.set, "--out", str(tmp),
               "--chunk", str(a.chunk), "--val-skip", str(a.val_skip)] + \
              (["--data-root", a.data_root] if a.data_root else []) + \
              (["--val-file", a.val_file] if a.val_file else []) + \
              (["--test-file", a.test_file] if a.test_file else [])
        t0 = time.time()
        p = subprocess.run(cmd, env=worker_env(run_geometry(run_dir)),
                           capture_output=True, text=True)
        if p.returncode:
            print(f"[eval] {name}: FAILED\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}",
                  flush=True)
            continue
        r = json.loads(tmp.read_text())
        tmp.unlink()
        r["run_dir"] = str(run_dir)
        results[name] = r
        out_path.write_text(json.dumps(results))
        print(f"[eval] {a.set:9s} {name:26s} {r['bits']:.6f} bits/target pooled "
              f"({r['mean_record_bits']:.6f} mean of records), {r['records']} "
              f"records, {r['bytes']:,} B file, {time.time() - t0:.0f}s",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
