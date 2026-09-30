#!/usr/bin/env python3
"""Section 14 first diagnostic: overfit 32 short records.

  "Before the full corpus, overfit 32 short records. If the search cannot lower
   target loss or reproduce local byte patterns, inspect routing usage,
   active-set size, score ties, saturation counts, and mutation acceptance.
   A persistent failure here is grounds to stop."
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

import numpy as np
from make_records import load_records
from mica_r1.engine import random_model
from mica_r1.batch import ModelStack, evaluate, objective
from mica_r1.search import SearchConfig, search
from mica_r1.diagnose import instrument
from mica_r1 import serialize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="r1/data/records/train.jsonl")
    ap.add_argument("--n-records", type=int, default=32)
    ap.add_argument("--record-bytes", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="r1/runs/diagnostic.json")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    pool = load_records(args.records)
    # "32 short records" - trimmed at a UTF-8 boundary so every record is valid
    recs = []
    for r in pool:
        if len(r) < args.record_bytes:
            continue
        k = args.record_bytes
        while k > 0 and (r[k] & 0xC0) == 0x80:
            k -= 1
        chunk = r[:k]
        if len(chunk) >= args.record_bytes // 2 and chunk.strip():
            recs.append(chunk)
        if len(recs) == args.n_records:
            break

    total = sum(len(r) for r in recs)
    print(f"[diag] {len(recs)} records, {total} bytes total, "
          f"mean {total/len(recs):.1f} bytes")
    print(f"[diag] sample: {recs[0][:60]!r}")

    cfg = SearchConfig(seed=args.seed, rounds=args.rounds,
                       batch_records=len(recs))

    init = random_model(args.seed)
    loss0, upd0, _ = evaluate(ModelStack([init]), recs)
    print(f"[diag] initial: {loss0[0]/np.log(2):.4f} bits/target "
          f"(uniform over 257 eligible symbols = {np.log2(257):.4f})")
    diag0 = instrument(init, recs[:8])
    print(f"[diag] initial instruments: {json.dumps(diag0, indent=2)}")

    t0 = time.time()
    best, history = search(recs, cfg, log_every=5)
    wall = time.time() - t0

    lossF, updF, _ = evaluate(ModelStack([best]), recs)
    diagF = instrument(best, recs[:8])
    blob_bytes = len(serialize.dumps(best))

    result = {
        "spec": "MICA R1 section 14 first diagnostic",
        "records": {"count": len(recs), "bytes": total,
                    "mean_bytes": round(total / len(recs), 1)},
        "rounds": args.rounds, "seed": args.seed,
        "wall_seconds": round(wall, 1),
        "seconds_per_round": round(wall / args.rounds, 2),
        "bits_per_target": {
            "uniform_reference": round(float(np.log2(257)), 4),
            "initial": round(float(loss0[0] / np.log(2)), 4),
            "final": round(float(lossF[0] / np.log(2)), 4),
        },
        "objective_J": {"initial": round(float(objective(loss0, upd0)[0]), 6),
                        "final": round(float(objective(lossF, updF)[0]), 6)},
        "accept_rate": history[-1]["accept_rate"],
        "instruments": {"initial": diag0, "final": diagF},
        "model_file_bytes": blob_bytes,
        "history": history,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    serialize.save(best, Path(args.out).with_suffix(".mica"))

    print(f"\n[diag] {args.rounds} rounds in {wall/60:.1f} min "
          f"({wall/args.rounds:.1f}s/round)")
    print(f"[diag] bits/target {result['bits_per_target']['initial']:.4f} "
          f"-> {result['bits_per_target']['final']:.4f} "
          f"(uniform {result['bits_per_target']['uniform_reference']:.4f})")
    print(f"[diag] acceptance rate {result['accept_rate']}")
    print(f"[diag] wrote {args.out} and the {blob_bytes}-byte model file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
