#!/usr/bin/env python3
"""Scale study: find the smallest geometry that clears a stated bar.

The v0.1 document planned this as "Phase Five scale study"; R1 §14 reaches the
same place from the other direction ("If the field forgets information too
quickly, compare more cells or different topology while counting every byte").

Two axes, and measuring them separately matters because they cost differently:

  field     cells x channels -- the model's working memory. Costs RAM at run
                               time and NOTHING in the model file, because the
                               file stores rule programs, not per-cell values.
  rule book pages x candidates -- what the model knows how to do. This is 75%
                               of the file, so it is what the size claim is
                               about.

Each configuration runs in its own process with the geometry set by
environment variable, so a run can never silently mix two geometries. Every
arm gets the same number of search rounds and the same records, so the
comparison is budget-matched in the sense §16 requires.

    python3 r1/scale_study.py --axis field --rounds 200
    python3 r1/scale_study.py --axis rules --rounds 200
    python3 r1/scale_study.py --configs "192x24,384x24,768x32" --rounds 200
"""

from __future__ import annotations

import argparse, json, math, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def offsets_for(n_cells: int) -> str:
    """Scale R1's neighbour offsets to a different field size.

    R1 uses +-1, +-16, +37, -53 on 192 cells: two immediate neighbours, two at
    about a twelfth of the field, and two long hops at roughly a fifth and a
    quarter. Keeping those proportions keeps the propagation character the same
    as the field grows. Long hops are nudged to be coprime with the field so
    they do not fold onto a short cycle.
    """
    def coprime(v: int) -> int:
        v = max(2, v)
        while math.gcd(v, n_cells) != 1:
            v += 1
        return v
    mid = coprime(round(n_cells / 12))
    a = coprime(round(n_cells * 0.193))
    b = coprime(round(n_cells * 0.276))
    return f"1,-1,{mid},-{mid},{a},-{b}"


FIELD_AXIS = [(192, 24), (384, 24), (768, 24), (768, 32), (1536, 32), (3072, 32)]
RULES_AXIS = [(64, 32), (128, 32), (256, 32), (512, 32), (1024, 32)]


def run_one(env_extra: dict, args, tag: str, out_root: Path) -> dict:
    out = out_root / tag
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    env["OMP_NUM_THREADS"] = str(args.threads)

    cmd = [sys.executable, str(HERE / "run_search.py"),
           "--backend", args.backend, "--device", args.device,
           "--restarts", "1", "--rounds", str(args.rounds),
           "--batch-records", str(args.batch_records),
           "--record-bytes", str(args.record_bytes),
           "--records", args.records, "--val-records", args.val_records,
           "--val-records-n", str(args.val_records_n),
           "--validate-every", str(max(args.rounds // 4, 1)),
           "--log-every", str(max(args.rounds // 5, 1)),
           "--out", str(out)]
    if args.fixes:
        cmd += ["--bias-init", "spread", "--probe-init", "zero",
                "--routing", "pairdiff"]

    print(f"\n=== {tag} ===", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-1500:]); print(proc.stderr[-1500:])
        return {"config": tag, "error": proc.stderr.strip().splitlines()[-1:] or ["failed"]}

    res = json.loads((out / "search_result.json").read_text())
    hist = res["history"]["restart1"]
    start, end = hist[0]["bits_per_target"], hist[-1]["bits_per_target"]
    rec = {
        "config": tag,
        "geometry": env_extra,
        "model_file_bytes": None,
        "start_bits_per_target": start,
        "end_bits_per_target": end,
        "gained_bits": round(start - end, 4),
        "val_bits_per_target": res["selected"]["val_bits_per_target"],
        "updates_per_symbol": res["selected"]["updates_per_symbol"],
        "seconds_per_round": res["seconds_per_round"],
        "wall_seconds": round(wall, 1),
    }
    inst = res.get("instruments", {})
    rec["pages_reached"] = inst.get("pages_reached")
    rec["pages_total"] = inst.get("pages_total")
    rec["score_tie_fraction"] = inst.get("score_tie_fraction")
    print(f"  {start:.4f} -> {end:.4f} bits/target  (gained {start-end:.4f}) "
          f"in {wall/60:.1f} min", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", default="field", choices=["field", "rules"])
    ap.add_argument("--configs", default="",
                    help="explicit list, e.g. '192x24,768x32' (field) or "
                         "'64x32,256x32' (rules)")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--batch-records", type=int, default=32)
    ap.add_argument("--record-bytes", type=int, default=64)
    ap.add_argument("--records", default="r1/data/records/train.jsonl")
    ap.add_argument("--val-records", default="r1/data/records/val.jsonl")
    ap.add_argument("--val-records-n", type=int, default=64)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fixes", action="store_true", default=True)
    ap.add_argument("--no-fixes", dest="fixes", action="store_false")
    ap.add_argument("--out", default="r1/runs/scale")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.configs:
        pairs = [tuple(int(x) for x in c.split("x")) for c in args.configs.split(",")]
    else:
        pairs = FIELD_AXIS if args.axis == "field" else RULES_AXIS

    results = []
    for a, b in pairs:
        if args.axis == "field":
            env = {"MICA_CELLS": a, "MICA_CHANNELS": b,
                   "MICA_OFFSETS": offsets_for(a)}
            tag = f"field_{a}x{b}"
        else:
            env = {"MICA_PAGES": a, "MICA_CANDIDATES": b}
            tag = f"rules_{a}x{b}"
        results.append(run_one(env, args, tag, out_root))

    # report
    print("\n" + "=" * 92)
    print(f"{'config':<18}{'file B':>10}{'state B':>10}{'start':>9}{'end':>9}"
          f"{'gained':>9}{'upd/sym':>10}{'s/round':>10}")
    print("-" * 92)
    for r in results:
        if "error" in r:
            print(f"{r['config']:<18}  failed: {r['error']}")
            continue
        g = r["geometry"]
        cells = int(g.get("MICA_CELLS", 192))
        chans = int(g.get("MICA_CHANNELS", 24))
        pages = int(g.get("MICA_PAGES", 64))
        cands = int(g.get("MICA_CANDIDATES", 32))
        file_b = 128 + 258 * 12 * 4 + pages * cands * 32 + 258 * 34
        r["model_file_bytes"] = file_b
        print(f"{r['config']:<18}{file_b:>10,}{cells*chans:>10,}"
              f"{r['start_bits_per_target']:>9.4f}{r['end_bits_per_target']:>9.4f}"
              f"{r['gained_bits']:>9.4f}{r['updates_per_symbol']:>10.0f}"
              f"{r['seconds_per_round']:>10.2f}")

    print("\nTrigram bar 2.1984 bits/byte; section 15 gate 2.0885.")
    payload = {"axis": args.axis, "rounds": args.rounds,
               "record_bytes": args.record_bytes, "fixes_applied": args.fixes,
               "results": results}
    (out_root / f"scale_{args.axis}.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_root}/scale_{args.axis}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
