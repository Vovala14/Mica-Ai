#!/usr/bin/env python3
"""The section 16 required ablations.

  "Keep training budget and stored bytes matched while disabling phase routing,
   replacing event activity with full-field updates, freezing instructions,
   removing long-hop neighbors, and randomizing trained rules. If randomized
   rules plus trained readout perform equally well, the learned rewrite core
   has not earned the claimed contribution."

All six arms (full model plus five ablations) get the identical round budget,
record batches, seed and storage. Each runs in its own process because three of
the switches are read at import time.

The last arm is the decisive one. If randomising the rules costs nothing, the
cell field is decoration and the contribution claim fails.
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent

ARMS = [
    ("full",              {},                                   []),
    ("no_phase_routing",  {"MICA_NO_PHASE_ROUTING": "1"},       []),
    ("full_field",        {"MICA_FULL_FIELD": "1"},             []),
    ("frozen_instr",      {},                                   ["--freeze-instructions"]),
    ("no_long_hops",      {"MICA_OFFSETS": "1,-1,16,-16"},      []),
    ("random_rules",      {},                                   ["--randomize-rules"]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--batch-records", type=int, default=32)
    ap.add_argument("--record-bytes", type=int, default=64)
    ap.add_argument("--records", default="r1/data/records/train.jsonl")
    ap.add_argument("--val-records", default="r1/data/records/val.jsonl")
    ap.add_argument("--val-records-n", type=int, default=48)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fixes", action="store_true", default=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="r1/runs/ablations")
    args = ap.parse_args()

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    arms = [a for a in ARMS if not args.only or a[0] in args.only]

    print(f"[ablations] {len(arms)} arms x {args.rounds} rounds, "
          f"{args.batch_records} records x {args.record_bytes} bytes, "
          f"identical seed and budget\n")

    results = []
    for tag, env_extra, flags in arms:
        env = dict(os.environ)
        env.update(env_extra)
        env["OMP_NUM_THREADS"] = str(args.threads)
        cmd = [sys.executable, str(HERE / "run_search.py"),
               "--backend", args.backend, "--restarts", "1",
               "--rounds", str(args.rounds),
               "--batch-records", str(args.batch_records),
               "--record-bytes", str(args.record_bytes),
               "--records", args.records, "--val-records", args.val_records,
               "--val-records-n", str(args.val_records_n),
               "--validate-every", str(max(args.rounds // 3, 1)),
               "--log-every", str(max(args.rounds // 3, 1)),
               "--out", str(out_root / tag)] + flags
        if args.fixes:
            cmd += ["--bias-init", "spread", "--probe-init", "zero",
                    "--routing", "pairdiff"]

        print(f"=== {tag} ===", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-800:]); print(proc.stderr[-800:])
            results.append({"arm": tag, "error": True})
            continue
        res = json.loads((out_root / tag / "search_result.json").read_text())
        h = res["history"]["restart1"]
        row = {"arm": tag,
               "start_bits_per_target": h[0]["bits_per_target"],
               "end_bits_per_target": h[-1]["bits_per_target"],
               "gained_bits": round(h[0]["bits_per_target"] - h[-1]["bits_per_target"], 4),
               "val_bits_per_target": res["selected"]["val_bits_per_target"],
               "updates_per_symbol": res["selected"]["updates_per_symbol"],
               "pages_reached": res.get("instruments", {}).get("pages_reached"),
               "seconds": round(time.time() - t0, 1)}
        results.append(row)
        print(f"  {row['start_bits_per_target']:.4f} -> {row['end_bits_per_target']:.4f} "
              f"(gained {row['gained_bits']:.4f})  val {row['val_bits_per_target']:.4f}  "
              f"{row['seconds']/60:.1f} min\n", flush=True)

    ok = [r for r in results if "error" not in r]
    print("=" * 86)
    print(f"{'arm':<20}{'start':>9}{'end':>9}{'gained':>9}{'val':>10}"
          f"{'upd/sym':>10}{'pages':>8}")
    print("-" * 86)
    for r in ok:
        print(f"{r['arm']:<20}{r['start_bits_per_target']:>9.4f}"
              f"{r['end_bits_per_target']:>9.4f}{r['gained_bits']:>9.4f}"
              f"{r['val_bits_per_target']:>10.4f}{r['updates_per_symbol']:>10.0f}"
              f"{str(r['pages_reached']):>8}")

    full = next((r for r in ok if r["arm"] == "full"), None)
    rand = next((r for r in ok if r["arm"] == "random_rules"), None)
    if full and rand:
        d = rand["val_bits_per_target"] - full["val_bits_per_target"]
        print(f"\nSection 16 decisive test -- randomised rules vs full model: "
              f"{d:+.4f} bits")
        print("  Randomising the rules costs "
              + ("almost nothing. On this evidence the learned rewrite core has\n"
                 "  NOT earned the contribution claim." if abs(d) < 0.05 else
                 f"{abs(d):.4f} bits, so the rules are doing work."))

    (out_root / "ablations.json").write_text(json.dumps(
        {"rounds": args.rounds, "record_bytes": args.record_bytes,
         "fixes_applied": args.fixes, "arms": results}, indent=2))
    print(f"\nwrote {out_root}/ablations.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
