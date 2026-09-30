#!/usr/bin/env python3
"""Bounded hard-selector proposal search using exact integer MICA validation.

The donor supplies only proposed sc_nb/sc_ch/sc_co changes. Every trial starts
from the unchanged control model; a trial is accepted only if its pooled loss
improves on a separate selection set. Clean validation is handled afterwards.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--donor", type=Path, required=True)
    ap.add_argument("--val-file", type=Path, required=True)
    ap.add_argument("--clean-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--records", type=int, default=64)
    ap.add_argument("--min-gain", type=float, default=0.002)
    ap.add_argument("--grouping", choices=("field_phase", "joint_phase"),
                    default="field_phase")
    args = ap.parse_args()
    if args.records < 1 or args.min_gain < 0:
        ap.error("records must be positive and min-gain must be nonnegative")

    here = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(here / "data"))
    control_info = json.loads((args.control / "run_info.json").read_text())
    donor_info = json.loads((args.donor / "run_info.json").read_text())
    if control_info["env"] != donor_info["env"]:
        raise ValueError("control and donor geometry differ")
    for key in list(os.environ):
        if key.startswith("MICA_"):
            del os.environ[key]
    os.environ.update({k: str(v) for k, v in control_info["env"].items()})
    from make_records import load_records
    from mica_r1 import batch, serialize, spec
    import numpy as np

    args.out.mkdir(parents=True, exist_ok=True)
    status = args.out / "status.json"
    write_json(status, {"stage": "preparing", "time": time.time()})
    clean = set(r[:256] for r in load_records(args.clean_file))
    candidates = load_records(args.val_file)[1000:]
    selection = []
    seen = set()
    for rec in candidates:
        rec = rec[:256]
        if rec and rec not in clean and rec not in seen:
            selection.append(rec)
            seen.add(rec)
            if len(selection) == args.records:
                break
    if len(selection) < args.records:
        raise ValueError("insufficient disjoint validation records")
    selection_file = args.out / f"selection_dev{args.records}.jsonl"
    selection_file.write_text("".join(r.hex() + "\n" for r in selection), encoding="ascii")

    control = serialize.load(args.control / "best.mica")
    donor = serialize.load(args.donor / "best.mica")
    batch.MAX_TICKS = spec.MAX_TICKS
    counts = np.asarray([len(r) + 1 for r in selection], dtype=np.int64)

    def exact_bits(model) -> float:
        stack = batch.ModelStack([model], score_backend="c")
        _loss, _u, _tgt, per_rec, _ur = batch.evaluate(
            stack, selection, per_record=True)
        return float(np.dot(np.asarray(per_rec[0]), counts) /
                     counts.sum() / math.log(2))

    start = time.time()
    base_bits = exact_bits(control)
    results = []
    trial = copy.deepcopy(control)
    fields = ("sc_nb", "sc_ch", "sc_co")
    groups = []
    for phase in range(spec.N_PHASE):
        page = slice(phase * spec.PAGE_STRIDE, (phase + 1) * spec.PAGE_STRIDE)
        field_groups = ((fields,) if args.grouping == "joint_phase" else
                        tuple((field,) for field in fields))
        for group_fields in field_groups:
            changed = sum(int(np.count_nonzero(
                getattr(control, field)[page] != getattr(donor, field)[page]))
                          for field in group_fields)
            if changed:
                groups.append((group_fields, phase, page, changed))
    write_json(status, {"stage": "evaluating", "base_bits": base_bits,
                        "groups": len(groups), "done": 0, "time": time.time()})

    for group_fields, phase, page, changed in groups:
        # Restore the prior rule page on every trial. A rejected proposal never
        # mutates the exported control or becomes the starting point for another.
        for field in group_fields:
            getattr(trial, field)[page] = getattr(donor, field)[page]
        bits = exact_bits(trial)
        for field in group_fields:
            getattr(trial, field)[page] = getattr(control, field)[page]
        row = {"fields": list(group_fields), "phase": phase,
               "changed_entries": changed,
               "selection_bits": bits, "gain": base_bits - bits}
        results.append(row)
        write_json(args.out / "proposal_results.json",
                   {"base_bits": base_bits, "min_gain": args.min_gain,
                    "selection_records": len(selection), "proposals": results})
        write_json(status, {"stage": "evaluating", "base_bits": base_bits,
                            "groups": len(groups), "done": len(results),
                            "best_gain": max(x["gain"] for x in results),
                            "time": time.time()})
        print(f"[exact-select] {len(results)}/{len(groups)} {group_fields} "
              f"phase={phase} changed={changed} bits={bits:.6f} "
              f"gain={base_bits-bits:+.6f}", flush=True)

    best = min(results, key=lambda x: x["selection_bits"]) if results else None
    accepted = best is not None and best["gain"] >= args.min_gain
    chosen = copy.deepcopy(control)
    if accepted:
        page = slice(best["phase"] * spec.PAGE_STRIDE,
                     (best["phase"] + 1) * spec.PAGE_STRIDE)
        for field in best["fields"]:
            getattr(chosen, field)[page] = getattr(donor, field)[page]
    candidate_dir = args.out / "candidate"
    candidate_dir.mkdir(exist_ok=True)
    serialize.save(chosen, candidate_dir / "best.mica")
    shutil.copy2(args.control / "run_info.json", candidate_dir / "run_info.json")
    chosen_bits = exact_bits(chosen)
    if accepted and abs(chosen_bits - best["selection_bits"]) > 1e-8:
        raise AssertionError("exported candidate does not reproduce accepted score")
    if not accepted and abs(chosen_bits - base_bits) > 1e-8:
        raise AssertionError("rollback did not preserve control score")
    write_json(status, {"stage": "selection_complete", "accepted": accepted,
                        "base_bits": base_bits, "candidate_bits": chosen_bits,
                        "best_proposal": best, "groups": len(groups),
                        "seconds": round(time.time() - start, 1),
                        "time": time.time()})
    print(f"[exact-select] accepted={accepted} base={base_bits:.6f} "
          f"candidate={chosen_bits:.6f} best={best}", flush=True)


if __name__ == "__main__":
    main()
