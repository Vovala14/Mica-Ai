#!/usr/bin/env python3
"""Search active MICA rule selectors with exact integer accept/reject.

No inference architecture changes. The model's own integer execution trace
identifies active winning rules. Small edits to their selection scores are
screened, confirmed on disjoint validation records, and otherwise rolled back.
The fixed clean evaluation corpus is excluded from every search stage.
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


def save_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--val-file", type=Path, required=True)
    ap.add_argument("--clean-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--active-rules", type=int, default=12)
    ap.add_argument("--screen-top", type=int, default=12)
    ap.add_argument("--confirm-top", type=int, default=4)
    ap.add_argument("--min-gain", type=float, default=0.002)
    args = ap.parse_args()
    if min(args.active_rules, args.screen_top, args.confirm_top) < 1:
        ap.error("search budgets must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    status_path = args.out / "status.json"
    save_json(status_path, {"stage": "preparing", "time": time.time()})
    r1 = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(r1), str(r1 / "data")]
    info = json.loads((args.base / "run_info.json").read_text())
    for key in list(os.environ):
        if key.startswith("MICA_"):
            del os.environ[key]
    os.environ.update({k: str(v) for k, v in info["env"].items()})
    from make_records import load_records
    from mica_r1 import batch, serialize, spec
    import numpy as np

    clean = set(rec[:256] for rec in load_records(args.clean_file))
    distinct = []
    seen = set()
    for rec in load_records(args.val_file)[1000:]:
        rec = rec[:256]
        if rec and rec not in seen and rec not in clean:
            distinct.append(rec)
            seen.add(rec)
            if len(distinct) == 96:
                break
    if len(distinct) < 96:
        raise ValueError("need 96 disjoint validation records")
    screen, accept = distinct[:32], distinct[32:96]
    (args.out / "screen_dev32.jsonl").write_text(
        "".join(rec.hex() + "\n" for rec in screen), encoding="ascii")
    (args.out / "accept_dev64.jsonl").write_text(
        "".join(rec.hex() + "\n" for rec in accept), encoding="ascii")

    base = serialize.load(args.base / "best.mica")
    trial = copy.deepcopy(base)
    batch.MAX_TICKS = spec.MAX_TICKS

    def bits(model, records):
        stack = batch.ModelStack([model], score_backend="c")
        _loss, _u, _target, per_record, _ur = batch.evaluate(
            stack, records, per_record=True)
        n = np.asarray([len(r) + 1 for r in records], dtype=np.int64)
        return float(np.dot(np.asarray(per_record[0]), n) /
                     n.sum() / math.log(2))

    # The trace is used only to prioritize which existing selector rules to
    # try; the loss of every trial is computed by the ordinary exact evaluator.
    trace = []
    batch.TRACE = trace
    base_screen8 = bits(base, screen[:8])
    batch.TRACE = None
    activity = np.zeros((spec.N_PAGES, spec.N_CANDIDATES), dtype=np.int64)
    for row in trace:
        if row[0] == "c":
            np.add.at(activity, (row[3], row[4]), 1)
    ranked = np.argsort(-activity.ravel(), kind="stable")
    active = [(int(ix // spec.N_CANDIDATES),
               int(ix % spec.N_CANDIDATES),
               int(activity.ravel()[ix]))
              for ix in ranked[:args.active_rules]
              if activity.ravel()[ix] > 0]
    if not active:
        raise RuntimeError("exact trace found no active rules")

    proposals = []
    for page, cand, count in active:
        old_bias = int(base.sc_bias[page, cand])
        for delta in (-16, -4, 4, 16):
            value = old_bias + delta
            if -32767 <= value <= 32767:
                proposals.append({"field": "sc_bias", "index": [page, cand],
                                  "old": old_bias, "new": value,
                                  "active_count": count})
        for term in range(min(2, spec.N_SCORE_TERMS)):
            old_co = int(base.sc_co[page, cand, term])
            for new_co in (-1, 0, 1):
                if new_co != old_co:
                    proposals.append({"field": "sc_co",
                                      "index": [page, cand, term],
                                      "old": old_co, "new": new_co,
                                      "active_count": count})

    def evaluate_trial(proposal, records):
        index = tuple(proposal["index"])
        arr = getattr(trial, proposal["field"])
        if int(arr[index]) != proposal["old"]:
            raise AssertionError("trial was not rolled back")
        arr[index] = proposal["new"]
        score = bits(trial, records)
        arr[index] = proposal["old"]
        return score

    start = time.time()
    screened = []
    save_json(status_path, {"stage": "screening", "done": 0,
                            "total": len(proposals), "active_rules": active,
                            "time": time.time()})
    for proposal in proposals:
        row = dict(proposal)
        row["screen8_bits"] = evaluate_trial(proposal, screen[:8])
        row["screen8_gain"] = base_screen8 - row["screen8_bits"]
        screened.append(row)
        save_json(args.out / "screen_results.json",
                  {"base_screen8_bits": base_screen8,
                   "proposals": screened})
        save_json(status_path, {"stage": "screening", "done": len(screened),
                                "total": len(proposals),
                                "best_screen8_gain": max(x["screen8_gain"]
                                                         for x in screened),
                                "time": time.time()})
        print(f"[active-rule] screen {len(screened)}/{len(proposals)} "
              f"{row['field']} {row['index']} {row['old']}->{row['new']} "
              f"gain={row['screen8_gain']:+.6f}", flush=True)

    base_screen32 = bits(base, screen)
    finalists32 = []
    for row in sorted(screened, key=lambda x: x["screen8_bits"])[
            :args.screen_top]:
        item = dict(row)
        item["screen32_bits"] = evaluate_trial(item, screen)
        item["screen32_gain"] = base_screen32 - item["screen32_bits"]
        finalists32.append(item)
        print(f"[active-rule] screen32 {item['field']} {item['index']} "
              f"gain={item['screen32_gain']:+.6f}", flush=True)
    save_json(args.out / "screen32_results.json",
              {"base_screen32_bits": base_screen32,
               "proposals": finalists32})

    base_accept = bits(base, accept)
    finalists64 = []
    for row in sorted(finalists32, key=lambda x: x["screen32_bits"])[
            :args.confirm_top]:
        item = dict(row)
        item["accept64_bits"] = evaluate_trial(item, accept)
        item["accept64_gain"] = base_accept - item["accept64_bits"]
        finalists64.append(item)
        print(f"[active-rule] accept64 {item['field']} {item['index']} "
              f"gain={item['accept64_gain']:+.6f}", flush=True)
    save_json(args.out / "accept_results.json",
              {"base_accept64_bits": base_accept,
               "min_gain": args.min_gain,
               "proposals": finalists64})

    best = min(finalists64, key=lambda x: x["accept64_bits"])
    accepted = best["accept64_gain"] >= args.min_gain
    chosen = copy.deepcopy(base)
    if accepted:
        getattr(chosen, best["field"])[tuple(best["index"])] = best["new"]
    candidate_dir = args.out / "candidate"
    candidate_dir.mkdir(exist_ok=True)
    serialize.save(chosen, candidate_dir / "best.mica")
    shutil.copy2(args.base / "run_info.json", candidate_dir / "run_info.json")
    chosen_bits = bits(chosen, accept)
    expected = best["accept64_bits"] if accepted else base_accept
    if abs(chosen_bits - expected) > 1e-8:
        raise AssertionError("exported candidate disagrees with acceptance")
    save_json(status_path, {"stage": "selection_complete",
                            "accepted": accepted,
                            "base_accept64_bits": base_accept,
                            "candidate_accept64_bits": chosen_bits,
                            "best_proposal": best,
                            "total_proposals": len(proposals),
                            "seconds": round(time.time() - start, 1),
                            "time": time.time()})
    print(f"[active-rule] accepted={accepted} "
          f"base={base_accept:.6f} candidate={chosen_bits:.6f}", flush=True)


if __name__ == "__main__":
    main()
