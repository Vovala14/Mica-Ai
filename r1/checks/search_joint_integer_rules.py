#!/usr/bin/env python3
"""Bounded native MICA search: change one selector and refit its rule immediate.

Proposals and immediate values are chosen using training records only. One
untouched everyday development slice accepts or rejects the chosen proposal;
the clean val1000 is evaluated only after acceptance. No inference code changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time


R1 = Path(__file__).resolve().parents[1]
ROOT = R1.parent
DEFAULT_BASE = R1 / "runs/overnight/mica_lag64_continuation_20260927"
DEFAULT_OUT = R1 / "runs/codex_joint_integer_rules_20260927"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reservoir_records(path: Path, size: int, scan: int, excluded: set[bytes]) -> list[bytes]:
    """Deterministic sample from a bounded prefix of the *training* split."""
    rng = random.Random(20260927)
    sample: list[bytes] = []
    eligible = 0
    with path.open(encoding="ascii") as src:
        for line_no, line in enumerate(src):
            if line_no >= scan:
                break
            rec = bytes.fromhex(line.strip())[:256]
            if not rec or rec in excluded:
                continue
            eligible += 1
            if len(sample) < size:
                sample.append(rec)
            else:
                j = rng.randrange(eligible)
                if j < size:
                    sample[j] = rec
    if len(sample) != size or len(set(sample)) != size:
        raise ValueError("not enough unique training records in sample")
    return sample


def immediate(model, page: int, candidate: int, lane: int) -> int:
    return int(model.op_b[page, candidate] if lane == 0 else
               model.op_v[page, candidate, lane - 1])


def put_immediate(model, page: int, candidate: int, lane: int, value: int) -> None:
    if lane == 0:
        model.op_b[page, candidate] = value
    else:
        model.op_v[page, candidate, lane - 1] = value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--train-file", type=Path,
                    default=R1 / "data/everyday/train.jsonl")
    ap.add_argument("--dev-file", type=Path,
                    default=R1 / "data/everyday/val.jsonl")
    ap.add_argument("--clean-file", type=Path,
                    default=R1 / "data/eval_clean/val1000.jsonl")
    ap.add_argument("--active-rules", type=int, default=8)
    ap.add_argument("--screen-top", type=int, default=8)
    ap.add_argument("--train-scan", type=int, default=10000)
    ap.add_argument("--min-gain", type=float, default=0.002)
    args = ap.parse_args()
    launcher_logs = {"search.stdout.log", "search.stderr.log"}
    if args.out.exists() and any(p.name not in launcher_logs
                                 for p in args.out.iterdir()):
        ap.error("output directory must be empty to preserve prior results")
    if min(args.active_rules, args.screen_top, args.train_scan) < 1:
        ap.error("search budgets must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    info = json.loads((args.base / "run_info.json").read_text(encoding="utf-8"))
    for key in list(os.environ):
        if key.startswith("MICA_"):
            del os.environ[key]
    os.environ.update({k: str(v) for k, v in info["env"].items()})
    sys.path[:0] = [str(R1), str(R1 / "data"), str(R1 / "checks")]
    from make_records import load_records
    from mica_r1 import batch, serialize, spec
    from common import paired_bootstrap
    import numpy as np

    clean = [rec[:256] for rec in load_records(args.clean_file)]
    clean_set = set(clean)
    train = reservoir_records(args.train_file, 32, args.train_scan, clean_set)
    dev = []
    seen = set()
    for rec in load_records(args.dev_file)[1000:]:
        rec = rec[:256]
        if rec and rec not in clean_set and rec not in seen and rec not in train:
            dev.append(rec)
            seen.add(rec)
            if len(dev) == 64:
                break
    if len(dev) != 64 or set(train) & set(dev) or set(train) & clean_set:
        raise ValueError("training, development and clean evaluation must be disjoint")
    for name, records in (("train_screen32", train), ("dev_accept64", dev)):
        (args.out / f"{name}.jsonl").write_text(
            "".join(rec.hex() + "\n" for rec in records), encoding="ascii")

    base_file = args.base / "best.mica"
    base_sha = sha256(base_file)
    base = serialize.load(base_file)
    trial = copy.deepcopy(base)
    batch.MAX_TICKS = spec.MAX_TICKS

    def exact_bits(model, records: list[bytes]) -> tuple[float, list[float]]:
        stack = batch.ModelStack([model], score_backend="c")
        _, _, _, per_record, _ = batch.evaluate(stack, records, per_record=True)
        nats = np.asarray(per_record[0], dtype=np.float64)
        counts = np.asarray([len(r) + 1 for r in records], dtype=np.int64)
        return (float(np.dot(nats, counts) / counts.sum() / math.log(2)),
                (nats * counts).tolist())

    # Trace only training records. It ranks existing, active hard rules; all
    # candidate and immediate choices are made on training records as well.
    trace = []
    batch.TRACE = trace
    try:
        base_train8, _ = exact_bits(base, train[:8])
    finally:
        batch.TRACE = None
    activity = np.zeros((spec.N_PAGES, spec.N_CANDIDATES), dtype=np.int64)
    for row in trace:
        if row[0] == "c":
            np.add.at(activity, (row[3], row[4]), 1)
    channel_importance = np.bincount(
        base.pr_chan.ravel(), weights=np.abs(base.pr_co.astype(np.int32)).ravel(),
        minlength=spec.N_CHANNELS)
    active = []
    for flat in np.argsort(-activity.ravel(), kind="stable"):
        page, candidate = divmod(int(flat), spec.N_CANDIDATES)
        count = int(activity[page, candidate])
        if count == 0:
            break
        if int(base.op_code[page, candidate]) != spec.VSET:
            continue
        terms = [term for term in range(spec.N_SCORE_TERMS)
                 if base.sc_co[page, candidate, term] != 0
                 and base.sc_ch[page, candidate, term] < spec.TAPE_CHANNELS
                 and 0 < base.sc_nb[page, candidate, term] < 5]
        if not terms:
            continue
        term = min(terms, key=lambda t: int(base.sc_nb[page, candidate, t]))
        dest = int(base.op_d[page, candidate])
        lanes = [lane for lane in range(spec.VSET_WIDTH)
                 if dest + lane < spec.N_CHANNELS]
        lanes.sort(key=lambda lane: -channel_importance[dest + lane])
        active.append({"page": page, "candidate": candidate,
                       "activity": count, "term": term, "lanes": lanes[:2]})
        if len(active) == args.active_rules:
            break
    if not active:
        raise RuntimeError("no active VSET rule with a near nonzero tape predicate")
    atomic_json(args.out / "protocol.json", {
        "base": str(base_file), "base_sha256": base_sha,
        "train_file": str(args.train_file), "dev_file": str(args.dev_file),
        "clean_file": str(args.clean_file),
        "train_screen_records": 32, "dev_accept_records": 64,
        "proposal": "active rule, one tape selector to offset -6/-8, refit one VSET immediate",
        "immediate_deltas": [-16, -4, 0, 4, 16],
        "active": active, "min_gain": args.min_gain,
    })

    def score_proposal(row: dict, records: list[bytes]) -> float:
        p, c, t = row["page"], row["candidate"], row["term"]
        lane = row["lane"]
        old_nb = int(trial.sc_nb[p, c, t])
        old_value = immediate(trial, p, c, lane)
        if old_nb != row["old_nb"] or old_value != row["old_immediate"]:
            raise AssertionError("previous trial was not rolled back")
        try:
            trial.sc_nb[p, c, t] = row["new_nb"]
            put_immediate(trial, p, c, lane, row["new_immediate"])
            bits, _ = exact_bits(trial, records)
            return bits
        finally:
            trial.sc_nb[p, c, t] = old_nb
            put_immediate(trial, p, c, lane, old_value)

    proposals = []
    for rule in active:
        p, c, t = rule["page"], rule["candidate"], rule["term"]
        old_nb = int(base.sc_nb[p, c, t])
        for new_nb in (5, 6):  # -6 and -8 in this model's offsets
            if new_nb == old_nb:
                continue
            for lane in rule["lanes"]:
                old = immediate(base, p, c, lane)
                for delta in (-16, -4, 0, 4, 16):
                    new = old + delta
                    if -127 <= new <= 127:
                        proposals.append({"page": p, "candidate": c, "term": t,
                                          "lane": lane, "activity": rule["activity"],
                                          "old_nb": old_nb, "new_nb": new_nb,
                                          "old_immediate": old, "new_immediate": new})
    if not proposals:
        raise RuntimeError("no valid bounded proposals")
    atomic_json(args.out / "status.json", {"stage": "training_screen8",
                                              "done": 0, "total": len(proposals)})
    screened = []
    for row in proposals:
        item = dict(row)
        item["train8_bits"] = score_proposal(item, train[:8])
        item["train8_gain"] = base_train8 - item["train8_bits"]
        screened.append(item)
        atomic_json(args.out / "status.json", {"stage": "training_screen8",
                    "done": len(screened), "total": len(proposals),
                    "best_train8_gain": max(r["train8_gain"] for r in screened)})
    atomic_json(args.out / "screen8.json", {
        "base_bits": base_train8, "proposals": screened})

    base_train32, _ = exact_bits(base, train)
    finalists = []
    for row in sorted(screened, key=lambda r: r["train8_bits"])[:args.screen_top]:
        item = dict(row)
        item["train32_bits"] = score_proposal(item, train)
        item["train32_gain"] = base_train32 - item["train32_bits"]
        finalists.append(item)
    atomic_json(args.out / "screen32.json", {
        "base_bits": base_train32, "proposals": finalists})
    best = min(finalists, key=lambda r: r["train32_bits"])

    # Exactly one training-selected proposal reaches the untouched development
    # accept set. A rejected trial exports the untouched base model.
    base_dev, base_nats = exact_bits(base, dev)
    trial_dev = score_proposal(best, dev)
    p, c, t, lane = best["page"], best["candidate"], best["term"], best["lane"]
    trial.sc_nb[p, c, t] = best["new_nb"]
    put_immediate(trial, p, c, lane, best["new_immediate"])
    _, trial_nats = exact_bits(trial, dev)
    trial.sc_nb[p, c, t] = best["old_nb"]
    put_immediate(trial, p, c, lane, best["old_immediate"])
    counts = [len(rec) + 1 for rec in dev]
    difference, lo, hi = paired_bootstrap(trial_nats, base_nats, counts)
    if abs((trial_dev - base_dev) - difference) > 1e-8:
        raise AssertionError("paired development difference disagrees with pooled bits")
    accepted = difference <= -args.min_gain and hi < 0
    chosen = copy.deepcopy(base)
    if accepted:
        chosen.sc_nb[p, c, t] = best["new_nb"]
        put_immediate(chosen, p, c, lane, best["new_immediate"])
    candidate_dir = args.out / "candidate"
    candidate_dir.mkdir()
    serialize.save(chosen, candidate_dir / "best.mica")
    shutil.copy2(args.base / "run_info.json", candidate_dir / "run_info.json")
    chosen_sha = sha256(candidate_dir / "best.mica")
    if not accepted and chosen_sha != base_sha:
        raise AssertionError("rejected proposal did not roll back byte-for-byte")
    reloaded = serialize.load(candidate_dir / "best.mica")
    chosen_dev, _ = exact_bits(reloaded, dev)
    expected = trial_dev if accepted else base_dev
    if abs(chosen_dev - expected) > 1e-8:
        raise AssertionError("exported candidate does not reproduce decision")
    result = {"accepted": accepted, "base": str(base_file),
              "base_sha256": base_sha, "candidate_sha256": chosen_sha,
              "training_best": best, "train32_base_bits": base_train32,
              "dev64_base_bits": base_dev, "dev64_trial_bits": trial_dev,
              "dev64_difference": difference, "dev64_95ci": [lo, hi],
              "clean_val1000": "not examined for rejected proposal",
              "seconds": round(time.time() - t0, 1)}
    atomic_json(args.out / "results.json", result)
    atomic_json(args.out / "status.json", {"stage": "selection_complete",
              "accepted": accepted, "result": str(args.out / "results.json")})
    print(json.dumps(result, indent=2), flush=True)

    if accepted:
        cmd = [sys.executable, str(R1 / "checks/eval_int.py"),
               "--run", f"candidate={candidate_dir}", "--set", "val1000",
               "--val-file", str(args.clean_file), "--val-skip", "0",
               "--out", str(args.out / "clean_val1000.json")]
        subprocess.run(cmd, cwd=ROOT, check=True)
        cmd = [sys.executable, str(R1 / "checks/eval_mica_raw_generation.py"),
               "--run", str(candidate_dir), "--prompts",
               str(R1 / "checks/generation_dev10.json"), "--limit", "10",
               "--byte-limit", "100", "--out",
               str(args.out / "generation_dev10.json")]
        subprocess.run(cmd, cwd=ROOT, check=True)
        baseline_eval = json.loads((args.base / "samples/on_exit/clean_val1000.json")
                                   .read_text(encoding="utf-8"))["lag64"]
        new_eval = json.loads((args.out / "clean_val1000.json")
                              .read_text(encoding="utf-8"))["candidate"]
        if baseline_eval["counts"] != new_eval["counts"]:
            raise AssertionError("clean evaluation target counts differ")
        clean_difference, clean_lo, clean_hi = paired_bootstrap(
            new_eval["nats"], baseline_eval["nats"], new_eval["counts"])
        result["clean_val1000"] = {
            "base_bits": baseline_eval["bits"], "candidate_bits": new_eval["bits"],
            "difference": clean_difference, "paired_95ci": [clean_lo, clean_hi],
            "targets": new_eval["targets"], "records": new_eval["records"]}
        atomic_json(args.out / "results.json", result)
        atomic_json(args.out / "status.json", {"stage": "complete",
                  "accepted": True, "result": str(args.out / "results.json")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
