#!/usr/bin/env python3
"""Exact integer development gate and conditional clean report for Flame refit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from common import paired_bootstrap
from run_mica_ember_balanced import EVALS
from run_mica_flame_refit import ROOT, R1, SOURCE, SOURCE_SHA
from run_mica_page_block_refit import digest, write_json


DEFAULT_OUT = R1 / "runs/codex_flame_refit_20260928_retry1"
DEV = ("chat", "everyday")
CLEAN = ("chat_clean_val1000", "everyday_clean_val1000")


def status(out: Path, stage: str, detail: str = "") -> None:
    write_json(out / "eval_status.json", {
        "stage": stage, "detail": detail, "updated": time.time()})


def verify(out: Path) -> tuple[dict, str]:
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if (digest(SOURCE / "best.mica") != SOURCE_SHA or
            protocol["source_sha256"] != SOURCE_SHA):
        raise RuntimeError("source checkpoint or protocol changed")
    train_status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    if train_status["stage"] != "training_complete":
        raise RuntimeError("candidate native training unfinished")
    candidate_sha = digest(out / "candidate/best.mica")
    if candidate_sha != train_status["detail"]:
        raise RuntimeError("candidate checkpoint changed")
    for item in protocol["fresh_dev"].values():
        if digest(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("fresh development set changed")
    return protocol, candidate_sha


def score(out: Path, arm: str, name: str, path: Path) -> dict:
    run = SOURCE if arm == "source" else out / "candidate"
    dest = out / f"eval_{arm}_{name}.json"
    if not dest.exists():
        cmd = [sys.executable, str(R1 / "checks/eval_int.py"),
               "--run", f"{arm}={run}", "--set", "val1000",
               "--val-file", str(path), "--val-skip", "0", "--out", str(dest)]
        with (out / "eval.log").open("a", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        if rc:
            raise RuntimeError(f"exact integer {arm} {name} failed")
    row = json.loads(dest.read_text(encoding="utf-8"))[arm]
    if (row["mica"] != str(run / "best.mica") or
            row["val_file"] != str(path) or row["records"] != 1000 or
            len(row["nats"]) != 1000):
        raise RuntimeError(f"stale exact integer result: {arm} {name}")
    return row


def paired(candidate: dict, source: dict) -> list[float]:
    if candidate["counts"] != source["counts"]:
        raise RuntimeError("candidate/source records not aligned")
    return list(paired_bootstrap(candidate["nats"], source["nats"],
                                 candidate["counts"], n_boot=5000,
                                 seed=20260930))


def equal_mean(candidate: dict, source: dict) -> list[float]:
    rng = np.random.default_rng(20260930)
    draws = np.zeros(5000, dtype=np.float64)
    observed = 0.0
    for name in DEV:
        a, b = candidate[name], source[name]
        if a["counts"] != b["counts"]:
            raise RuntimeError(f"{name} records not aligned")
        counts = np.asarray(a["counts"], dtype=np.float64)
        delta = np.asarray(a["nats"], dtype=np.float64) - np.asarray(
            b["nats"], dtype=np.float64)
        observed += 0.5 * delta.sum() / counts.sum() / math.log(2)
        indexes = rng.integers(0, len(counts), (5000, len(counts)))
        draws += 0.5 * delta[indexes].sum(axis=1) / (
            counts[indexes].sum(axis=1) * math.log(2))
    low, high = np.percentile(draws, [2.5, 97.5])
    return [float(observed), float(low), float(high)]


def dev(out: Path) -> None:
    protocol, candidate_sha = verify(out)
    rows = {arm: {} for arm in ("candidate", "source")}
    for name in DEV:
        path = Path(protocol["fresh_dev"][name]["path"])
        for arm in rows:
            status(out, "evaluating_dev", f"{arm}: {name}")
            rows[arm][name] = score(out, arm, name, path)
    result = {
        "protocol": "exact exported integer C engine, bytes plus EOS from BOS; 5000 paired record bootstrap resamples; fresh disjoint development only",
        "candidate_sha256": candidate_sha, "source_sha256": SOURCE_SHA,
        "scores": {arm: {name: rows[arm][name]["bits"] for name in DEV}
                   for arm in rows},
        "targets": {name: rows["candidate"][name]["targets"] for name in DEV},
        "candidate_minus_source": {
            "per_set": {name: paired(rows["candidate"][name],
                                     rows["source"][name]) for name in DEV},
            "equal_weight_mean": equal_mean(rows["candidate"], rows["source"]),
        },
        "fresh_dev_sha256": {name: protocol["fresh_dev"][name]["sha256"]
                             for name in DEV},
    }
    write_json(out / "dev_results.json", result)
    status(out, "dev_complete")


def gate(out: Path) -> None:
    result = json.loads((out / "dev_results.json").read_text(encoding="utf-8"))
    _, candidate_sha = verify(out)
    if result["candidate_sha256"] != candidate_sha:
        raise RuntimeError("dev result belongs to another checkpoint")
    combined = result["candidate_minus_source"]["equal_weight_mean"]
    per_set = result["candidate_minus_source"]["per_set"]
    accepted = (combined[0] <= -.005 and combined[2] < 0 and
                all(per_set[name][0] <= .002 for name in DEV))
    write_json(out / "decision.json", {
        "accepted_on_fresh_development": accepted,
        "candidate_sha256": candidate_sha,
        "source_sha256": SOURCE_SHA,
        "rule": json.loads((out / "protocol.json").read_text(
            encoding="utf-8"))["selection"],
        "candidate_minus_source": result["candidate_minus_source"],
        "clean_sets": "withheld unless accepted",
    })
    status(out, "gate_complete", "accepted" if accepted else "rejected; clean withheld")


def clean(out: Path) -> None:
    decision = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    _, candidate_sha = verify(out)
    if not decision["accepted_on_fresh_development"] or \
            decision["candidate_sha256"] != candidate_sha:
        raise RuntimeError("clean evaluation blocked by development decision")
    results = {}
    for name in CLEAN:
        path = EVALS[name]
        results[name] = {"set_sha256": digest(path)}
        rows = {}
        for arm in ("candidate", "source"):
            status(out, "evaluating_clean", f"{arm}: {name}")
            rows[arm] = score(out, arm, name, path)
            results[name][arm + "_bits"] = rows[arm]["bits"]
        results[name]["targets"] = rows["candidate"]["targets"]
        results[name]["candidate_minus_source"] = paired(
            rows["candidate"], rows["source"])
        write_json(out / "clean_results.json", results)
    status(out, "clean_complete")


def generate(out: Path) -> None:
    verify(out)
    status(out, "generating")
    cmd = [sys.executable, str(R1 / "checks/eval_mica_raw_generation.py"),
           "--run", str(out / "candidate"), "--prompts",
           str(R1 / "checks/generation_dev10.json"), "--limit", "10",
           "--byte-limit", "100", "--out", str(out / "generation.json")]
    with (out / "eval.log").open("a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError("candidate generation failed")
    status(out, "generation_complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "gate", "clean", "generate"),
                        required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    try:
        {"dev": dev, "gate": gate, "clean": clean,
         "generate": generate}[args.stage](out)
    except Exception as exc:
        status(out, "failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
