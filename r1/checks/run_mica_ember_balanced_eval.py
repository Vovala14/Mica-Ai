#!/usr/bin/env python3
"""Evaluate a completed balanced Ember with the exact exported integer engine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_mica_ember_balanced import EVALS, paths
from run_mica_page_block_refit import digest, write_json


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
SET_ORDER = (
    "chat_dev1000", "everyday_dev_fresh1000",
    "chat_clean_val1000", "everyday_clean_val1000",
)


def status(out: Path, stage: str, detail: str = "") -> None:
    write_json(out / "eval_status.json", {
        "stage": stage, "detail": detail, "updated": time.time()})


def score(out: Path, train: Path, name: str, source: Path) -> dict:
    destination = out / f"eval_{name}.json"
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        row = payload.get("balanced")
        if (row is None or row.get("mica") != str(train / "best.mica") or
                row.get("val_file") != str(source) or row.get("records") != 1000):
            raise RuntimeError(f"existing {name} evaluation has different inputs")
        return row
    cmd = [sys.executable, str(R1 / "checks/eval_int.py"),
           "--run", f"balanced={train}", "--set", "val1000",
           "--val-file", str(source), "--val-skip", "0",
           "--out", str(destination)]
    with (out / "eval.log").open("a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode or not destination.exists():
        raise RuntimeError(f"exact integer {name} failed; see eval.log")
    row = json.loads(destination.read_text(encoding="utf-8")).get("balanced")
    if (row is None or row.get("mica") != str(train / "best.mica") or
            row.get("val_file") != str(source) or row.get("records") != 1000 or
            len(row.get("nats", [])) != 1000 or
            len(row.get("counts", [])) != 1000):
        raise RuntimeError(f"exact integer {name} returned invalid records")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("v02a", "v02b"), required=True)
    args = parser.parse_args()
    mix, out, train = paths(args.variant)
    if not (train / "FINISHED").is_file():
        raise RuntimeError("balanced Ember native fit has not finished")
    pre = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
    if (pre["variant"] != args.variant or
            digest(mix / "manifest.json") != pre["mix_manifest_sha256"]):
        raise RuntimeError("balanced mixture changed since preflight")
    sha = digest(train / "best.mica")
    result_path = out / "eval_results.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["model_sha256"] != sha or result["variant"] != args.variant:
            raise RuntimeError("existing evaluation belongs to another checkpoint")
    else:
        result = {
            "variant": args.variant,
            "model": str(train / "best.mica"), "model_sha256": sha,
            "mix_manifest_sha256": pre["mix_manifest_sha256"],
            "protocol": "exact exported integer C engine; pooled bytes plus EOS from BOS",
            "selection": "equal-weight mean of chat_dev1000 and everyday_dev_fresh1000 bits/target; clean sets for reporting only",
            "scores": {},
        }
        write_json(result_path, result)
    for name in SET_ORDER:
        source = EVALS[name]
        if digest(source) != pre["heldout_sha256"][name]:
            raise RuntimeError(f"held-out set changed: {name}")
        status(out, "evaluating", name)
        row = score(out, train, name, source)
        result["scores"][name] = {
            "set_sha256": pre["heldout_sha256"][name],
            "bits": row["bits"], "records": row["records"],
            "targets": row["targets"],
        }
        write_json(result_path, result)
    result["selection_mean_dev_bits"] = (
        result["scores"]["chat_dev1000"]["bits"] +
        result["scores"]["everyday_dev_fresh1000"]["bits"]) / 2
    write_json(result_path, result)
    if digest(train / "best.mica") != sha:
        raise RuntimeError("balanced checkpoint changed during evaluation")
    status(out, "generating")
    cmd = [sys.executable, str(R1 / "checks/eval_mica_raw_generation.py"),
           "--run", str(train), "--prompts",
           str(R1 / "checks/generation_dev10.json"), "--limit", "10",
           "--byte-limit", "100", "--out", str(out / "generation.json")]
    with (out / "eval.log").open("a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError("raw generation failed; see eval.log")
    status(out, "complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--variant")
        variant, _ = parser.parse_known_args()
        if variant.variant in ("v02a", "v02b"):
            _, output, _ = paths(variant.variant)
            status(output, "failed", f"{type(exc).__name__}: {exc}")
        raise
