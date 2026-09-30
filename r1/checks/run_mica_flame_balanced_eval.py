#!/usr/bin/env python3
"""Exact integer evaluation of F1 Flame trained on selected balanced mix A."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from common import paired_bootstrap
from compare_mica_ember_balanced import mean_dev_interval
from run_mica_ember_balanced import EVALS
from run_mica_ember_balanced_eval import SET_ORDER, score
from run_mica_page_block_refit import digest, write_json


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
OUT = R1 / "runs/codex_flame_balanced_v02a_20260928"
TRAIN = OUT / "train"
A = R1 / "runs/codex_ember_balanced_v02a_20260928"
A_SHA = "3b2a94e203fc6b0cd20365e4d5c46085035722f578161a2213b8cf148a126e72"


def status(stage: str, detail: str = "") -> None:
    write_json(OUT / "eval_status.json", {
        "stage": stage, "detail": detail, "updated": time.time()})


def main() -> None:
    if not (TRAIN / "FINISHED").is_file():
        raise RuntimeError("native Flame fit has not finished")
    launch = json.loads((OUT / "launch.json").read_text(encoding="utf-8"))
    pre = json.loads((A / "preflight.json").read_text(encoding="utf-8"))
    if (digest(R1 / "data/mix/v02a/manifest.json") !=
            launch["mix_manifest_sha256"] or
            launch["mix_manifest_sha256"] != pre["mix_manifest_sha256"] or
            digest(A / "train/best.mica") != A_SHA):
        raise RuntimeError("balanced mix or Ember A comparator changed")
    model_sha = digest(TRAIN / "best.mica")
    result_path = OUT / "eval_results.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["model_sha256"] != model_sha:
            raise RuntimeError("existing evaluation belongs to another checkpoint")
    else:
        result = {
            "model": str(TRAIN / "best.mica"),
            "model_sha256": model_sha,
            "comparator": str(A / "train/best.mica"),
            "comparator_sha256": A_SHA,
            "protocol": "exact exported integer C engine, bytes plus EOS from BOS; 5000 paired record resamples",
            "predeclared_development_gate": "equal-weight chat dev1000 and everyday dev_fresh1000 mean <= Ember A minus 0.020 bits/target with paired 95% upper CI below zero, and neither domain regresses by more than 0.005; clean sets reporting only",
            "scores": {},
        }
        write_json(result_path, result)
    for name in SET_ORDER:
        source = EVALS[name]
        if digest(source) != pre["heldout_sha256"][name]:
            raise RuntimeError(f"held-out set changed: {name}")
        status("evaluating", name)
        own = score(OUT, TRAIN, name, source)
        prior = json.loads((A / f"eval_{name}.json").read_text(
            encoding="utf-8"))["balanced"]
        if (own["counts"] != prior["counts"] or
                len(own["nats"]) != 1000):
            raise RuntimeError(f"{name} comparator records do not align")
        result["scores"][name] = {
            "set_sha256": pre["heldout_sha256"][name],
            "targets": own["targets"],
            "flame_bits": own["bits"],
            "ember_a_bits": prior["bits"],
            "flame_minus_ember_a": list(paired_bootstrap(
                own["nats"], prior["nats"], own["counts"],
                n_boot=5000, seed=20260928)),
        }
        write_json(result_path, result)
    dev = ("chat_dev1000", "everyday_dev_fresh1000")
    result["equal_weight_dev_mean_flame"] = sum(
        result["scores"][name]["flame_bits"] for name in dev) / 2
    result["equal_weight_dev_mean_ember_a"] = sum(
        result["scores"][name]["ember_a_bits"] for name in dev) / 2
    flame_rows = {name: json.loads((OUT / f"eval_{name}.json").read_text(
        encoding="utf-8"))["balanced"] for name in dev}
    ember_rows = {name: json.loads((A / f"eval_{name}.json").read_text(
        encoding="utf-8"))["balanced"] for name in dev}
    combined = mean_dev_interval(flame_rows, ember_rows)
    result["equal_weight_dev_mean_flame_minus_ember_a"] = combined
    result["development_gate_passed"] = bool(
        combined[0] <= -0.020 and combined[2] < 0 and
        all(result["scores"][name]["flame_minus_ember_a"][0] <= 0.005
            for name in dev))
    write_json(result_path, result)
    if digest(TRAIN / "best.mica") != model_sha:
        raise RuntimeError("checkpoint changed during evaluation")
    status("generating")
    cmd = [sys.executable, str(R1 / "checks/eval_mica_raw_generation.py"),
           "--run", str(TRAIN), "--prompts",
           str(R1 / "checks/generation_dev10.json"), "--limit", "10",
           "--byte-limit", "100", "--out", str(OUT / "generation.json")]
    with (OUT / "eval.log").open("a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError("raw generation failed; see eval.log")
    status("complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise
