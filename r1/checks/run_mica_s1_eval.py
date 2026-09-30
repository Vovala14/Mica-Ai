#!/usr/bin/env python3
"""Development-only S1 acceptance, then conditional clean integer reporting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from common import paired_bootstrap
from compare_mica_ember_balanced import mean_dev_interval
from run_mica_ember_balanced import EVALS
from run_mica_page_block_refit import digest, write_json
from run_mica_s1_pilot import ARMS, MIX, OUT, verify


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
DEV = ("chat_dev1000", "everyday_dev_fresh1000")
CLEAN = ("chat_clean_val1000", "everyday_clean_val1000")
PROBES = ("chat_long600", "everyday_long600")


def status(stage: str, detail: str = "") -> None:
    write_json(OUT / "eval_status.json", {
        "stage": stage, "detail": detail, "updated": time.time()})


def model_path(arm: str) -> Path:
    path = OUT / arm / "train"
    if not (path / "FINISHED").is_file():
        raise RuntimeError(f"{arm} native fit unfinished")
    return path


def score(arm: str, name: str) -> dict:
    run = model_path(arm)
    path = OUT / f"eval_{arm}_{name}.json"
    if not path.exists():
        cmd = [sys.executable, str(R1 / "checks/eval_int.py"),
               "--run", f"{arm}={run}", "--set", "val1000",
               "--val-file", str(EVALS[name]), "--val-skip", "0",
               "--out", str(path)]
        with (OUT / "eval.log").open("a", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        if rc:
            raise RuntimeError(f"{arm} {name} integer evaluation failed")
    payload = json.loads(path.read_text(encoding="utf-8"))[arm]
    if (payload["mica"] != str(run / "best.mica") or
            payload["val_file"] != str(EVALS[name]) or
            payload["records"] != 1000 or len(payload["nats"]) != 1000):
        raise RuntimeError(f"{arm} {name} integer evaluation is stale")
    return payload


def dev() -> None:
    if json.loads((OUT / "status.json").read_text(
            encoding="utf-8"))["stage"] != "training_complete":
        raise RuntimeError("matched arms have not all finished")
    rows = {arm: {} for arm in ARMS}
    for name in DEV:
        for arm in ARMS:
            status("evaluating_dev", f"{arm}: {name}")
            rows[arm][name] = score(arm, name)
    result = {
        "protocol": "exact exported integer C engine, bytes plus EOS from BOS; 5000 paired record bootstrap resamples; clean sets untouched",
        "model_sha256": {arm: digest(model_path(arm) / "best.mica") for arm in ARMS},
        "sets_sha256": {name: digest(EVALS[name]) for name in DEV},
        "scores": {arm: {name: rows[arm][name]["bits"] for name in DEV}
                   for arm in ARMS},
        "comparisons": {},
    }
    for prior in ("f1_control", "state_only"):
        per_set = {}
        for name in DEV:
            candidate, old = rows["s1"][name], rows[prior][name]
            if candidate["counts"] != old["counts"]:
                raise RuntimeError("paired development records not aligned")
            per_set[name] = list(paired_bootstrap(
                candidate["nats"], old["nats"], candidate["counts"],
                n_boot=5000, seed=20260928))
        result["comparisons"][f"s1_minus_{prior}"] = {
            "per_set": per_set,
            "equal_weight_mean": mean_dev_interval(rows["s1"], rows[prior]),
        }
    write_json(OUT / "dev_results.json", result)
    status("dev_complete")


def context() -> None:
    for arm in ("state_only", "s1"):
        run = model_path(arm)
        for probe in PROBES:
            out = OUT / f"context_{arm}_{probe}.json"
            if out.exists():
                row = json.loads(out.read_text(encoding="utf-8"))
                if row["model_sha256"] != digest(run / "best.mica"):
                    raise RuntimeError(f"stale {arm} {probe} context result")
                continue
            status("evaluating_context", f"{arm}: {probe}")
            cmd = [sys.executable, str(R1 / "checks/run_mica_context_use_int.py"),
                   "--run", str(run), "--probe", probe, "--out", str(out)]
            with (OUT / "context.log").open("a", encoding="utf-8") as log:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
            if rc:
                raise RuntimeError(f"{arm} {probe} context-use evaluation failed")
    status("context_complete")


def gate() -> None:
    dev_result = json.loads((OUT / "dev_results.json").read_text(encoding="utf-8"))
    vs_f1 = dev_result["comparisons"]["s1_minus_f1_control"]["equal_weight_mean"]
    vs_state = dev_result["comparisons"]["s1_minus_state_only"]["equal_weight_mean"]
    ctx = {arm: {} for arm in ("state_only", "s1")}
    for arm in ctx:
        for probe in PROBES:
            row = json.loads((OUT / f"context_{arm}_{probe}.json").read_text(
                encoding="utf-8"))
            if row["model_sha256"] != dev_result["model_sha256"][arm]:
                raise RuntimeError("context-use result checkpoint mismatch")
            ctx[arm][probe] = {k: row["k"][k] for k in ("8", "16")}
    gate_result = {
        "loss_s1_minus_f1": vs_f1,
        "loss_s1_minus_state_only": vs_state,
        "context_use": ctx,
        "loss_f1_noninferiority": vs_f1[0] <= .005 and vs_f1[2] <= .005,
        "loss_interaction_gain": vs_state[0] < 0 and vs_state[2] < 0,
        "use8_gate": all(ctx["s1"][probe]["8"]["use_bits"] >= .010
                         for probe in PROBES),
        "use16": "reported only; not a gate for a one-word state",
    }
    gate_result["accepted_on_development"] = all(
        gate_result[k] for k in ("loss_f1_noninferiority",
                                 "loss_interaction_gain", "use8_gate"))
    write_json(OUT / "gate.json", gate_result)
    status("gate_complete", "accepted" if gate_result["accepted_on_development"]
           else "rejected; clean sets withheld")


def clean() -> None:
    decision = json.loads((OUT / "gate.json").read_text(encoding="utf-8"))
    if not decision["accepted_on_development"]:
        raise RuntimeError("S1 failed development gate; clean sets withheld")
    results = {arm: {} for arm in ARMS}
    for name in CLEAN:
        for arm in ARMS:
            status("evaluating_clean", f"{arm}: {name}")
            row = score(arm, name)
            results[arm][name] = {"bits": row["bits"],
                                  "targets": row["targets"],
                                  "set_sha256": digest(EVALS[name])}
    write_json(OUT / "clean_results.json", results)
    status("generating")
    cmd = [sys.executable, str(R1 / "checks/eval_mica_raw_generation.py"),
           "--run", str(model_path("s1")), "--prompts",
           str(R1 / "checks/generation_dev10.json"), "--limit", "10",
           "--byte-limit", "100", "--out", str(OUT / "generation_s1.json")]
    with (OUT / "eval.log").open("a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError("S1 raw generation failed")
    status("complete")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("dev", "context", "gate", "clean"),
                    required=True)
    args = ap.parse_args()
    verify()
    {"dev": dev, "context": context, "gate": gate,
     "clean": clean}[args.stage]()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise
