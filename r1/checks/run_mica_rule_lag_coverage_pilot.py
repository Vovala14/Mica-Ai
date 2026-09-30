#!/usr/bin/env python3
"""Matched native-MICA pilot for longer, distinct rule-scoring lags."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_overnight import train_command, write_atomic

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
PYTHON = sys.executable
SOURCE = R1 / "runs/ablation/mica_probe64_lag64_full_round1"
OUT = R1 / "runs/codex_rule_lag_20260927"
LONG_LAGS = ",".join(["6"] * 16)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def status(phase: str, detail: str = "") -> None:
    write_atomic(OUT / "status.json", json.dumps({
        "phase": phase, "detail": detail, "updated": time.time(),
    }, indent=2) + "\n")


def run(label: str, cmd: list[str], env: dict[str, str]) -> None:
    with (OUT / f"{label}.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"{label} failed ({proc.returncode}); read {label}.log")


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"refusing nonempty output directory {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    if info["env"]["MICA_PROBE_WINDOW"] != "64" or info["args"]["steps"] != 1:
        raise SystemExit("source must be the one-round native lag-64 control")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("MICA_")}
    env.update(info["env"])
    env["PYTHONUTF8"] = "1"
    write_atomic(OUT / "protocol.json", json.dumps({
        "source": str(SOURCE), "source_sha256": digest(SOURCE / "best.mica"),
        "rule_max_back": LONG_LAGS, "arms": ["random_long", "covered_long"],
        "training": "same everyday records, seed, one round, 40000 fit records, 1500 steps",
        "clean_validation": "r1/data/eval_clean/val1000.jsonl; exact integer engine",
    }, indent=2) + "\n")
    try:
        for arm, covered in (("random_long", False), ("covered_long", True)):
            status("training", arm)
            run_dir = OUT / arm
            cmd = train_command(info, run_dir, 1)
            cmd[cmd.index("--rule-max-back") + 1] = LONG_LAGS
            cmd.extend(["--tape-lags", info["args"]["tape_lags"]])
            if covered:
                cmd.append("--rule-lag-coverage")
            run(arm + "_train", cmd, env)
            if not (run_dir / "FINISHED").exists():
                raise RuntimeError(f"{arm} did not finish")
            status("evaluating", arm)
            run(arm + "_eval", [PYTHON, str(R1 / "checks/eval_int.py"),
                "--run", f"{arm}={run_dir}", "--set", "val1000",
                "--val-file", str(R1 / "data/eval_clean/val1000.jsonl"),
                "--val-skip", "0", "--out", str(OUT / f"{arm}_val1000.json")],
                env)
            run(arm + "_generation", [PYTHON,
                str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(run_dir), "--prompts",
                str(R1 / "checks/generation_dev10.json"),
                "--limit", "10", "--byte-limit", "100", "--out",
                str(OUT / f"{arm}_generation.json")], env)
        sys.path.insert(0, str(R1 / "checks"))
        from common import paired_bootstrap
        rows = {}
        for arm in ("random_long", "covered_long"):
            rows[arm] = json.loads((OUT / f"{arm}_val1000.json").read_text(
                encoding="utf-8"))[arm]
        random, covered = rows["random_long"], rows["covered_long"]
        if random["counts"] != covered["counts"]:
            raise AssertionError("validation target counts differ")
        difference, lower, upper = paired_bootstrap(
            covered["nats"], random["nats"], covered["counts"])
        results = {
            "random_long_bits": random["bits"],
            "covered_long_bits": covered["bits"],
            "covered_minus_random": difference,
            "paired_95ci": [lower, upper],
            "models": {arm: {"path": str(OUT / arm / "best.mica"),
                             "sha256": digest(OUT / arm / "best.mica")}
                       for arm in rows},
        }
        write_atomic(OUT / "results.json", json.dumps(results, indent=2) + "\n")
        status("complete", str(OUT / "results.json"))
        print(json.dumps(results, indent=2), flush=True)
        return 0
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
