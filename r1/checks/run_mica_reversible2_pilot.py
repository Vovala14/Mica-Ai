#!/usr/bin/env python3
"""Matched short pilot of two reversible state tracks in native MICA."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_state_learning_pilot import ROOT, R1, SOURCE, train_args

OUT = R1 / "runs/ablation/mica_reversible2_pilot_20260927"
ONE = R1 / "runs/ablation/mica_reversible_state_pilot_20260927/reversible"
TWO = OUT / "two_track"
PYTHON = sys.executable


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps({"stage": stage, "detail": detail,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, OUT / "status.json")
    print(f"[reversible2] {stage} {detail}", flush=True)


def run(stage: str, command: list[str], env: dict) -> None:
    status(stage)
    with (OUT / f"{stage}.log").open("a", encoding="utf-8") as stream:
        rc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{stage} exited {rc}; see {stage}.log")


def main() -> int:
    try:
        info = json.loads(SOURCE.read_text(encoding="utf-8"))
        env = dict(os.environ)
        env.update({k: str(v) for k, v in info["env"].items()})
        env["PYTHONUTF8"] = "1"
        if (ROOT.parent / "STOP_MICA").exists():
            raise RuntimeError("STOP_MICA blocks training")
        if not (ONE / "FINISHED").exists():
            raise RuntimeError("matched one-track control missing")
        if not (TWO / "FINISHED").exists():
            cmd = train_args(info["args"], TWO, "fit", 3)
            cmd[cmd.index("--rule-state") + 1] = "reversible2"
            run("train", cmd, env)
        clean = OUT / "clean_val128.json"
        if not clean.exists():
            run("eval_clean", [PYTHON, str(R1 / "checks/eval_int.py"),
                "--run", f"one={ONE}", "--run", f"two={TWO}",
                "--set", "val128", "--val-file",
                str(R1 / "data/eval_clean/val1000.jsonl"),
                "--val-skip", "0", "--out", str(clean)], env)
        horizon = OUT / "horizon_two.json"
        if not horizon.exists():
            run("horizon", [PYTHON, str(R1 / "checks/context_horizon.py"),
                "--run", str(TWO), "--out", str(horizon)], env)
        generation = OUT / "generation_two.json"
        if not generation.exists():
            run("generation", [PYTHON,
                str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(TWO),
                "--prompts", str(R1 / "checks/generation_dev10.json"),
                "--out", str(generation), "--limit", "10",
                "--byte-limit", "100"], env)
        val = json.loads(clean.read_text(encoding="utf-8"))
        rows = json.loads(generation.read_text(encoding="utf-8"))["rows"]
        hz = json.loads(horizon.read_text(encoding="utf-8"))["rows"]
        lines = ["# Two reversible state tracks in native MICA", "",
                 "Both arms use the same original engine, 768 cells, 112 "
                 "channels, 16 phases, 256 candidates per page, same corpus "
                 "and three 500-record/30-step fit rounds. The two-track arm "
                 "allocates two of the existing VSET phase groups to "
                 "independent reversible integer rules and uses phases 2 and "
                 "3 to learn features from their states.", "",
                 f"Clean validation: one track **{val['one']['bits']:.6f}**, "
                 f"two tracks **{val['two']['bits']:.6f}** bits/target "
                 "(128 fixed records).", "",
                 "| Shared suffix | Equal-score pairs, two tracks |",
                 "|---:|---:|"]
        for row in hz:
            if row["shared_suffix_bytes"] in (8, 16, 32, 64, 128):
                lines.append(f"| {row['shared_suffix_bytes']} bytes | "
                             f"{row['exactly_equal_pairs']}/6 |")
        lines += ["", "## Direct generation", ""]
        lines.extend(f"- `{row['prompt']}` → `{row['full_text']}`" for row in rows)
        (OUT / "REPORT.md").write_text("\n".join(lines) + "\n",
                                       encoding="utf-8")
        status("complete", str(OUT / "REPORT.md"))
        return 0
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
