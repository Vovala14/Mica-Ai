#!/usr/bin/env python3
"""One full-budget native MICA fit of the reversible rule-state candidate."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_state_learning_pilot import ROOT, R1, SOURCE, train_args

OUT = R1 / "runs/ablation/mica_reversible_full_20260927"
FIT = OUT / "fit"
PYTHON = sys.executable


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps({"stage": stage, "detail": detail,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, OUT / "status.json")
    print(f"[reversible-full] {stage} {detail}", flush=True)


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
        if not (FIT / "FINISHED").exists():
            cmd = train_args(info["args"], FIT, "fit", 1)
            values = {"rule-state": "reversible", "fit-records": "40000",
                      "fit-steps": "1500", "val-records-n": "256",
                      "warmup": "200", "save-every": "1"}
            for key, value in values.items():
                cmd[cmd.index("--" + key) + 1] = value
            run("train", cmd, env)
        clean = R1 / "data/eval_clean/val1000.jsonl"
        result = OUT / "clean_val1000.json"
        if not result.exists():
            run("eval_clean", [PYTHON, str(R1 / "checks/eval_int.py"),
                "--run", f"reversible={FIT}", "--set", "val1000",
                "--val-file", str(clean), "--val-skip", "0",
                "--out", str(result)], env)
        horizon = OUT / "horizon.json"
        if not horizon.exists():
            run("horizon", [PYTHON, str(R1 / "checks/context_horizon.py"),
                "--run", str(FIT), "--out", str(horizon)], env)
        generation = OUT / "generation.json"
        if not generation.exists():
            run("generation", [PYTHON,
                str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(FIT),
                "--prompts", str(R1 / "checks/generation_dev10.json"),
                "--out", str(generation), "--limit", "10",
                "--byte-limit", "100"], env)
        rows = json.loads(generation.read_text(encoding="utf-8"))["rows"]
        value = json.loads(result.read_text(encoding="utf-8"))["reversible"]["bits"]
        lines = ["# Full-budget reversible MICA rule pilot", "",
                 "Same original MICA engine and rule book; one round of 40,000 "
                 "records and 1,500 fit steps, after the normal full init.", "",
                 f"Clean validation (1,000 records): **{value:.6f} bits/target**.",
                 "", "## Direct generation", ""]
        lines.extend(f"- `{row['prompt']}` → `{row['full_text']}`" for row in rows)
        (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        status("complete", str(OUT / "REPORT.md"))
        return 0
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
