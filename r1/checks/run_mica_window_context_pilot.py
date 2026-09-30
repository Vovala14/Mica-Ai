#!/usr/bin/env python3
"""Matched native-MICA window 1 versus window 2 context pilot."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_state_learning_pilot import ROOT, R1, SOURCE, train_args

OUT = R1 / "runs/ablation/mica_window_context_pilot_20260927"
PYTHON = sys.executable


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps({"stage": stage, "detail": detail,
                               "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                              indent=2), encoding="utf-8")
    os.replace(tmp, OUT / "status.json")
    print(f"[window-pilot] {stage} {detail}", flush=True)


def run(stage: str, cmd: list[str], env: dict) -> None:
    status(stage)
    with (OUT / f"{stage}.log").open("a", encoding="utf-8") as stream:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{stage} exited {rc}; see {stage}.log")


def main() -> int:
    try:
        info = json.loads(SOURCE.read_text(encoding="utf-8"))
        if (ROOT.parent / "STOP_MICA").exists():
            raise RuntimeError("STOP_MICA blocks training")
        for window in (1, 2):
            folder = OUT / f"window{window}"
            if (folder / "FINISHED").exists():
                continue
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("MICA_")}
            env.update({k: str(v) for k, v in info["env"].items()})
            env["MICA_WINDOW"] = str(window)
            env["PYTHONUTF8"] = "1"
            cmd = train_args(info["args"], folder, "fit", 3)
            cmd[cmd.index("--rule-state") + 1] = "none"
            run(f"train_window{window}", cmd, env)
        clean = R1 / "data/eval_clean/val1000.jsonl"
        if not (OUT / "clean_val128.json").exists():
            run("eval_clean", [PYTHON, str(R1 / "checks/eval_int.py"),
                "--run", f"window1={OUT / 'window1'}",
                "--run", f"window2={OUT / 'window2'}",
                "--set", "val128", "--val-file", str(clean),
                "--val-skip", "0", "--out", str(OUT / "clean_val128.json")],
                dict(os.environ))
        for window in (1, 2):
            horizon = OUT / f"horizon_window{window}.json"
            if not horizon.exists():
                run(f"horizon_window{window}", [PYTHON,
                    str(R1 / "checks/context_horizon.py"),
                    "--run", str(OUT / f"window{window}"),
                    "--out", str(horizon)], dict(os.environ))
        gen = OUT / "generation_window2.json"
        if not gen.exists():
            run("generation_window2", [PYTHON,
                str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(OUT / "window2"),
                "--prompts", str(R1 / "checks/generation_dev10.json"),
                "--out", str(gen), "--limit", "10", "--byte-limit", "100"],
                dict(os.environ))
        status("complete", str(OUT / "clean_val128.json"))
        return 0
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
