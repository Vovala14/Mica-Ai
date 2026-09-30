#!/usr/bin/env python3
"""Short MICA lexical-state pilot with exact letter routing in its byte codes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_state_learning_pilot import ROOT, R1, SOURCE, train_args

OUT = R1 / "runs/ablation/mica_letter_route_pilot_20260927"
MODEL = OUT / "lexical_letters"
LEXICAL = R1 / "runs/ablation/mica_lexical_state_pilot_20260927/lexical2"
CONTROL = R1 / "runs/ablation/mica_window_context_pilot_20260927/window1"
PYTHON = sys.executable


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps({"stage": stage, "detail": detail,
                               "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                              indent=2), encoding="utf-8")
    os.replace(tmp, OUT / "status.json")
    print(f"[letter-route] {stage} {detail}", flush=True)


def run(stage: str, cmd: list[str], env: dict) -> None:
    status(stage)
    with (OUT / f"{stage}.log").open("a", encoding="utf-8") as stream:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{stage} exited {rc}; see {stage}.log")


def main() -> int:
    try:
        if (ROOT.parent / "STOP_MICA").exists():
            raise RuntimeError("STOP_MICA blocks training")
        if not (LEXICAL / "FINISHED").exists() or not (CONTROL / "FINISHED").exists():
            raise RuntimeError("matched short controls missing")
        info = json.loads(SOURCE.read_text(encoding="utf-8"))
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("MICA_")}
        env.update({k: str(v) for k, v in info["env"].items()})
        env["PYTHONUTF8"] = "1"
        if not (MODEL / "FINISHED").exists():
            cmd = train_args(info["args"], MODEL, "fit", 3)
            cmd[cmd.index("--rule-state") + 1] = "lexical2-letters"
            run("train_letters", cmd, env)
        clean = R1 / "data/eval_clean/val1000.jsonl"
        if not (OUT / "clean_val128.json").exists():
            run("eval_clean", [PYTHON, str(R1 / "checks/eval_int.py"),
                "--run", f"control={CONTROL}",
                "--run", f"lexical2={LEXICAL}",
                "--run", f"letters={MODEL}",
                "--set", "val128", "--val-file", str(clean),
                "--val-skip", "0", "--out", str(OUT / "clean_val128.json")], env)
        context = OUT / "lexical_context.json"
        if not context.exists():
            run("lexical_context", [PYTHON,
                str(R1 / "checks/lexical_context_probe.py"),
                "--run", str(MODEL), "--out", str(context)], env)
        generation = OUT / "generation.json"
        if not generation.exists():
            run("generation", [PYTHON,
                str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(MODEL),
                "--prompts", str(R1 / "checks/generation_dev10.json"),
                "--out", str(generation), "--limit", "10",
                "--byte-limit", "100"], env)
        status("complete", str(OUT / "clean_val128.json"))
        return 0
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
