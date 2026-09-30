#!/usr/bin/env python3
"""Matched native-MICA reversible-state preflight against random stream state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_state_learning_pilot import FIT as STREAM, ROOT, R1, SOURCE, train_args

OUT = R1 / "runs/ablation/mica_reversible_state_pilot_20260927"
REV = OUT / "reversible"
PYTHON = sys.executable


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps({"stage": stage, "detail": detail,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, OUT / "status.json")
    print(f"[reversible-pilot] {stage} {detail}", flush=True)


def run(stage: str, command: list[str], env: dict) -> None:
    status(stage)
    with (OUT / f"{stage}.log").open("a", encoding="utf-8") as stream:
        rc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{stage} exited {rc}; see {stage}.log")


def report() -> None:
    evals = json.loads((OUT / "clean_val128.json").read_text(encoding="utf-8"))
    horizons = {}
    for key in ("stream", "reversible"):
        rows = json.loads((OUT / f"horizon_{key}.json").read_text(encoding="utf-8"))["rows"]
        horizons[key] = {r["shared_suffix_bytes"]: r["exactly_equal_pairs"] for r in rows}
    generations = json.loads((OUT / "generation_reversible.json").read_text(
        encoding="utf-8"))["rows"]
    lines = ["# MICA reversible-rule state preflight", "",
             "Both arms use the original Minimal Inference Cellular Automaton "
             "engine, geometry, corpus, 500-record/30-step fit rounds and "
             "three-round budget. The stream control uses random state rules; "
             "the reversible arm uses exact integer-rule selection among 243 "
             "ternary state codes. Its phase-1 rule values and readout are "
             "learned from text by the existing fitter.", "",
             f"Clean validation (128 fixed records): stream "
             f"**{evals['stream']['bits']:.6f}**, reversible "
             f"**{evals['reversible']['bits']:.6f}** bits/target.", "",
             "| Common suffix | Stream equal-score pairs | Reversible equal-score pairs |",
             "|---:|---:|---:|"]
    for length in (8, 16, 32, 64, 128):
        lines.append(f"| {length} bytes | {horizons['stream'][length]}/6 | "
                     f"{horizons['reversible'][length]}/6 |")
    lines += ["", "## Direct generations from reversible MICA", ""]
    for row in generations:
        lines.append(f"- `{row['prompt']}` → `{row['full_text']}`")
    lines += ["", "This is a structural short pilot. Release decisions require "
              "a full-budget fit, clean val1000, and readable novel sentences.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    try:
        info = json.loads(SOURCE.read_text(encoding="utf-8"))
        env = dict(os.environ)
        env.update({k: str(v) for k, v in info["env"].items()})
        env["PYTHONUTF8"] = "1"
        if (ROOT.parent / "STOP_MICA").exists():
            raise RuntimeError("STOP_MICA blocks training")
        if not (STREAM / "FINISHED").exists():
            raise RuntimeError("matched stream control is missing")
        OUT.mkdir(parents=True, exist_ok=True)
        if not (REV / "FINISHED").exists():
            cmd = train_args(info["args"], REV, "fit", 3)
            index = cmd.index("--rule-state")
            cmd[index + 1] = "reversible"
            run("train_reversible", cmd, env)
        clean = R1 / "data/eval_clean/val1000.jsonl"
        if not (OUT / "clean_val128.json").exists():
            run("eval_clean", [PYTHON, str(R1 / "checks/eval_int.py"),
                "--run", f"stream={STREAM}", "--run", f"reversible={REV}",
                "--set", "val128", "--val-file", str(clean),
                "--val-skip", "0", "--out", str(OUT / "clean_val128.json")], env)
        for key, folder in (("stream", STREAM), ("reversible", REV)):
            path = OUT / f"horizon_{key}.json"
            if not path.exists():
                run("horizon_" + key, [PYTHON, str(R1 / "checks/context_horizon.py"),
                    "--run", str(folder), "--out", str(path)], env)
        if not (OUT / "generation_reversible.json").exists():
            run("generation_reversible", [PYTHON,
                str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(REV),
                "--prompts", str(R1 / "checks/generation_dev10.json"),
                "--out", str(OUT / "generation_reversible.json"),
                "--limit", "10", "--byte-limit", "100"], env)
        report()
        status("complete", str(OUT / "REPORT.md"))
        return 0
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
