#!/usr/bin/env python3
"""Matched MICA rule-choice pilot: context templates vs random balanced rules."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_state_learning_pilot import ROOT, R1, SOURCE, train_args

OUT = R1 / "runs/ablation/mica_template_rule_pilot_20260927"
ARMS = {"random": OUT / "random_balanced",
        "template": OUT / "template_balanced"}
PYTHON = sys.executable


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps({"stage": stage, "detail": detail,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, OUT / "status.json")
    print(f"[template-rules] {stage} {detail}", flush=True)


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
        for key, folder in ARMS.items():
            if (folder / "FINISHED").exists():
                continue
            cmd = train_args(info["args"], folder, "fit", 3)
            values = {"rule-state": "none", "rule-scoring":
                      "balance" if key == "random" else "template+balance",
                      "fit-records": "1000", "fit-steps": "100"}
            for flag, value in values.items():
                cmd[cmd.index("--" + flag) + 1] = value
            run("train_" + key, cmd, env)
        clean = OUT / "clean_val128.json"
        if not clean.exists():
            run("eval_clean", [PYTHON, str(R1 / "checks/eval_int.py"),
                "--run", f"random={ARMS['random']}",
                "--run", f"template={ARMS['template']}", "--set", "val128",
                "--val-file", str(R1 / "data/eval_clean/val1000.jsonl"),
                "--val-skip", "0", "--out", str(clean)], env)
        generation = OUT / "generation_template.json"
        if not generation.exists():
            run("generation", [PYTHON,
                str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(ARMS["template"]),
                "--prompts", str(R1 / "checks/generation_dev10.json"),
                "--out", str(generation), "--limit", "10",
                "--byte-limit", "100"], env)
        val = json.loads(clean.read_text(encoding="utf-8"))
        rows = json.loads(generation.read_text(encoding="utf-8"))["rows"]
        lines = ["# Data-driven selection of original MICA integer rules", "",
                 "Both arms use the same original integer engine, geometry, "
                 "corpus and three 1,000-record/100-step fit rounds. The control "
                 "uses random score selectors followed by occupancy balancing. "
                 "The treatment chooses score terms from frequent byte contexts "
                 "of each rule page, then balances only the eligible candidates. "
                 "The template construction uses the current-byte cell for all "
                 "16 phases, matching this run's one-cell execution window.", "",
                 f"Clean validation: random **{val['random']['bits']:.6f}**, "
                 f"template **{val['template']['bits']:.6f}** bits/target "
                 "(128 fixed records).", "", "## Direct template generation", ""]
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
