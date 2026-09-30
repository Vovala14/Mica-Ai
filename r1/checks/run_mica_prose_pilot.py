#!/usr/bin/env python3
"""One full MICA fit on deduplicated English prose, with matched evaluations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_state_learning_pilot import ROOT, R1, SOURCE, train_args

OUT = R1 / "runs/ablation/mica_prose_pilot_20260927"
FIT = OUT / "fit"
BASELINE = R1 / "runs/overnight/mica_original_20260927/samples/plus_2h"
PROSE = R1 / "data/prose_v1_320"
PYTHON = sys.executable


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps({"stage": stage, "detail": detail,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, OUT / "status.json")
    print(f"[prose-pilot] {stage} {detail}", flush=True)


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
        if not (BASELINE / "best.mica").exists():
            raise RuntimeError("original MICA baseline missing")
        if not (FIT / "FINISHED").exists():
            cmd = train_args(info["args"], FIT, "fit", 1)
            values = {"records": str(PROSE / "train.jsonl"),
                      "val-records": str(PROSE / "val.jsonl"),
                      "rule-state": "none", "rule-scoring": "balance",
                      "fit-records": "40000", "fit-steps": "1500",
                      "val-records-n": "256", "warmup": "200",
                      "save-every": "1"}
            for key, value in values.items():
                cmd[cmd.index("--" + key) + 1] = value
            run("train", cmd, env)
        prose_eval = OUT / "prose_val1000.json"
        if not prose_eval.exists():
            run("eval_prose", [PYTHON, str(R1 / "checks/eval_int.py"),
                "--run", f"baseline={BASELINE}",
                "--run", f"prose={FIT}", "--set", "val1000",
                "--val-file", str(PROSE / "val.jsonl"),
                "--val-skip", "256", "--out", str(prose_eval)], env)
        everyday_eval = OUT / "everyday_val1000.json"
        if not everyday_eval.exists():
            run("eval_everyday", [PYTHON, str(R1 / "checks/eval_int.py"),
                "--run", f"prose={FIT}", "--set", "val1000",
                "--val-file", str(R1 / "data/eval_clean/val1000.jsonl"),
                "--val-skip", "0", "--out", str(everyday_eval)], env)
        generation = OUT / "generation.json"
        if not generation.exists():
            run("generation", [PYTHON,
                str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(FIT),
                "--prompts", str(R1 / "checks/generation_dev10.json"),
                "--out", str(generation), "--limit", "10",
                "--byte-limit", "100"], env)
        pval = json.loads(prose_eval.read_text(encoding="utf-8"))
        eval_everyday = json.loads(everyday_eval.read_text(encoding="utf-8"))
        rows = json.loads(generation.read_text(encoding="utf-8"))["rows"]
        lines = ["# Original MICA on deduplicated English prose", "",
                 "One 40,000-record/1,500-step full fit with the same original "
                 "integer engine, geometry and random-balanced rule selection. "
                 "Only the training corpus differs. The prose corpus has "
                 "document/record deduplication and filters for math and "
                 "boilerplate.", "",
                 f"Prose validation (1,000 records): baseline "
                 f"**{pval['baseline']['bits']:.6f}**, trained on prose "
                 f"**{pval['prose']['bits']:.6f}** bits/target.",
                 f"Everyday clean validation (1,000 records): trained on "
                 f"prose **{eval_everyday['prose']['bits']:.6f}** bits/target.",
                 "", "## Direct generation from prose-trained MICA", ""]
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
