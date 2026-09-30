#!/usr/bin/env python3
"""Choose which original MICA rule phases receive corpus-selected templates."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_state_learning_pilot import ROOT, R1, SOURCE, train_args

OUT = R1 / "runs/ablation/mica_template_phase_search_20260927"
PILOT = R1 / "runs/ablation/mica_template_rule_pilot_20260927"
ARMS = {
    "random": (PILOT / "random_balanced", None),
    "all": (PILOT / "template_balanced", "all"),
    "back1": (OUT / "back1", "0,8"),
    "short": (OUT / "short", "0,1,2,4,8,9,12"),
    "early": (OUT / "early", "0,1,2,3"),
}
PYTHON = sys.executable


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps({"stage": stage, "detail": detail,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, OUT / "status.json")
    print(f"[phase-search] {stage} {detail}", flush=True)


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
        for key, (folder, phases) in ARMS.items():
            if (folder / "FINISHED").exists():
                continue
            if phases is None:
                raise RuntimeError("matched random control missing")
            cmd = train_args(info["args"], folder, "fit", 3)
            values = {"rule-state": "none",
                      "rule-scoring": "template+balance",
                      "fit-records": "1000", "fit-steps": "100"}
            for flag, value in values.items():
                cmd[cmd.index("--" + flag) + 1] = value
            cmd += ["--template-phases", phases]
            run("train_" + key, cmd, env)
        scores = {}
        for key, (folder, _) in ARMS.items():
            progress = json.loads((folder / "progress.json").read_text(
                encoding="utf-8"))
            scores[key] = min(float(row["val_bits"])
                              for row in progress["history"])
        selected = min(ARMS, key=lambda key: scores[key])
        status("selected", f"{selected}: trainer val {scores[selected]:.4f}")
        clean = OUT / "clean_val128.json"
        if not clean.exists():
            cmd = [PYTHON, str(R1 / "checks/eval_int.py")]
            for key, (folder, _) in ARMS.items():
                cmd += ["--run", f"{key}={folder}"]
            cmd += ["--set", "val128", "--val-file",
                    str(R1 / "data/eval_clean/val1000.jsonl"),
                    "--val-skip", "0", "--out", str(clean)]
            run("eval_clean", cmd, env)
        generation = OUT / "generation_selected.json"
        if not generation.exists():
            run("generation", [PYTHON,
                str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(ARMS[selected][0]),
                "--prompts", str(R1 / "checks/generation_dev10.json"),
                "--out", str(generation), "--limit", "10",
                "--byte-limit", "100"], env)
        val = json.loads(clean.read_text(encoding="utf-8"))
        rows = json.loads(generation.read_text(encoding="utf-8"))["rows"]
        lines = ["# Phase-selective MICA integer-rule templates", "",
                 "The original MICA engine, geometry, corpus and three "
                 "1,000-record/100-step fit rounds are matched. Only the "
                 "selected rule-page phases receive frequency templates; "
                 "the others keep their random balanced scoring rules.", "",
                 "| Template phases | Trainer validation | Clean validation |",
                 "|---|---:|---:|"]
        for key in ARMS:
            lines.append(f"| {key} | {scores[key]:.4f} | "
                         f"{val[key]['bits']:.6f} |")
        lines += ["", f"Selected by trainer validation: **{selected}**.",
                  "", "## Direct generation from selected arm", ""]
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
