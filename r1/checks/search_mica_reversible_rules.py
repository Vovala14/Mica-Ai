#!/usr/bin/env python3
"""Choose hard, reversible MICA state rules by held-out prediction loss.

All arms use the identical integer engine, geometry, corpus and fit budget.
Only the phase-0 VSET rule transition multiplier changes. A multiplier is
eligible only when coprime to 243, so every page remains a permutation and
cannot erase a distinct state. The trainer validation set selects the rule;
the separate clean set checks whether that choice generalises.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_state_learning_pilot import ROOT, R1, SOURCE, train_args

OUT = R1 / "runs/ablation/mica_reversible_rule_search_20260927"
BASE4 = R1 / "runs/ablation/mica_reversible_state_pilot_20260927/reversible"
MULTIPLIERS = (2, 4, 5, 7, 10)
PYTHON = sys.executable


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps({"stage": stage, "detail": detail,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, OUT / "status.json")
    print(f"[rule-search] {stage} {detail}", flush=True)


def run(stage: str, cmd: list[str], env: dict) -> None:
    status(stage)
    with (OUT / f"{stage}.log").open("a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{stage} exited {rc}; see {stage}.log")


def arm(multiplier: int) -> Path:
    return BASE4 if multiplier == 4 else OUT / f"mul_{multiplier}"


def main() -> int:
    try:
        info = json.loads(SOURCE.read_text(encoding="utf-8"))
        env = dict(os.environ)
        env.update({k: str(v) for k, v in info["env"].items()})
        env["PYTHONUTF8"] = "1"
        if (ROOT.parent / "STOP_MICA").exists():
            raise RuntimeError("STOP_MICA blocks training")
        if not (BASE4 / "FINISHED").exists():
            raise RuntimeError("matched multiplier-4 arm is missing")
        for multiplier in MULTIPLIERS:
            folder = arm(multiplier)
            if (folder / "FINISHED").exists():
                continue
            cmd = train_args(info["args"], folder, "fit", 3)
            cmd[cmd.index("--rule-state") + 1] = "reversible"
            cmd += ["--reversible-multiplier", str(multiplier)]
            run(f"train_{multiplier}", cmd, env)
        scores = {}
        for multiplier in MULTIPLIERS:
            progress = json.loads((arm(multiplier) / "progress.json").read_text(
                encoding="utf-8"))
            scores[multiplier] = min(float(row["val_bits"])
                                     for row in progress["history"])
        selected = min(MULTIPLIERS, key=lambda m: scores[m])
        status("selected", f"multiplier {selected}: trainer val {scores[selected]:.4f}")
        clean = OUT / "clean_val128.json"
        if not clean.exists():
            cmd = [PYTHON, str(R1 / "checks/eval_int.py")]
            for multiplier in MULTIPLIERS:
                cmd += ["--run", f"mul_{multiplier}={arm(multiplier)}"]
            cmd += ["--set", "val128", "--val-file",
                    str(R1 / "data/eval_clean/val1000.jsonl"),
                    "--val-skip", "0", "--out", str(clean)]
            run("eval_clean", cmd, env)
        horizon = OUT / "horizon_selected.json"
        if not horizon.exists():
            run("horizon_selected", [PYTHON,
                str(R1 / "checks/context_horizon.py"),
                "--run", str(arm(selected)), "--out", str(horizon)], env)
        generation = OUT / "generation_selected.json"
        if not generation.exists():
            run("generation_selected", [PYTHON,
                str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(arm(selected)),
                "--prompts", str(R1 / "checks/generation_dev10.json"),
                "--out", str(generation), "--limit", "10",
                "--byte-limit", "100"], env)
        exact = json.loads(clean.read_text(encoding="utf-8"))
        rows = json.loads(generation.read_text(encoding="utf-8"))["rows"]
        lines = ["# Held-out selection of reversible MICA integer rules", "",
                 "The original MICA engine and geometry are identical across "
                 "arms. Each arm runs three 500-record/30-step fit rounds. "
                 "Only the phase-0 permutation multiplier differs. The "
                 "trainer's separate validation loss selects a multiplier; "
                 "the clean 128-record set is an independent check.", "",
                 "| Multiplier | Trainer validation bits/target | Clean validation bits/target |",
                 "|---:|---:|---:|"]
        for multiplier in MULTIPLIERS:
            lines.append(f"| {multiplier} | {scores[multiplier]:.4f} | "
                         f"{exact[f'mul_{multiplier}']['bits']:.6f} |")
        lines += ["", f"Selected multiplier: **{selected}**.", "",
                  "## Direct generation from selected MICA", ""]
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
