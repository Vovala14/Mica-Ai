#!/usr/bin/env python3
"""Matched discrete-selector MICA training pilot, against ordinary ST."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_mica_state_learning_pilot import FIT, ROOT, R1, SOURCE, train_args

OUT = R1 / "runs/ablation/mica_selector_projection_pilot_20260927"
PROJECTED = OUT / "projected_rules"
LAST = OUT / "projected_last"
ORDINARY = R1 / "runs/ablation/mica_state_learning_pilot_20260927/rule_learning"
PYTHON = sys.executable


def status(stage: str, note: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps({"stage": stage, "note": note,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, OUT / "status.json")
    print(f"[selector-pilot] {stage} {note}", flush=True)


def run(stage: str, cmd: list[str], env: dict) -> None:
    status(stage)
    with (OUT / f"{stage}.log").open("a", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{stage} exited {rc}; see {stage}.log")


def clone_control() -> None:
    import torch
    PROJECTED.mkdir(parents=True, exist_ok=True)
    for name in ("resume.pt", "best.pt", "best.mica", "progress.json", "run_info.json"):
        shutil.copy2(FIT / name, PROJECTED / name)
    ck = torch.load(PROJECTED / "resume.pt", map_location="cpu", weights_only=False)
    ck["model"] = torch.load(PROJECTED / "best.pt", map_location="cpu", weights_only=True)
    hist = json.loads((PROJECTED / "progress.json").read_text(encoding="utf-8"))["history"]
    ck["best"] = min(row["val_bits"] for row in hist)
    ck["step"] = hist[-1]["step"]
    ck["hist"] = hist
    tmp = PROJECTED / "resume.repaired.tmp"
    torch.save(ck, tmp)
    os.replace(tmp, PROJECTED / "resume.pt")


def export_last(env: dict) -> None:
    code = ("import sys;sys.path.insert(0,r'" + str(R1) + "'); "
            "import torch;from mica_r1 import soft,discretise,serialize; "
            "m=soft.SoftMica(ticks=16,tau=0.5,hard=True,sel_init=0.05,sel_tau=1.0); "
            "ck=torch.load(r'" + str(PROJECTED / "resume.pt") + "',map_location='cpu',weights_only=False); "
            "m.load_state_dict(ck['model']); "
            "serialize.save(discretise.to_integer(m),r'" + str(LAST / "best.mica") + "')")
    LAST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECTED / "run_info.json", LAST / "run_info.json")
    run("export_last", [PYTHON, "-c", code], env)


def selector_changes(env: dict) -> dict:
    code = ("import sys,json,numpy as np;sys.path.insert(0,r'" + str(R1) + "'); "
            "from mica_r1 import serialize; "
            "a=serialize.load(r'" + str(FIT / "best.mica") + "'); "
            "b=serialize.load(r'" + str(LAST / "best.mica") + "'); "
            "print(json.dumps({k:int(np.count_nonzero(getattr(a,k)!=getattr(b,k))) "
            "for k in ('sc_nb','sc_ch','sc_co','sc_bias','op_b','op_v','pr_co','pr_bias')}))")
    p = subprocess.run([PYTHON, "-c", code], cwd=ROOT, env=env,
                       capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def make_report(env: dict) -> None:
    scores = json.loads((OUT / "clean_val128.json").read_text(encoding="utf-8"))
    counts = selector_changes(env)
    horizon = {}
    for name in ("projected_best", "projected_last"):
        rows = json.loads((OUT / f"horizon_{name}.json").read_text(encoding="utf-8"))["rows"]
        horizon[name] = {r["shared_suffix_bytes"]: r["exactly_equal_pairs"] for r in rows}
    samples = json.loads((OUT / "generation_projected_last.json").read_text(
        encoding="utf-8"))["rows"]
    lines = ["# MICA discrete selector search pilot", "",
             "All arms use the original Minimal Inference Cellular Automaton "
             "geometry and integer inference engine. The projected and ordinary "
             "straight-through arms start from the same stateful control checkpoint, "
             "see the same corpus, and run for 60 gradient steps. Every fifth step, "
             "the projected arm moves a few hard score-selector argmaxes toward "
             "the best first-order gradient alternative; exact integer validation "
             "selects the best checkpoint.", "",
             "## Clean validation (128 fixed records)", "",
             "| Arm | bits/target |", "|---|---:|"]
    for key in ("control", "ordinary_best", "ordinary_last",
                "projected_best", "projected_last"):
        lines.append(f"| {key} | {scores[key]['bits']:.6f} |")
    lines += ["", "## Exported integer-selector changes vs control (last projected)", "",
              "| Field | Changed entries |", "|---|---:|"]
    for key, value in counts.items():
        lines.append(f"| `{key}` | {value} |")
    lines += ["", "## Context influence: exactly equal next-byte scores", "",
              "| Common suffix | Projected best | Projected last |",
              "|---:|---:|---:|"]
    for n in (8, 16, 32, 64, 128):
        lines.append(f"| {n} bytes | {horizon['projected_best'][n]}/6 | "
                     f"{horizon['projected_last'][n]}/6 |")
    lines += ["", "## Raw direct generations from projected last checkpoint", ""]
    for row in samples:
        lines.append(f"- `{row['prompt']}` → `{row['full_text']}`")
    lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    try:
        info = json.loads(SOURCE.read_text(encoding="utf-8"))
        env = dict(os.environ)
        env.update({k: str(v) for k, v in info["env"].items()})
        env["PYTHONUTF8"] = "1"
        if (ROOT.parent / "STOP_MICA").exists():
            raise RuntimeError("STOP_MICA blocks training")
        if not (FIT / "FINISHED").exists() or not (ORDINARY / "FINISHED").exists():
            raise RuntimeError("matched control and ordinary ST must finish first")
        OUT.mkdir(parents=True, exist_ok=True)
        if not (PROJECTED / "resume.pt").exists():
            status("clone_control")
            clone_control()
        if not (PROJECTED / "FINISHED").exists():
            cmd = train_args(info["args"], PROJECTED, "st", 63)
            cmd += ["--selector-flips-per-phase", "2",
                    "--selector-project-every", "5",
                    "--selector-project-margin", "1.0"]
            run("train_projected", cmd, env)
        if not (LAST / "best.mica").exists():
            export_last(env)
        clean = R1 / "data/eval_clean/val1000.jsonl"
        if not (OUT / "clean_val128.json").exists():
            run("eval_clean", [PYTHON, str(R1 / "checks/eval_int.py"),
                "--run", f"control={FIT}",
                "--run", f"ordinary_best={ORDINARY}",
                "--run", f"ordinary_last={ORDINARY.parent / 'last_rule_state'}",
                "--run", f"projected_best={PROJECTED}",
                "--run", f"projected_last={LAST}",
                "--set", "val128", "--val-file", str(clean),
                "--val-skip", "0", "--out", str(OUT / "clean_val128.json")], env)
        for name, folder in (("projected_best", PROJECTED),
                             ("projected_last", LAST)):
            path = OUT / f"horizon_{name}.json"
            if not path.exists():
                run("horizon_" + name, [PYTHON, str(R1 / "checks/context_horizon.py"),
                    "--run", str(folder), "--out", str(path)], env)
        if not (OUT / "generation_projected_last.json").exists():
            run("generation_projected_last", [PYTHON,
                str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(LAST),
                "--prompts", str(R1 / "checks/generation_dev10.json"),
                "--out", str(OUT / "generation_projected_last.json"),
                "--limit", "10", "--byte-limit", "100"], env)
        make_report(env)
        status("complete", str(OUT / "REPORT.md"))
        return 0
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
