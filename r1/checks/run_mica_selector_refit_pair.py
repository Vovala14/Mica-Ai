#!/usr/bin/env python3
"""Refit matched MICA rulebooks after discrete selector proposals."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_mica_state_learning_pilot import ROOT, R1, SOURCE, train_args

OUT = R1 / "runs/ablation/mica_selector_refit_pair_20260927"
ORIGINAL = R1 / "runs/ablation/mica_state_learning_pilot_20260927/rule_learning"
PROJECTED = R1 / "runs/ablation/mica_selector_projection_pilot_20260927/projected_rules"
ARMS = (("ordinary", ORIGINAL), ("projected", PROJECTED))
PYTHON = sys.executable


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps({"stage": stage, "detail": detail,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, OUT / "status.json")
    print(f"[selector-refit] {stage} {detail}", flush=True)


def run(stage: str, cmd: list[str], env: dict) -> None:
    status(stage)
    with (OUT / f"{stage}.log").open("a", encoding="utf-8") as stream:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{stage} exited {rc}")


def clone_for_refit(source: Path, target: Path) -> None:
    import torch
    target.mkdir(parents=True, exist_ok=True)
    for name in ("resume.pt", "best.pt", "best.mica", "progress.json", "run_info.json"):
        shutil.copy2(source / name, target / name)
    ck = torch.load(target / "resume.pt", map_location="cpu", weights_only=False)
    if ck["step"] != 63:
        raise RuntimeError(f"expected final ST step 63, got {ck['step']}")
    # Each arm must export its new refitted model, even if that one round is
    # worse than the pre-refit control. This does not alter either source run.
    ck["best"] = float("inf")
    tmp = target / "resume.repaired.tmp"
    torch.save(ck, tmp)
    os.replace(tmp, target / "resume.pt")


def command(info: dict, target: Path) -> list[str]:
    cmd = train_args(info["args"], target, "fit", 64)
    for flag, value in (("--fit-records", "40000"),
                        ("--fit-steps", "1500")):
        index = cmd.index(flag)
        cmd[index + 1] = value
    return cmd


def report() -> None:
    scores = json.loads((OUT / "clean_val128.json").read_text(encoding="utf-8"))
    lines = ["# MICA discrete-rule selection followed by matched refit", "",
             "The two native integer-rule MICA models used identical geometry, "
             "corpus, fresh 40,000-record sample and 1,500-step refit budget. "
             "The only starting-rulebook difference was whether hard candidate "
             "scoring selectors had been gradient-projected during the preceding "
             "60-step pilot. The readout and rule immediates were refitted "
             "after those selector changes.", "",
             "| Rulebook | Clean val128 bits/target |",
             "|---|---:|"]
    for key in ("ordinary", "projected"):
        lines.append(f"| {key} | {scores[key]['bits']:.6f} |")
    lines += ["", f"Projected minus ordinary: "
              f"{scores['projected']['bits'] - scores['ordinary']['bits']:+.6f} bits/target.", ""]
    for key in ("ordinary", "projected"):
        path = OUT / f"generation_{key}.json"
        if path.exists():
            lines += [f"## Direct MICA generation: {key}", ""]
            for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
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
        OUT.mkdir(parents=True, exist_ok=True)
        for name, source in ARMS:
            target = OUT / name
            if not (target / "resume.pt").exists():
                status("clone_" + name)
                clone_for_refit(source, target)
            if not (target / "FINISHED").exists():
                run("refit_" + name, command(info, target), env)
        clean = R1 / "data/eval_clean/val1000.jsonl"
        if not (OUT / "clean_val128.json").exists():
            run("eval_clean", [PYTHON, str(R1 / "checks/eval_int.py"),
                "--run", f"ordinary={OUT / 'ordinary'}",
                "--run", f"projected={OUT / 'projected'}",
                "--set", "val128", "--val-file", str(clean),
                "--val-skip", "0", "--out", str(OUT / "clean_val128.json")], env)
        for name, _ in ARMS:
            path = OUT / f"generation_{name}.json"
            if not path.exists():
                run("generation_" + name, [PYTHON,
                    str(R1 / "checks/eval_mica_raw_generation.py"),
                    "--run", str(OUT / name),
                    "--prompts", str(R1 / "checks/generation_dev10.json"),
                    "--out", str(path), "--limit", "10", "--byte-limit", "100"], env)
        report()
        status("complete", str(OUT / "REPORT.md"))
        return 0
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
