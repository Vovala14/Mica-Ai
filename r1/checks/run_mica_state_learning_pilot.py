#!/usr/bin/env python3
"""Bounded MICA-only pilot: train native integer-rule selectors and state.

The control and treatment share the original engine, geometry, corpus, and
stateful initialization. The treatment then uses the existing straight-through
trainer to update rule selectors as well as the readout. This is a feasibility
check, not a replacement architecture or a release candidate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
SOURCE = R1 / "runs/sweep/everyday_c256/run_info.json"
OUT = R1 / "runs/ablation/mica_state_learning_pilot_20260927"
FIT = OUT / "fit_control"
ST = OUT / "rule_learning"
LAST = OUT / "last_rule_state"
PYTHON = sys.executable


def atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def status(stage: str, extra: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    atomic(OUT / "status.json", json.dumps({"stage": stage, "extra": extra,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2) + "\n")
    print(f"[mica-pilot] {stage} {extra}", flush=True)


def run(name: str, cmd: list[str], env: dict[str, str]) -> None:
    status(name)
    with (OUT / f"{name}.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name}: exit {result.returncode}; see {name}.log")


def train_args(a: dict, out: Path, mode: str, steps: int) -> list[str]:
    keys = ("records", "val_records", "batch", "record_bytes", "ticks", "tau",
            "sel_init", "sel_tau", "lr", "int_lr", "init", "work_probes",
            "bias_init", "clip", "rule_max_back", "rule_scoring",
            "rule_self_terms", "min_batch", "device", "gpu_mem_fraction",
            "min_free_ram_gb", "tbptt", "stall_minutes", "yield_gb")
    values = {key: a[key] for key in keys}
    values.update({"rule_state": "stream", "mode": mode, "steps": steps,
                   "fit_records": 500, "fit_steps": 30, "round_lr": 0.3,
                   "val_records_n": 64, "checkpoint_every": 0,
                   "save_every": 1 if mode == "fit" else 5,
                   "warmup": 20, "val_every": 10, "log_every": 10,
                   "out": str(out)})
    if mode == "st":
        values["lr"] = 0.02
        values["int_lr"] = 0.1
    cmd = [PYTHON, str(R1 / "train_soft.py")]
    for key, value in values.items():
        cmd.extend(("--" + key.replace("_", "-"), str(value)))
    cmd.append("--segments")
    return cmd


def clone_best_as_resume() -> None:
    ST.mkdir(parents=True, exist_ok=True)
    for name in ("resume.pt", "best.pt", "best.mica", "progress.json", "run_info.json"):
        shutil.copy2(FIT / name, ST / name)
    import torch
    ck = torch.load(ST / "resume.pt", map_location="cpu", weights_only=False)
    ck["model"] = torch.load(ST / "best.pt", map_location="cpu", weights_only=True)
    hist = json.loads((ST / "progress.json").read_text(encoding="utf-8"))["history"]
    ck["best"] = min(row["val_bits"] for row in hist)
    ck["step"] = hist[-1]["step"]
    ck["hist"] = hist
    tmp = ST / "resume.repaired.tmp"
    torch.save(ck, tmp)
    os.replace(tmp, ST / "resume.pt")


def export_last(env: dict[str, str]) -> None:
    # Separate process so the geometry is imported under the saved MICA env.
    code = ("import sys; sys.path.insert(0,r'" + str(R1) + "'); "
            "import torch; from mica_r1 import soft, discretise, serialize; "
            "m=soft.SoftMica(ticks=16,tau=0.5,hard=True,sel_init=0.05,sel_tau=1.0); "
            "ck=torch.load(r'" + str(ST / "resume.pt") + "',map_location='cpu',weights_only=False); "
            "m.load_state_dict(ck['model']); "
            "serialize.save(discretise.to_integer(m),r'" + str(ST / "last.mica") + "')")
    run("export_last", [PYTHON, "-c", code], env)
    LAST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ST / "last.mica", LAST / "best.mica")
    shutil.copy2(ST / "run_info.json", LAST / "run_info.json")


def report() -> None:
    comparison = json.loads((OUT / "clean_val128.json").read_text(encoding="utf-8"))
    base, learned, last = (comparison[key] for key in ("control", "learned", "last"))
    horizon = {}
    for key in ("control", "learned", "last"):
        rows = json.loads((OUT / f"horizon_{key}.json").read_text(encoding="utf-8"))["rows"]
        horizon[key] = {row["shared_suffix_bytes"]: row["exactly_equal_pairs"]
                        for row in rows}
    gen = json.loads((OUT / "generation_learned.json").read_text(encoding="utf-8"))
    lines = ["# MICA native rule-learning pilot", "",
             "Same Minimal Inference Cellular Automaton engine, original 32-byte probe window, "
             "same everyday corpus. Both arms start from the same stateful integer-rule fit. "
             "The treatment then trains MICA's rule selectors with the native straight-through trainer.", "",
             f"Clean validation, 128 records: control **{base['bits']:.6f}**, "
             f"best rule-learning checkpoint **{learned['bits']:.6f}**, "
             f"last checkpoint **{last['bits']:.6f}** bits/target.", "",
             "| Shared suffix | Control equal-score pairs | Best learned | Last learned |",
             "|---:|---:|---:|---:|"]
    for n in (8, 16, 32, 64, 128):
        lines.append(f"| {n} bytes | {horizon['control'][n]}/6 | "
                     f"{horizon['learned'][n]}/6 | {horizon['last'][n]}/6 |")
    lines += ["", "## Direct raw MICA generations", ""]
    for row in gen["rows"]:
        lines.append(f"- `{row['prompt']}` → `{row['full_text']}`")
    lines += ["", "This is a feasibility pilot. A clean 1,000-record validation, "
              "paired uncertainty check, and quality review are required before any long run.", ""]
    atomic(OUT / "REPORT.md", "\n".join(lines))


def main() -> int:
    try:
        info = json.loads(SOURCE.read_text(encoding="utf-8"))
        env = dict(os.environ)
        env.update({k: str(v) for k, v in info["env"].items()})
        env["PYTHONUTF8"] = "1"
        if (ROOT.parent / "STOP_MICA").exists():
            raise RuntimeError("STOP_MICA blocks training")
        OUT.mkdir(parents=True, exist_ok=True)
        if not (FIT / "FINISHED").exists():
            run("fit_control", train_args(info["args"], FIT, "fit", 3), env)
        if not (ST / "resume.pt").exists():
            status("clone_control")
            clone_best_as_resume()
        if not (ST / "FINISHED").exists():
            run("learn_rules", train_args(info["args"], ST, "st", 63), env)
        if not (ST / "last.mica").exists():
            export_last(env)
        if not (LAST / "best.mica").exists():
            LAST.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ST / "last.mica", LAST / "best.mica")
            shutil.copy2(ST / "run_info.json", LAST / "run_info.json")
        clean = R1 / "data/eval_clean/val1000.jsonl"
        if not (OUT / "clean_val128.json").exists():
            run("eval_clean", [PYTHON, str(R1 / "checks/eval_int.py"),
                 "--run", f"control={FIT}", "--run", f"learned={ST}",
                 "--run", f"last={LAST}",
                 "--set", "val128", "--val-file", str(clean),
                 "--val-skip", "0", "--out", str(OUT / "clean_val128.json")], env)
        for key, folder in (("control", FIT), ("learned", ST), ("last", LAST)):
            path = OUT / f"horizon_{key}.json"
            if not path.exists():
                run("horizon_" + key, [PYTHON, str(R1 / "checks/context_horizon.py"),
                    "--run", str(folder), "--out", str(path)], env)
        if not (OUT / "generation_learned.json").exists():
            run("generation_learned", [PYTHON, str(R1 / "checks/eval_mica_raw_generation.py"),
                 "--run", str(ST), "--prompts", str(R1 / "checks/generation_dev10.json"),
                 "--out", str(OUT / "generation_learned.json"),
                 "--limit", "10", "--byte-limit", "100"], env)
        report()
        status("complete", str(OUT / "REPORT.md"))
        return 0
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
