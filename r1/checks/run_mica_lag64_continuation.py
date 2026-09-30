#!/usr/bin/env python3
"""Isolated continuation of the native lag-64 integer-rule MICA checkpoint.

The source checkpoint and fixed clean evaluation set are never modified. The
exact exported model is sampled at two-hour intervals and at process exit.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_mica_overnight import file_hash, progress, train_command, write_atomic

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
PYTHON = Path(sys.executable)
STOP = ROOT.parent / "STOP_MICA"


def status(out: Path, phase: str, detail: str = "") -> None:
    step, internal_best = progress(out)
    write_atomic(out / "status.json", json.dumps({
        "phase": phase, "detail": detail, "round": step,
        "internal_best_bits": internal_best,
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
    }, indent=2) + "\n")


def sample(out: Path, label: str) -> None:
    step, _ = progress(out)
    snap = out / "samples" / label
    snap.mkdir(parents=True, exist_ok=True)
    source = out / "best.mica"
    shutil.copy2(source, snap / "best.mica")
    shutil.copy2(out / "run_info.json", snap / "run_info.json")
    digest = file_hash(snap / "best.mica")
    result = {"label": label, "round": step, "sha256": digest,
              "clean_val1000_bits": None, "generation": []}
    eval_file = snap / "clean_val1000.json"
    cmd = [str(PYTHON), str(R1 / "checks/eval_int.py"),
           "--run", f"lag64={snap}", "--set", "val1000",
           "--val-file", str(R1 / "data/eval_clean/val1000.jsonl"),
           "--val-skip", "0", "--out", str(eval_file)]
    ev = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                        timeout=1800)
    (snap / "eval.log").write_text(ev.stdout + ev.stderr, encoding="utf-8")
    if ev.returncode:
        raise RuntimeError(f"clean evaluation failed ({ev.returncode})")
    result["clean_val1000_bits"] = json.loads(eval_file.read_text(
        encoding="utf-8"))["lag64"]["bits"]
    gen_file = snap / "generation_dev10.json"
    cmd = [str(PYTHON), str(R1 / "checks/eval_mica_raw_generation.py"),
           "--run", str(snap), "--prompts",
           str(R1 / "checks/generation_dev10.json"), "--out", str(gen_file),
           "--limit", "10", "--byte-limit", "100"]
    gen = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         timeout=300)
    (snap / "generation.log").write_text(gen.stdout + gen.stderr,
                                          encoding="utf-8")
    if gen.returncode:
        raise RuntimeError(f"raw generation failed ({gen.returncode})")
    result["generation"] = json.loads(gen_file.read_text(
        encoding="utf-8"))["rows"]
    samples_file = out / "samples.json"
    samples = json.loads(samples_file.read_text(encoding="utf-8")) \
        if samples_file.exists() else []
    samples.append(result)
    write_atomic(samples_file, json.dumps(samples, indent=2,
                                          ensure_ascii=False) + "\n")
    print(f"[sample] {label} round {step} clean "
          f"{result['clean_val1000_bits']:.6f} SHA {digest[:12]}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()
    source, out = a.source.resolve(), a.out.resolve()
    if STOP.exists():
        raise SystemExit(f"training blocked by {STOP}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing nonempty output directory {out}")
    info = json.loads((source / "run_info.json").read_text(encoding="utf-8"))
    if info["args"]["rule_state"] != "none" or info["env"]["MICA_PROBE_WINDOW"] != "64":
        raise SystemExit("source is not the native lag-64 MICA control")
    if a.steps <= info["args"]["steps"]:
        raise SystemExit("target steps must exceed source round")
    out.mkdir(parents=True, exist_ok=True)
    status(out, "copying")
    for name in ("resume.pt", "best.pt", "best.mica", "progress.json",
                 "run_info.json"):
        shutil.copy2(source / name, out / name)
    env = dict(os.environ)
    env.update(info["env"])
    env["PYTHONUTF8"] = "1"
    cmd = train_command(info, out, a.steps)
    cmd.extend(["--tape-lags", info["args"]["tape_lags"]])
    if info["args"]["work_lags"]:
        cmd.extend(["--work-lags", info["args"]["work_lags"]])
    write_atomic(out / "launch.json", json.dumps({
        "source": str(source), "source_sha256": file_hash(source / "best.mica"),
        "steps": a.steps, "command": cmd, "env": info["env"],
    }, indent=2) + "\n")
    status(out, "training")
    with (out / "train.log").open("w", encoding="utf-8") as log:
        flags = (getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) |
                 getattr(subprocess, "CREATE_NO_WINDOW", 0))
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, creationflags=flags)
        write_atomic(out / "train.pid", str(proc.pid) + "\n")
        next_sample = time.monotonic() + 2 * 3600
        count = 1
        while proc.poll() is None:
            if time.monotonic() >= next_sample:
                try:
                    sample(out, f"plus_{2 * count}h")
                except Exception as exc:
                    print(f"[sample] failed: {type(exc).__name__}: {exc}",
                          flush=True)
                count += 1
                next_sample += 2 * 3600
            time.sleep(30)
        rc = proc.returncode
    try:
        sample(out, "on_exit")
    except Exception as exc:
        status(out, "evaluation_failed", f"{type(exc).__name__}: {exc}")
        return 1
    phase = "complete" if rc == 0 and (out / "FINISHED").exists() else "stopped"
    status(out, phase, f"trainer exit {rc}")
    return rc if rc else (0 if phase == "complete" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
