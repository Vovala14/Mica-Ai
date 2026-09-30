#!/usr/bin/env python3
"""Train native F1 Flame on the selected balanced mix A, preserving provenance."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from run_mica_overnight import train_command
from run_mica_page_block_refit import digest, write_json


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
SOURCE = R1 / "runs/codex_flame_scaling_20260928/F1/train"
SOURCE_SHA = "0c3287d16dca0622e3fdb497b2670206e6d73d9d72d8e45799975cb13c4f2a76"
MIX = R1 / "data/mix/v02a"
A = R1 / "runs/codex_ember_balanced_v02a_20260928"
OUT = R1 / "runs/codex_flame_balanced_v02a_20260928"
TRAIN = OUT / "train"


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "status.json", {
        "stage": stage, "detail": detail, "updated": time.time()})


def main() -> None:
    if digest(SOURCE / "best.mica") != SOURCE_SHA:
        raise RuntimeError("F1 one-round source changed")
    pre = json.loads((A / "preflight.json").read_text(encoding="utf-8"))
    if pre["variant"] != "v02a" or digest(MIX / "manifest.json") != pre["mix_manifest_sha256"]:
        raise RuntimeError("approved balanced mix A changed")
    for rel, sha in pre["mix_file_sha256"].items():
        if digest(R1 / "data" / rel) != sha:
            raise RuntimeError(f"balanced mix A file changed: {rel}")
    for name, sha in pre["heldout_sha256"].items():
        from run_mica_ember_balanced import EVALS
        if digest(EVALS[name]) != sha:
            raise RuntimeError(f"held-out set changed: {name}")
    if (ROOT.parent / "STOP_MICA").exists():
        raise RuntimeError("STOP_MICA blocks training")
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    args = info["args"]
    if (info["env"]["MICA_CANDIDATES"] != "1024" or args["steps"] != 1 or
            args["fit_records"] != 40000 or args["fit_steps"] != 1500 or
            args["round_lr"] != 0.3 or args["mode"] != "fit" or
            args["init"] != "fit-rules"):
        raise RuntimeError("F1 native training recipe changed")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update({k: str(v) for k, v in info["env"].items()})
    env["PYTHONUTF8"] = "1"
    cmd = train_command(info, TRAIN, 40)
    cmd[cmd.index("--records") + 1] = str(MIX / "train.jsonl")
    cmd[cmd.index("--val-records") + 1] = str(MIX / "val.jsonl")
    cmd += ["--tape-lags", args["tape_lags"]]
    write_json(OUT / "launch.json", {
        "command": cmd, "geometry_env": info["env"],
        "source_sha256": SOURCE_SHA,
        "mix_manifest_sha256": pre["mix_manifest_sha256"],
        "fit_code": "phase-stream bucket balancing patch applied 2026-09-28 09:20 UTC",
        "selection": "exact exported integer equal-weight mean chat dev1000 and everyday dev_fresh1000; clean chat/everyday val1000 reporting only",
        "rounds": 40,
    })
    if TRAIN.exists() and any(TRAIN.iterdir()) and not (
            (TRAIN / "resume.pt").is_file() and (TRAIN / "run_info.json").is_file()):
        raise RuntimeError("nonempty training folder without resumable state")
    status("training")
    with (OUT / "train.log").open("a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"trainer exited {rc}; see train.log")
    if (TRAIN / "FINISHED").is_file() and (TRAIN / "best.mica").is_file():
        status("training_complete", digest(TRAIN / "best.mica"))
    elif (TRAIN / "resume.pt").is_file():
        status("paused", "trainer saved resumable state")
    else:
        raise RuntimeError("trainer stopped without FINISHED or resume.pt")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise
