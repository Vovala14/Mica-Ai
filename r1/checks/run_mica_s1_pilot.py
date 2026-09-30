#!/usr/bin/env python3
"""Matched one-round native MICA F1, word-track, and S1 hard-bucket pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_mica_overnight import train_command
from run_mica_page_block_refit import digest, write_json
from run_mica_ember_balanced import EVALS


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
SOURCE = R1 / "runs/codex_flame_scaling_20260928/F1/train"
SOURCE_SHA = "0c3287d16dca0622e3fdb497b2670206e6d73d9d72d8e45799975cb13c4f2a76"
MIX = R1 / "data/mix/v02a"
A = R1 / "runs/codex_ember_balanced_v02a_20260928"
OUT = R1 / "runs/codex_s1_20260928"
ARMS = {"f1_control": "none", "state_only": "lexical2-track",
        "s1": "lexical2-s1"}


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "status.json", {
        "stage": stage, "detail": detail, "updated": time.time()})


def verify() -> dict:
    pre = json.loads((A / "preflight.json").read_text(encoding="utf-8"))
    if (digest(SOURCE / "best.mica") != SOURCE_SHA or
            pre["variant"] != "v02a" or
            digest(MIX / "manifest.json") != pre["mix_manifest_sha256"]):
        raise RuntimeError("F1 source or approved mix A changed")
    for rel, sha in pre["mix_file_sha256"].items():
        if digest(R1 / "data" / rel) != sha:
            raise RuntimeError(f"balanced training file changed: {rel}")
    for name, sha in pre["heldout_sha256"].items():
        if digest(EVALS[name]) != sha:
            raise RuntimeError(f"held-out file changed: {name}")
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    a = info["args"]
    if (info["env"]["MICA_CANDIDATES"] != "1024" or a["steps"] != 1 or
            a["mode"] != "fit" or a["init"] != "fit-rules" or
            a["fit_records"] != 40000 or a["fit_steps"] != 1500 or
            a["round_lr"] != 0.3):
        raise RuntimeError("F1 one-round recipe changed")
    proof = {
        "source": str(SOURCE / "best.mica"), "source_sha256": SOURCE_SHA,
        "mix_manifest_sha256": pre["mix_manifest_sha256"],
        "heldout_sha256": pre["heldout_sha256"],
        "training_code_sha256": {
            "fit.py": digest(R1 / "mica_r1/fit.py"),
            "train_soft.py": digest(R1 / "train_soft.py"),
        },
        "arms": ARMS,
        "train": "same seed, balanced mix A, native F1 geometry, one round, 40000 fit records and 1500 fit steps each",
        "gates": "see docs/2026-09-28-mica-s1-pilot.md; exact integer dev loss F1/state-only/S1 and use(8) on two disjoint context probes; clean sets only after both gates",
        "sealed_test": "untouched",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    saved = OUT / "preflight.json"
    if saved.is_file() and json.loads(saved.read_text(encoding="utf-8")) != proof:
        raise RuntimeError("existing S1 preflight differs from current files")
    write_json(saved, proof)
    return info


def train_arm(arm: str, info: dict) -> None:
    dest = OUT / arm / "train"
    if (dest / "FINISHED").is_file() and (dest / "best.mica").is_file():
        status("arm_complete", f"{arm}: {digest(dest / 'best.mica')}")
        return
    if dest.exists() and any(dest.iterdir()) and not (
            (dest / "resume.pt").is_file() and (dest / "run_info.json").is_file()):
        raise RuntimeError(f"{arm} has nonempty nonresumable training folder")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update({k: str(v) for k, v in info["env"].items()})
    env["PYTHONUTF8"] = "1"
    cmd = train_command(info, dest, 1)
    cmd[cmd.index("--records") + 1] = str(MIX / "train.jsonl")
    cmd[cmd.index("--val-records") + 1] = str(MIX / "val.jsonl")
    cmd[cmd.index("--rule-state") + 1] = ARMS[arm]
    cmd += ["--tape-lags", info["args"]["tape_lags"]]
    write_json(OUT / arm / "launch.json", {
        "arm": arm, "command": cmd, "geometry_env": info["env"],
        "preflight_sha256": digest(OUT / "preflight.json")})
    status("training", arm)
    with (OUT / arm / "train.log").open("a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{arm} trainer exited {rc}; see train.log")
    if (dest / "FINISHED").is_file() and (dest / "best.mica").is_file():
        status("arm_complete", f"{arm}: {digest(dest / 'best.mica')}")
    elif (dest / "resume.pt").is_file():
        status("paused", arm)
        raise RuntimeError(f"{arm} paused with resumable state")
    else:
        raise RuntimeError(f"{arm} trainer stopped without FINISHED or resume.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("verify", "train"), required=True)
    args = ap.parse_args()
    if (ROOT.parent / "STOP_MICA").exists() and args.stage == "train":
        raise RuntimeError("STOP_MICA blocks training")
    info = verify()
    if args.stage == "verify":
        status("verified")
        return
    for arm in ARMS:
        train_arm(arm, info)
    status("training_complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise
