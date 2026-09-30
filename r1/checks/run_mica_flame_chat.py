#!/usr/bin/env python3
"""Train the nominated 1,024-candidate native MICA Flame on chat records."""
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
EMBER = R1 / "runs/codex_ember_chat_20260928"
OUT = R1 / "runs/codex_flame_chat_20260928_retry1"
TRAIN = OUT / "train"
CHAT = R1 / "data/chat"
FILTERED_VAL = EMBER / "val_without_clean_overlap.jsonl"


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "status.json", {
        "stage": stage, "detail": detail, "updated": time.time()})


def verify() -> dict:
    if digest(SOURCE / "best.mica") != SOURCE_SHA:
        raise RuntimeError("F1 nominated one-round source hash changed")
    preflight = json.loads((EMBER / "preflight.json").read_text(encoding="utf-8"))
    if (digest(CHAT / "manifest.json") != preflight["manifest_sha256"] or
            digest(FILTERED_VAL) != preflight["filtered_val_sha256"] or
            digest(CHAT / "dev1000.jsonl") != preflight["dev1000_sha256"] or
            digest(R1 / "data/eval_clean/chat_val1000.jsonl") !=
            preflight["chat_clean_sha256"] or
            digest(R1 / "data/eval_clean/val1000.jsonl") !=
            preflight["everyday_clean_sha256"]):
        raise RuntimeError("approved chat corpus, val copy or held-out set changed")
    manifest = json.loads((CHAT / "manifest.json").read_text(encoding="utf-8"))
    if (digest(CHAT / "train.jsonl") != manifest["files_sha256"]["chat/train.jsonl"] or
            preflight["records"]["train"] != 9149875):
        raise RuntimeError("approved training split changed")
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    a = info["args"]
    if (info["env"]["MICA_CANDIDATES"] != "1024" or a["steps"] != 1 or
            a["mode"] != "fit" or a["init"] != "fit-rules" or
            a["fit_records"] != 40000 or a["fit_steps"] != 1500 or
            a["round_lr"] != 0.3):
        raise RuntimeError("F1 nominated one-round training recipe changed")
    result = {
        "source": str(SOURCE / "best.mica"), "source_sha256": SOURCE_SHA,
        "manifest_sha256": preflight["manifest_sha256"],
        "chat_train_sha256": manifest["files_sha256"]["chat/train.jsonl"],
        "filtered_val_sha256": preflight["filtered_val_sha256"],
        "chat_dev1000_sha256": preflight["dev1000_sha256"],
        "chat_clean_val1000_sha256": preflight["chat_clean_sha256"],
        "everyday_clean_val1000_sha256": preflight["everyday_clean_sha256"],
        "train_records": preflight["records"]["train"],
        "selection": "within-training validation checkpoint; final exact integer acceptance on separate chat dev1000; clean sets for reporting only",
        "geometry": "native learned integer-rule cellular automaton, F1 1024 candidates, otherwise original c256 lag64 geometry",
    }
    write_json(OUT / "preflight.json", result)
    return result


def train() -> None:
    approved = verify()
    if (ROOT.parent / "STOP_MICA").exists():
        raise RuntimeError("STOP_MICA is present")
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update({k: str(v) for k, v in info["env"].items()})
    env["PYTHONUTF8"] = "1"
    cmd = train_command(info, TRAIN, 40)
    cmd[cmd.index("--records") + 1] = str(CHAT / "train.jsonl")
    cmd[cmd.index("--val-records") + 1] = str(FILTERED_VAL)
    cmd += ["--tape-lags", info["args"]["tape_lags"]]
    write_json(OUT / "launch.json", {
        "command": cmd, "geometry_env": info["env"],
        "manifest_sha256": approved["manifest_sha256"],
        "filtered_val_sha256": approved["filtered_val_sha256"],
        "rounds": 40, "training": "chat train only",
        "development": "chat/dev1000 exact integer, for choice after training",
        "clean": "chat and everyday clean val1000 for reporting only",
    })
    if TRAIN.exists() and any(TRAIN.iterdir()):
        if not ((TRAIN / "resume.pt").exists() and
                (TRAIN / "run_info.json").exists()):
            raise RuntimeError("nonempty training folder without resumable state")
    status("training")
    with (OUT / "train.log").open("a", encoding="utf-8") as stream:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"trainer exited {rc}; inspect train.log")
    if (TRAIN / "FINISHED").exists() and (TRAIN / "best.mica").exists():
        status("training_complete", digest(TRAIN / "best.mica"))
    elif (TRAIN / "resume.pt").exists():
        status("paused", "trainer saved a resumable state")
    else:
        raise RuntimeError("trainer stopped without FINISHED or resume.pt")


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise
