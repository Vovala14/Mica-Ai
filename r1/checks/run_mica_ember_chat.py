#!/usr/bin/env python3
"""Verify the approved chat corpus, then train native MICA Ember v0.1.

The architecture and 40-round fit recipe are the existing lag-64 MICA.
This script never reads the sealed completion test for model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_overnight import train_command
from run_mica_page_block_refit import digest, write_json


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
DATA = R1 / "data"
CHAT = DATA / "chat"
SOURCE = R1 / "runs/codex_page_block_refit_20260927_v2/control"
SOURCE_SHA = "d6cedad89ba210efe4ee50c2b87d6989f9fff241b73c75718b401d14b6e127cc"
OUT = R1 / "runs/codex_ember_chat_20260928"
TRAIN = OUT / "train"
FILTERED_VAL = OUT / "val_without_clean_overlap.jsonl"


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "status.json", {"stage": stage, "detail": detail,
                                     "updated": time.time()})


def records(path: Path):
    with path.open(encoding="ascii") as stream:
        for line in stream:
            text = line.strip()
            if not text or len(text) % 2:
                raise ValueError(f"invalid hex record in {path}")
            row = bytes.fromhex(text)
            if not (2 <= len(row) <= 256):
                raise ValueError(f"invalid record length in {path}")
            yield row


def verify() -> dict:
    manifest_path = CHAT / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("chat corpus manifest not ready")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"chat/train.jsonl", "chat/val.jsonl", "chat/test.jsonl",
                "chat/dev1000.jsonl", "eval_clean/chat_val1000.jsonl"}
    if set(manifest["files_sha256"]) != expected:
        raise RuntimeError("manifest file set differs from approved builder")
    for rel, sha in manifest["files_sha256"].items():
        path = DATA / rel
        if not path.is_file() or digest(path) != sha:
            raise RuntimeError(f"corpus hash mismatch: {rel}")
    evals = {}
    for name, path in (("dev1000", CHAT / "dev1000.jsonl"),
                       ("chat_clean", DATA / "eval_clean/chat_val1000.jsonl"),
                       ("everyday_clean", DATA / "eval_clean/val1000.jsonl")):
        rows = set(records(path))
        if len(rows) != 1000:
            raise RuntimeError(f"{name} needs 1000 distinct records")
        evals[name] = rows
    if (evals["dev1000"] & evals["chat_clean"] or
            evals["dev1000"] & evals["everyday_clean"] or
            evals["chat_clean"] & evals["everyday_clean"]):
        raise RuntimeError("development/clean sets overlap")
    reserved = set().union(*evals.values())
    counts = {"train": 0, "val": 0}
    for row in records(CHAT / "train.jsonl"):
        counts["train"] += 1
        if row in reserved:
            raise RuntimeError("train overlaps development or clean records")
    if counts["train"] != manifest["records"]["train"]:
        raise RuntimeError("train count differs from manifest")
    OUT.mkdir(parents=True, exist_ok=True)
    pending_val = FILTERED_VAL.with_suffix(".tmp")
    removed_val = []
    filtered_count = 0
    with pending_val.open("w", encoding="ascii", newline="\n") as stream:
        for row in records(CHAT / "val.jsonl"):
            counts["val"] += 1
            if row in reserved:
                removed_val.append(row.decode("utf-8", errors="replace"))
            else:
                stream.write(row.hex() + "\n")
                filtered_count += 1
    if counts["val"] != manifest["records"]["val"]:
        raise RuntimeError("val count differs from manifest")
    if len(removed_val) > 10:
        raise RuntimeError("unexpectedly many val records overlap clean data")
    pending_val.replace(FILTERED_VAL)
    if digest(SOURCE / "best.mica") != SOURCE_SHA:
        raise RuntimeError("source geometry checkpoint changed")
    report = {
        "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "source": str(SOURCE / "best.mica"),
        "source_sha256": SOURCE_SHA,
        "records": counts,
        "filtered_val_records": filtered_count,
        "removed_val_overlap": removed_val,
        "filtered_val_path": str(FILTERED_VAL),
        "filtered_val_sha256": digest(FILTERED_VAL),
        "dev1000_sha256": digest(CHAT / "dev1000.jsonl"),
        "chat_clean_sha256": digest(DATA / "eval_clean/chat_val1000.jsonl"),
        "everyday_clean_sha256": digest(DATA / "eval_clean/val1000.jsonl"),
        "protocol": "verified corpus hashes; train exact-separated; derived val copy excludes development and clean records",
    }
    write_json(OUT / "preflight.json", report)
    return report


def train() -> None:
    preflight_path = OUT / "preflight.json"
    if not preflight_path.exists():
        raise RuntimeError("run --stage verify before training")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (digest(CHAT / "manifest.json") != preflight["manifest_sha256"] or
            digest(FILTERED_VAL) != preflight["filtered_val_sha256"] or
            digest(SOURCE / "best.mica") != SOURCE_SHA):
        raise RuntimeError("manifest, filtered validation or source changed after preflight")
    if (ROOT.parent / "STOP_MICA").exists():
        raise RuntimeError("STOP_MICA is present")
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    args = info["args"]
    if (args["steps"] != 40 or args["fit_records"] != 40000 or
            args["fit_steps"] != 1500 or args["round_lr"] != 0.3 or
            args["mode"] != "fit" or args["init"] != "fit-rules"):
        raise RuntimeError("source training recipe changed")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("MICA_")}
    env.update({key: str(value) for key, value in info["env"].items()})
    env["PYTHONUTF8"] = "1"
    cmd = train_command(info, TRAIN, 40)
    cmd[cmd.index("--records") + 1] = str(CHAT / "train.jsonl")
    cmd[cmd.index("--val-records") + 1] = str(FILTERED_VAL)
    cmd += ["--tape-lags", args["tape_lags"]]
    write_json(OUT / "launch.json", {
        "command": cmd, "geometry_env": info["env"],
        "manifest_sha256": preflight["manifest_sha256"],
        "filtered_val_sha256": preflight["filtered_val_sha256"],
        "selection": "trainer chooses within the chat validation split; final model selection on separate chat/dev1000 exact integer",
        "clean": "existing clean everyday val1000 and new clean chat val1000 for reporting only",
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
        status("training_complete")
    elif (TRAIN / "resume.pt").exists():
        status("paused", "trainer saved a resumable state")
    else:
        raise RuntimeError("trainer stopped without FINISHED or resume.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=("verify", "train"))
    args = ap.parse_args()
    try:
        if args.stage == "verify":
            status("verifying")
            print(json.dumps(verify(), indent=2), flush=True)
            status("verified")
        else:
            train()
    except Exception as exc:
        status("failed", f"{args.stage}: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
