#!/usr/bin/env python3
"""Bounded native word-symbol Flame smoke/P1 runs, with source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from run_mica_overnight import train_command


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
SOURCE = R1 / "runs/codex_flame_balanced_v02a_20260928/train/run_info.json"
WORD = R1 / "data/word"
MANIFEST_SHA = "331687945906ddc634714f769e5e91878ffd09d94c065a2e4d163ddcd330dff3"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_arg(cmd: list[str], name: str, value: object) -> None:
    cmd[cmd.index(name) + 1] = str(value)


def run(stage: str) -> int:
    if digest(WORD / "manifest.json") != MANIFEST_SHA:
        raise RuntimeError("word corpus manifest changed")
    manifest = json.loads((WORD / "manifest.json").read_text())
    if manifest["n_symbols"] != 16384 or manifest["report"]["train_records"] != 4507891:
        raise RuntimeError("unexpected word corpus geometry or record count")
    if any((ROOT.parent / name).exists() for name in ("STOP_MICA", "SUPERVISOR_EXIT")):
        raise RuntimeError("a training stop file is present")
    source = json.loads(SOURCE.read_text())
    if source["args"]["mode"] != "fit" or source["args"]["init"] != "fit-rules":
        raise RuntimeError("source recipe is no longer the native fit mode")
    out = R1 / "runs" / ("codex_flame_word_smoke_20260928" if stage == "smoke"
                          else "codex_flame_word_p1_20260928")
    train = out / "train"
    if train.exists() and any(train.iterdir()):
        raise RuntimeError(f"refusing existing training output: {train}")
    out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update({k: str(v) for k, v in source["env"].items()})
    env["MICA_SYMBOLS"] = "16384"
    env["PYTHONUTF8"] = "1"
    cmd = train_command(source, train, 1)
    set_arg(cmd, "--records", WORD / "v02a/train.jsonl")
    set_arg(cmd, "--val-records", WORD / "v02a/val.jsonl")
    set_arg(cmd, "--record-bytes", 64)
    set_arg(cmd, "--val-records-n", 4 if stage == "smoke" else 64)
    if stage == "smoke":
        set_arg(cmd, "--fit-records", 128)
        set_arg(cmd, "--fit-steps", 2)
        set_arg(cmd, "--rule-scoring", "random")
    cmd += ["--tape-lags", source["args"]["tape_lags"]]
    protocol = {"stage": stage, "word_manifest_sha256": MANIFEST_SHA,
                "source_recipe": str(SOURCE), "source_geometry": source["env"],
                "geometry": env["MICA_SYMBOLS"], "command": cmd,
                "selection": "P1 continues only after separate chat/everyday development next-word top-1 gate; clean sets withheld"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    status = out / "status.json"
    status.write_text(json.dumps({"stage": "training", "updated": time.time()}))
    with (out / "train.log").open("a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        status.write_text(json.dumps({"stage": "failed", "exit_code": rc,
                                      "updated": time.time()}))
        return rc
    if not (train / "FINISHED").exists() or not (train / "best.mica").exists():
        status.write_text(json.dumps({"stage": "incomplete", "updated": time.time()}))
        return 1
    status.write_text(json.dumps({"stage": "finished", "model_sha256": digest(train / "best.mica"),
                                  "updated": time.time()}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "p1"), required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.stage))
