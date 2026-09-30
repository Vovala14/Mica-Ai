#!/usr/bin/env python3
"""Flame-W training runs on the PC GPU (Claude, 2026-09-28; owner-approved).

Adapted from Codex's r1/checks/run_mica_flame_word.py, which is left
unchanged. Differences:
  * outputs go under r1/runs/claude_flame_word_20260928/<stage>/;
  * the only stop file honoured is STOP_MICA, the training stop. The
    SUPERVISOR_EXIT flag of the uninstalled auto-restart watcher is not a
    training stop;
  * stage "smoke+p1" runs the smoke and, if it finishes, P1, in one process.

    python run_flamew.py --stage smoke|p1|smoke+p1|full40

Stages (Flame F1 native fit recipe of codex_flame_balanced_v02a_20260928,
with MICA_SYMBOLS=16384 and the word mix A from r1/data/word):
  smoke   128 fit records, 2 fit steps, random rule scoring, one round
  p1      the full recipe for one round (the P1 pilot)
  full40  the full recipe for 40 rounds (only after P1 passes its gate)
Clean evaluation sets are never read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                     # the mica repository
R1 = ROOT / "r1"
sys.path.insert(0, str(R1 / "checks"))
from run_mica_overnight import train_command  # noqa: E402

SOURCE = R1 / "runs/codex_flame_balanced_v02a_20260928/train/run_info.json"
WORD = R1 / "data/word"
MANIFEST_SHA = "331687945906ddc634714f769e5e91878ffd09d94c065a2e4d163ddcd330dff3"
STOP = ROOT.parent / "STOP_MICA"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_arg(cmd: list[str], name: str, value: object) -> None:
    cmd[cmd.index(name) + 1] = str(value)


def run(stage: str) -> int:
    if STOP.exists():
        raise RuntimeError(f"{STOP} exists: training is stopped by the owner")
    if digest(WORD / "manifest.json") != MANIFEST_SHA:
        raise RuntimeError("word corpus manifest changed")
    manifest = json.loads((WORD / "manifest.json").read_text())
    if manifest["n_symbols"] != 16384 or manifest["report"]["train_records"] != 4507891:
        raise RuntimeError("unexpected word corpus geometry or record count")
    source = json.loads(SOURCE.read_text())
    if source["args"]["mode"] != "fit" or source["args"]["init"] != "fit-rules":
        raise RuntimeError("source recipe is no longer the native fit mode")
    out = HERE / stage
    train = out / "train"
    if train.exists() and any(train.iterdir()):
        raise RuntimeError(f"refusing existing training output: {train}")
    out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update({k: str(v) for k, v in source["env"].items()})
    env["MICA_SYMBOLS"] = "16384"
    env["PYTHONUTF8"] = "1"
    cmd = train_command(source, train, 40 if stage == "full40" else 1)
    set_arg(cmd, "--records", WORD / "v02a/train.jsonl")
    set_arg(cmd, "--val-records", WORD / "v02a/val.jsonl")
    set_arg(cmd, "--record-bytes", 64)
    set_arg(cmd, "--val-records-n", 4 if stage == "smoke" else 64)
    if stage == "smoke":
        set_arg(cmd, "--fit-records", 128)
        set_arg(cmd, "--fit-steps", 2)
        set_arg(cmd, "--rule-scoring", "random")
    cmd += ["--tape-lags", source["args"]["tape_lags"]]
    protocol = {"stage": stage, "by": "Claude (owner-approved, Codex away)",
                "word_manifest_sha256": MANIFEST_SHA, "source_recipe": str(SOURCE.relative_to(ROOT)),
                "source_geometry": source["env"], "symbols": env["MICA_SYMBOLS"], "command": cmd,
                "gate": "P1 continues to full40 only if the mean next-word top-1 on chat dev1000 "
                        "and everyday dev_fresh1000 (letter metric positions) is at least the "
                        "letter Flame refit 6cdcd921's 16.075%; clean sets withheld"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    status = out / "status.json"
    status.write_text(json.dumps({"stage": "training", "updated": time.time()}))
    t0 = time.time()
    with (out / "train.log").open("a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        status.write_text(json.dumps({"stage": "failed", "exit_code": rc,
                                      "seconds": round(time.time() - t0), "updated": time.time()}))
        return rc
    if not (train / "FINISHED").exists() or not (train / "best.mica").exists():
        status.write_text(json.dumps({"stage": "incomplete", "seconds": round(time.time() - t0),
                                      "updated": time.time()}))
        return 1
    status.write_text(json.dumps({"stage": "finished", "model_sha256": digest(train / "best.mica"),
                                  "seconds": round(time.time() - t0), "updated": time.time()}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "p1", "smoke+p1", "full40"), required=True)
    args = parser.parse_args()
    if args.stage == "smoke+p1":
        rc = run("smoke")
        raise SystemExit(rc if rc else run("p1"))
    raise SystemExit(run(args.stage))
