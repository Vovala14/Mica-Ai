#!/usr/bin/env python3
"""Verify a balanced v02 corpus and train original-geometry native MICA Ember."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from run_mica_overnight import train_command
from run_mica_page_block_refit import digest, write_json


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
DATA = R1 / "data"
SOURCE = R1 / "runs/codex_page_block_refit_20260927_v2/control"
SOURCE_SHA = "d6cedad89ba210efe4ee50c2b87d6989f9fff241b73c75718b401d14b6e127cc"
SHARES = {
    "v02a": {"conversation": .45, "everyday": .40, "tinystories": .15},
    "v02b": {"conversation": .60, "everyday": .30, "tinystories": .10},
}
EVALS = {
    "chat_dev1000": DATA / "chat/dev1000.jsonl",
    "everyday_dev_fresh1000": R1 / "runs/codex_flame_scaling_20260928/dev_fresh1000.jsonl",
    "chat_clean_val1000": DATA / "eval_clean/chat_val1000.jsonl",
    "everyday_clean_val1000": DATA / "eval_clean/val1000.jsonl",
}


def key(raw: bytes) -> bytes:
    text = raw[:256].decode("utf-8", errors="replace").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.blake2b(text.encode(), digest_size=16).digest()


def records(path: Path):
    with path.open(encoding="ascii") as stream:
        for line in stream:
            value = line.strip()
            if not value or len(value) % 2:
                raise ValueError(f"invalid hex record in {path}")
            row = bytes.fromhex(value)
            if not (2 <= len(row) <= 256):
                raise ValueError(f"invalid record length in {path}")
            yield row


def paths(variant: str) -> tuple[Path, Path, Path]:
    mix = DATA / "mix" / variant
    out = R1 / "runs" / f"codex_ember_balanced_{variant}_20260928"
    return mix, out, out / "train"


def status(out: Path, stage: str, detail: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "status.json", {
        "stage": stage, "detail": detail, "updated": time.time()})


def verify(variant: str) -> dict:
    mix, out, _ = paths(variant)
    man_path = mix / "manifest.json"
    if not man_path.is_file():
        raise RuntimeError("balanced mixture is not ready")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if (man.get("version") != 2 or man.get("name") != variant or
            man.get("shares") != SHARES[variant] or
            man.get("chat_manifest_sha256") != digest(DATA / "chat/manifest.json")):
        raise RuntimeError("balanced mixture provenance or shares changed")
    expected = {f"mix/{variant}/{split}.{suffix}" for split in ("train", "val")
                for suffix in ("jsonl", "src")}
    if set(man["files_sha256"]) != expected:
        raise RuntimeError("mixture file list changed")
    for rel, expected_sha in man["files_sha256"].items():
        if digest(DATA / rel) != expected_sha:
            raise RuntimeError(f"mixture hash mismatch: {rel}")
    if digest(SOURCE / "best.mica") != SOURCE_SHA:
        raise RuntimeError("original geometry checkpoint changed")
    reserved = {}
    for name, path in EVALS.items():
        rows = list(records(path))
        if len(rows) != 1000:
            raise RuntimeError(f"{name} is not 1000 records")
        reserved[name] = {key(row) for row in rows}
    union = set().union(*reserved.values())
    counts = {}
    for split in ("train", "val"):
        count = 0
        for row in records(mix / f"{split}.jsonl"):
            if key(row) in union:
                raise RuntimeError(f"{variant} {split} has normalized held-out overlap")
            count += 1
        if count != man["splits"][split]["total"]["records"]:
            raise RuntimeError(f"{variant} {split} record count changed")
        counts[split] = count
    result = {
        "variant": variant, "mix_manifest_sha256": digest(man_path),
        "chat_manifest_sha256": digest(DATA / "chat/manifest.json"),
        "source": str(SOURCE / "best.mica"), "source_sha256": SOURCE_SHA,
        "mix_file_sha256": man["files_sha256"],
        "heldout_sha256": {name: digest(path) for name, path in EVALS.items()},
        "records": counts,
        "selection": "equal-weight mean of exact integer chat dev1000 and everyday dev_fresh1000; clean chat/everyday val1000 reporting only",
        "architecture": "original c256 lag64 Minimal Inference Cellular Automaton with learned integer rules",
    }
    write_json(out / "preflight.json", result)
    return result


def train(variant: str) -> None:
    mix, out, dest = paths(variant)
    pre_path = out / "preflight.json"
    if not pre_path.is_file():
        raise RuntimeError("run --stage verify first")
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    if (pre["variant"] != variant or
            digest(mix / "manifest.json") != pre["mix_manifest_sha256"] or
            digest(DATA / "chat/manifest.json") != pre["chat_manifest_sha256"] or
            digest(SOURCE / "best.mica") != SOURCE_SHA):
        raise RuntimeError("approved mixture, chat manifest or source changed")
    for rel, expected_sha in pre["mix_file_sha256"].items():
        if digest(DATA / rel) != expected_sha:
            raise RuntimeError(f"mixture file changed since preflight: {rel}")
    if (ROOT.parent / "STOP_MICA").exists():
        raise RuntimeError("STOP_MICA is present")
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    args = info["args"]
    if (args["steps"] != 40 or args["fit_records"] != 40000 or
            args["fit_steps"] != 1500 or args["round_lr"] != 0.3 or
            args["mode"] != "fit" or args["init"] != "fit-rules"):
        raise RuntimeError("original MICA training recipe changed")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update({k: str(v) for k, v in info["env"].items()})
    env["PYTHONUTF8"] = "1"
    cmd = train_command(info, dest, 40)
    cmd[cmd.index("--records") + 1] = str(mix / "train.jsonl")
    cmd[cmd.index("--val-records") + 1] = str(mix / "val.jsonl")
    cmd += ["--tape-lags", args["tape_lags"]]
    write_json(out / "launch.json", {"command": cmd, "geometry_env": info["env"],
                                      "mix_manifest_sha256": pre["mix_manifest_sha256"],
                                      "source_sha256": SOURCE_SHA})
    if dest.exists() and any(dest.iterdir()):
        if not ((dest / "resume.pt").exists() and
                (dest / "run_info.json").exists()):
            raise RuntimeError("nonempty training folder without resumable state")
    status(out, "training")
    with (out / "train.log").open("a", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"trainer exited {rc}; inspect train.log")
    if (dest / "FINISHED").exists() and (dest / "best.mica").exists():
        status(out, "training_complete", digest(dest / "best.mica"))
    elif (dest / "resume.pt").exists():
        status(out, "paused", "trainer saved a resumable state")
    else:
        raise RuntimeError("trainer stopped without FINISHED or resume.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=tuple(SHARES))
    ap.add_argument("--stage", required=True, choices=("verify", "train"))
    args = ap.parse_args()
    _, out, _ = paths(args.variant)
    try:
        if args.stage == "verify":
            status(out, "verifying")
            print(json.dumps(verify(args.variant), indent=2), flush=True)
            status(out, "verified")
        else:
            train(args.variant)
    except Exception as exc:
        status(out, "failed", f"{args.stage}: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
