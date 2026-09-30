#!/usr/bin/env python3
"""Exact integer held-out evaluation of completed 1,024-candidate MICA Flame."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from common import paired_bootstrap
from run_mica_page_block_refit import digest, write_json


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
OUT = R1 / "runs/codex_flame_chat_20260928_retry1"
TRAIN = OUT / "train"
EMBER_BASE = R1 / "runs/codex_ember_chat_20260928"
EMBER_REFIT = R1 / "runs/codex_ember_refit_20260928"
SETS = (
    ("chat_dev1000", R1 / "data/chat/dev1000.jsonl"),
    ("chat_clean_val1000", R1 / "data/eval_clean/chat_val1000.jsonl"),
    ("everyday_clean_val1000", R1 / "data/eval_clean/val1000.jsonl"),
)


def status(stage: str, detail: str = "") -> None:
    write_json(OUT / "eval_status.json", {
        "stage": stage, "detail": detail, "updated": time.time()})


def paired(a: dict, b: dict) -> list[float]:
    if a["counts"] != b["counts"] or a["records"] != 1000:
        raise RuntimeError("held-out integer evaluations are not record-aligned")
    return list(paired_bootstrap(a["nats"], b["nats"],
                                 a["counts"], n_boot=5000))


def cached(name: str) -> tuple[dict, dict]:
    if name == "chat_dev1000":
        ember = json.loads((EMBER_REFIT / "eval_chat_dev1000.json").read_text(
            encoding="utf-8"))["candidate"]
    else:
        ember = json.loads((EMBER_REFIT / f"eval_{name}.json").read_text(
            encoding="utf-8"))["candidate"]
    everyday = json.loads((EMBER_BASE / f"eval_{name}.json").read_text(
        encoding="utf-8"))["old_best"]
    return ember, everyday


def main() -> None:
    if not (TRAIN / "FINISHED").is_file():
        raise RuntimeError("Flame native fit has not finished")
    flame_sha = digest(TRAIN / "best.mica")
    ember_sha = digest(EMBER_REFIT / "candidate/best.mica")
    everyday_sha = digest(R1 / "runs/codex_page_block_refit_20260927_v2/control/best.mica")
    if ember_sha != "f863bf6189863c4f5cbfe5357064dde560150946cc6cdef9832c3ac26edb0903":
        raise RuntimeError("accepted Ember refit checkpoint changed")
    if everyday_sha != "d6cedad89ba210efe4ee50c2b87d6989f9fff241b73c75718b401d14b6e127cc":
        raise RuntimeError("mature everyday checkpoint changed")
    result = {
        "protocol": "exact exported integer C engine; pooled bytes plus EOS from BOS; 5,000-resample paired record bootstrap",
        "flame": str(TRAIN / "best.mica"), "flame_sha256": flame_sha,
        "ember_refit_sha256": ember_sha, "mature_everyday_sha256": everyday_sha,
        "selection_gate": "Flame minus accepted Ember <= -0.020 bits/target on disjoint chat dev1000 with paired 95% CI upper < 0; clean chat confirms before retention",
        "scores": {},
    }
    write_json(OUT / "eval_results.json", result)
    for name, path in SETS:
        status("evaluating", name)
        out = OUT / f"eval_{name}.json"
        cmd = [sys.executable, str(R1 / "checks/eval_int.py"),
               "--run", f"flame={TRAIN}", "--set", "val1000",
               "--val-file", str(path), "--val-skip", "0", "--out", str(out)]
        with (OUT / "eval.log").open("a", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                                  stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"exact integer {name} failed: {proc.returncode}")
        flame = json.loads(out.read_text(encoding="utf-8"))["flame"]
        ember, everyday = cached(name)
        result["scores"][name] = {
            "set_sha256": digest(path), "targets": flame["targets"],
            "flame_bits": flame["bits"], "ember_refit_bits": ember["bits"],
            "mature_everyday_bits": everyday["bits"],
            "flame_minus_ember_refit": paired(flame, ember),
            "flame_minus_mature_everyday": paired(flame, everyday),
        }
        write_json(OUT / "eval_results.json", result)
    if digest(TRAIN / "best.mica") != flame_sha:
        raise RuntimeError("Flame checkpoint changed during evaluation")
    status("generating")
    cmd = [sys.executable, str(R1 / "checks/eval_mica_raw_generation.py"),
           "--run", str(TRAIN), "--prompts",
           str(R1 / "checks/generation_dev10.json"), "--limit", "10",
           "--byte-limit", "100", "--out", str(OUT / "generation.json")]
    with (OUT / "eval.log").open("a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"raw generation failed: {proc.returncode}")
    dev = result["scores"]["chat_dev1000"]["flame_minus_ember_refit"]
    clean = result["scores"]["chat_clean_val1000"]["flame_minus_ember_refit"]
    result["accepted"] = bool(dev[0] <= -0.020 and dev[2] < 0 and clean[0] < 0 and clean[2] < 0)
    write_json(OUT / "eval_results.json", result)
    status("complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise
