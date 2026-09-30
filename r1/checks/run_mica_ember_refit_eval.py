#!/usr/bin/env python3
"""Apply the predeclared chat-dev gate, then report the Ember refit on clean sets."""
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
BASE = R1 / "runs/codex_ember_chat_20260928"
REFIT = R1 / "runs/codex_ember_refit_20260928"
CANDIDATE = REFIT / "candidate"
SOURCE_SHA = "8f64bc80df1f02b71a0959d84d3df11b033e9f6091d3fc59173684fe85d607ac"
CANDIDATE_SHA = "f863bf6189863c4f5cbfe5357064dde560150946cc6cdef9832c3ac26edb0903"
SETS = (
    ("chat_clean_val1000", R1 / "data/eval_clean/chat_val1000.jsonl"),
    ("everyday_clean_val1000", R1 / "data/eval_clean/val1000.jsonl"),
)


def status(stage: str, detail: str = "") -> None:
    write_json(REFIT / "eval_status.json", {
        "stage": stage, "detail": detail, "updated": time.time()})


def model_result(path: Path, key: str) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))[key]


def paired(candidate: dict, comparator: dict) -> list[float]:
    if (candidate["counts"] != comparator["counts"] or
            candidate["records"] != 1000 or comparator["records"] != 1000):
        raise RuntimeError("exact integer evaluations are not record-aligned")
    return list(paired_bootstrap(candidate["nats"], comparator["nats"],
                                 candidate["counts"], n_boot=5000))


def main() -> None:
    if digest(BASE / "train/best.mica") != SOURCE_SHA or \
            digest(CANDIDATE / "best.mica") != CANDIDATE_SHA:
        raise RuntimeError("source or candidate checkpoint hash changed")
    candidate_dev = model_result(REFIT / "eval_chat_dev1000.json", "candidate")
    source_dev = model_result(BASE / "eval_chat_dev1000.json", "ember")
    dev_diff = paired(candidate_dev, source_dev)
    accepted = dev_diff[0] <= -0.005 and dev_diff[2] < 0
    decision = {
        "accepted": accepted, "rule": "candidate minus source <= -0.005 bits/target and paired 95% CI upper < 0 on disjoint chat dev1000",
        "candidate_sha256": CANDIDATE_SHA, "source_sha256": SOURCE_SHA,
        "candidate_bits": candidate_dev["bits"], "source_bits": source_dev["bits"],
        "difference_ci95": dev_diff, "targets": candidate_dev["targets"],
        "protocol": "exact exported integer C engine, pooled bytes plus EOS from BOS, 5,000 record-bootstrap resamples",
    }
    write_json(REFIT / "acceptance.json", decision)
    if not accepted:
        status("rejected", "chat dev1000 gate not met")
        return
    report = {"accepted": True, "candidate": str(CANDIDATE / "best.mica"),
              "candidate_sha256": CANDIDATE_SHA,
              "source_sha256": SOURCE_SHA,
              "development": decision, "clean": {}}
    write_json(REFIT / "eval_results.json", report)
    for name, path in SETS:
        status("evaluating", name)
        out = REFIT / f"eval_{name}.json"
        cmd = [sys.executable, str(R1 / "checks/eval_int.py"),
               "--run", f"candidate={CANDIDATE}", "--set", "val1000",
               "--val-file", str(path), "--val-skip", "0", "--out", str(out)]
        with (REFIT / "eval_clean.log").open("a", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                                  stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"exact integer {name} failed: {proc.returncode}")
        candidate = model_result(out, "candidate")
        existing = json.loads((BASE / f"eval_{name}.json").read_text(
            encoding="utf-8"))
        source, old = existing["ember"], existing["old_best"]
        report["clean"][name] = {
            "set_sha256": digest(path), "targets": candidate["targets"],
            "candidate_bits": candidate["bits"],
            "source_bits": source["bits"], "old_best_bits": old["bits"],
            "candidate_minus_source": paired(candidate, source),
            "candidate_minus_old_best": paired(candidate, old),
        }
        write_json(REFIT / "eval_results.json", report)
    if digest(CANDIDATE / "best.mica") != CANDIDATE_SHA:
        raise RuntimeError("candidate checkpoint changed during evaluation")
    status("generating")
    cmd = [sys.executable, str(R1 / "checks/eval_mica_raw_generation.py"),
           "--run", str(CANDIDATE), "--prompts",
           str(R1 / "checks/generation_dev10.json"), "--limit", "10",
           "--byte-limit", "100", "--out", str(REFIT / "generation.json")]
    with (REFIT / "eval_clean.log").open("a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"raw generation failed: {proc.returncode}")
    status("complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise
