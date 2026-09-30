#!/usr/bin/env python3
"""Paired clean everyday comparison of accepted Flame refit and mature MICA."""
from __future__ import annotations

import json
from pathlib import Path

from common import paired_bootstrap
from run_mica_page_block_refit import digest, write_json


R1 = Path(__file__).resolve().parents[1]
RUN = R1 / "runs/codex_flame_refit_20260928_retry2"
MATURE = R1 / "runs/codex_page_block_refit_20260927_v2/control/best.mica"
MATURE_SHA = "d6cedad89ba210efe4ee50c2b87d6989f9fff241b73c75718b401d14b6e127cc"
SOURCE_DATA = R1 / "runs/codex_ember_chat_20260928/eval_everyday_clean_val1000.json"
TARGET_DATA = RUN / "eval_candidate_everyday_clean_val1000.json"


def main() -> None:
    if digest(MATURE) != MATURE_SHA:
        raise RuntimeError("mature everyday checkpoint changed")
    decision = json.loads((RUN / "decision.json").read_text(encoding="utf-8"))
    if not decision["accepted_on_fresh_development"]:
        raise RuntimeError("candidate did not pass development gate")
    candidate = json.loads(TARGET_DATA.read_text(encoding="utf-8"))["candidate"]
    mature = json.loads(SOURCE_DATA.read_text(encoding="utf-8"))["old_best"]
    if (digest(Path(candidate["mica"])) != decision["candidate_sha256"] or
            mature["mica"] != str(MATURE) or
            candidate["val_file"] != mature["val_file"] or
            candidate["counts"] != mature["counts"] or
            candidate["records"] != mature["records"] or
            candidate["records"] != 1000):
        raise RuntimeError("model or per-record clean evaluation mismatch")
    difference = list(paired_bootstrap(candidate["nats"], mature["nats"],
                                       candidate["counts"], n_boot=5000,
                                       seed=20260930))
    write_json(RUN / "comparison_mature_everyday.json", {
        "protocol": "exact exported integer C engine, bytes plus EOS from BOS, 5000 paired record bootstrap resamples",
        "set": candidate["val_file"], "records": 1000,
        "targets": candidate["targets"],
        "candidate_sha256": decision["candidate_sha256"],
        "mature_sha256": MATURE_SHA,
        "candidate_bits": candidate["bits"],
        "mature_bits": mature["bits"],
        "candidate_minus_mature": difference,
        "clean_reporting_only": True,
    })


if __name__ == "__main__":
    main()
