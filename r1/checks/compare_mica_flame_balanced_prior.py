#!/usr/bin/env python3
"""Paired clean-set comparisons of balanced Flame against prior checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

from common import paired_bootstrap
from run_mica_ember_balanced import EVALS
from run_mica_page_block_refit import digest, write_json


R1 = Path(__file__).resolve().parents[1]
NEW = R1 / "runs/codex_flame_balanced_v02a_20260928"
OLD_FLAME = R1 / "runs/codex_flame_chat_20260928_retry1"
OLD_EMBER = R1 / "runs/codex_ember_chat_20260928"
MATURE = R1 / "runs/codex_page_block_refit_20260927_v2/control"
EXPECTED = {
    "new": "46be93f265743baacd391698aac9721a6bfba2b616180c00eaa02e421dac2e92",
    "old_flame": "157671811845fb7426d734eed79d9f81dfcbe43957d1b6eda6e0dc62c508c9ce",
    "mature_everyday": "d6cedad89ba210efe4ee50c2b87d6989f9fff241b73c75718b401d14b6e127cc",
}


def row(folder: Path, name: str, key: str) -> dict:
    return json.loads((folder / f"eval_{name}.json").read_text(
        encoding="utf-8"))[key]


def compare(name: str, comparator: str) -> dict:
    candidate = row(NEW, name, "balanced")
    if comparator == "old_flame":
        prior = row(OLD_FLAME, name, "flame")
    else:
        prior = row(OLD_EMBER, name, "old_best")
    if (candidate["counts"] != prior["counts"] or
            candidate["records"] != 1000 or
            candidate["val_file"] != str(EVALS[name])):
        raise RuntimeError(f"{name} paired records do not align")
    return {
        "set_sha256": digest(EVALS[name]),
        "targets": candidate["targets"],
        "candidate_bits": candidate["bits"],
        "prior_bits": prior["bits"],
        "candidate_minus_prior": list(paired_bootstrap(
            candidate["nats"], prior["nats"], candidate["counts"],
            n_boot=5000, seed=20260928)),
    }


def main() -> None:
    if digest(NEW / "train/best.mica") != EXPECTED["new"]:
        raise RuntimeError("new Flame checkpoint changed")
    if digest(OLD_FLAME / "train/best.mica") != EXPECTED["old_flame"]:
        raise RuntimeError("prior Flame checkpoint changed")
    if digest(MATURE / "best.mica") != EXPECTED["mature_everyday"]:
        raise RuntimeError("mature everyday checkpoint changed")
    scores = json.loads((NEW / "eval_results.json").read_text(encoding="utf-8"))
    if (scores["model_sha256"] != EXPECTED["new"] or
            json.loads((NEW / "eval_status.json").read_text(
                encoding="utf-8"))["stage"] != "complete"):
        raise RuntimeError("new exact integer evaluation unfinished or stale")
    results = {
        "protocol": "exact exported integer C engine, bytes plus EOS from BOS, 5000 paired record bootstrap resamples",
        "candidate": str(NEW / "train/best.mica"),
        "candidate_sha256": EXPECTED["new"],
        "comparator_sha256": {k: EXPECTED[k] for k in ("old_flame", "mature_everyday")},
        "clean_reporting_only": True,
        "comparisons": {
            "chat_clean_vs_old_flame": compare("chat_clean_val1000", "old_flame"),
            "everyday_clean_vs_old_flame": compare("everyday_clean_val1000", "old_flame"),
            "everyday_clean_vs_mature": compare("everyday_clean_val1000", "mature_everyday"),
        },
    }
    write_json(NEW / "comparison_prior.json", results)
    print(json.dumps(results["comparisons"], indent=2))


if __name__ == "__main__":
    main()
