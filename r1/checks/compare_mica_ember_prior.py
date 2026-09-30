#!/usr/bin/env python3
"""Paired exact integer comparisons for one completed balanced Ember run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import paired_bootstrap
from run_mica_ember_balanced import EVALS, paths
from run_mica_page_block_refit import digest, write_json


R1 = Path(__file__).resolve().parents[1]
FLAME = R1 / "runs/codex_flame_chat_20260928_retry1"
EMBER = R1 / "runs/codex_ember_refit_20260928"
OLD = R1 / "runs/codex_ember_chat_20260928"
MATURE = R1 / "runs/codex_page_block_refit_20260927_v2/control"
SHA = {
    "flame": "157671811845fb7426d734eed79d9f81dfcbe43957d1b6eda6e0dc62c508c9ce",
    "ember_refit": "f863bf6189863c4f5cbfe5357064dde560150946cc6cdef9832c3ac26edb0903",
    "mature_everyday": "d6cedad89ba210efe4ee50c2b87d6989f9fff241b73c75718b401d14b6e127cc",
}
COMPARATORS = {
    "chat_dev1000": {
        "flame": (FLAME / "eval_chat_dev1000.json", "flame"),
        "ember_refit": (EMBER / "eval_chat_dev1000.json", "candidate"),
        "mature_everyday": (OLD / "eval_chat_dev1000.json", "old_best"),
    },
    "chat_clean_val1000": {
        "flame": (FLAME / "eval_chat_clean_val1000.json", "flame"),
        "ember_refit": (EMBER / "eval_chat_clean_val1000.json", "candidate"),
        "mature_everyday": (OLD / "eval_chat_clean_val1000.json", "old_best"),
    },
    "everyday_clean_val1000": {
        "flame": (FLAME / "eval_everyday_clean_val1000.json", "flame"),
        "ember_refit": (EMBER / "eval_everyday_clean_val1000.json", "candidate"),
        "mature_everyday": (OLD / "eval_everyday_clean_val1000.json", "old_best"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("v02a", "v02b"))
    variant = parser.parse_args().variant
    _, out, train = paths(variant)
    evaluation = json.loads((out / "eval_results.json").read_text(encoding="utf-8"))
    if (json.loads((out / "eval_status.json").read_text(encoding="utf-8"))["stage"]
            != "complete" or digest(train / "best.mica") != evaluation["model_sha256"]):
        raise RuntimeError("balanced checkpoint has no complete exact evaluation")
    for name, path in (("flame", FLAME / "train/best.mica"),
                       ("ember_refit", EMBER / "candidate/best.mica"),
                       ("mature_everyday", MATURE / "best.mica")):
        if digest(path) != SHA[name]:
            raise RuntimeError(f"{name} comparator checkpoint changed")
    comparisons = {}
    for set_name, comparators in COMPARATORS.items():
        if digest(EVALS[set_name]) != evaluation["scores"][set_name]["set_sha256"]:
            raise RuntimeError(f"{set_name} held-out file changed")
        own = json.loads((out / f"eval_{set_name}.json").read_text(
            encoding="utf-8"))["balanced"]
        rows = {}
        for name, (path, key) in comparators.items():
            prior = json.loads(path.read_text(encoding="utf-8"))[key]
            if (own["counts"] != prior["counts"] or own["records"] != 1000 or
                    len(own["nats"]) != 1000 or len(prior["nats"]) != 1000):
                raise RuntimeError(f"{set_name}: {name} record misalignment")
            rows[name] = {
                "prior_bits": prior["bits"],
                "balanced_minus_prior": list(paired_bootstrap(
                    own["nats"], prior["nats"], own["counts"],
                    n_boot=5000, seed=20260928)),
            }
        comparisons[set_name] = {
            "balanced_bits": own["bits"], "targets": own["targets"],
            "comparisons": rows,
        }
    result = {
        "variant": variant, "model": str(train / "best.mica"),
        "model_sha256": evaluation["model_sha256"],
        "comparator_sha256": SHA,
        "protocol": "exact exported integer C engine; pooled bytes plus EOS from BOS; 5000 paired record bootstrap resamples",
        "note": "development set is for decisions; clean sets are for reporting only",
        "sets": comparisons,
    }
    write_json(out / "comparison_prior.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
