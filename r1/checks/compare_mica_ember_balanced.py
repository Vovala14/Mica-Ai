#!/usr/bin/env python3
"""Compare the two balanced native MICA fits on paired held-out records."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from common import paired_bootstrap
from run_mica_ember_balanced import EVALS, paths
from run_mica_page_block_refit import digest, write_json


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "r1/runs/codex_ember_balanced_comparison_20260928"
DEV = ("chat_dev1000", "everyday_dev_fresh1000")
SETS = DEV + ("chat_clean_val1000", "everyday_clean_val1000")


def load(variant: str) -> tuple[dict, dict]:
    _, run, train = paths(variant)
    if json.loads((run / "eval_status.json").read_text(encoding="utf-8"))["stage"] != "complete":
        raise RuntimeError(f"{variant} exact integer evaluation is unfinished")
    results = json.loads((run / "eval_results.json").read_text(encoding="utf-8"))
    if (results["variant"] != variant or
            digest(train / "best.mica") != results["model_sha256"] or
            set(results["scores"]) != set(SETS)):
        raise RuntimeError(f"{variant} checkpoint or results changed")
    rows = {}
    for name in SETS:
        path = EVALS[name]
        if digest(path) != results["scores"][name]["set_sha256"]:
            raise RuntimeError(f"{name} held-out data changed")
        row = json.loads((run / f"eval_{name}.json").read_text(encoding="utf-8"))["balanced"]
        if (row["mica"] != str(train / "best.mica") or
                row["val_file"] != str(path) or row["records"] != 1000 or
                abs(row["bits"] - results["scores"][name]["bits"]) > 1e-12):
            raise RuntimeError(f"{variant} {name} records do not match summary")
        rows[name] = row
    return results, rows


def paired(a: dict, b: dict) -> list[float]:
    if a["counts"] != b["counts"] or len(a["counts"]) != 1000:
        raise RuntimeError("A/B held-out records are not aligned")
    return list(paired_bootstrap(a["nats"], b["nats"], a["counts"],
                                 n_boot=5000, seed=20260928))


def mean_dev_interval(a_rows: dict, b_rows: dict) -> list[float]:
    """Equal-weight mean of two pooled dev-set differences, paired by record."""
    rng = np.random.default_rng(20260928)
    samples = np.zeros(5000, dtype=np.float64)
    observed = 0.0
    for name in DEV:
        a, b = a_rows[name], b_rows[name]
        if a["counts"] != b["counts"]:
            raise RuntimeError(f"{name} A/B counts differ")
        counts = np.asarray(a["counts"], dtype=np.float64)
        difference = np.asarray(a["nats"], dtype=np.float64) - np.asarray(
            b["nats"], dtype=np.float64)
        observed += 0.5 * difference.sum() / counts.sum() / math.log(2)
        indexes = rng.integers(0, len(counts), (5000, len(counts)))
        samples += 0.5 * difference[indexes].sum(axis=1) / (
            counts[indexes].sum(axis=1) * math.log(2))
    low, high = np.percentile(samples, [2.5, 97.5])
    return [float(observed), float(low), float(high)]


def main() -> None:
    a, a_rows = load("v02a")
    b, b_rows = load("v02b")
    differences = {name: paired(a_rows[name], b_rows[name]) for name in SETS}
    combined = mean_dev_interval(a_rows, b_rows)
    chosen = "v02a" if combined[0] < 0 else "v02b"
    if combined[0] == 0:
        chosen = "tie"
    result = {
        "protocol": "exact exported integer C engine, pooled bytes plus EOS from BOS; A minus B; 5000 paired record bootstrap resamples",
        "A": {"variant": "v02a", "model": a["model"],
              "sha256": a["model_sha256"], "scores": a["scores"]},
        "B": {"variant": "v02b", "model": b["model"],
              "sha256": b["model_sha256"], "scores": b["scores"]},
        "A_minus_B": differences,
        "equal_weight_dev_mean_A_minus_B": combined,
        "selected_on_development_only": chosen,
        "selection_uncertain_at_95_percent": combined[1] <= 0 <= combined[2],
        "clean_sets_use": "reporting only",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "comparison.json", result)
    print(json.dumps({
        "A_dev_mean": a["selection_mean_dev_bits"],
        "B_dev_mean": b["selection_mean_dev_bits"],
        "A_minus_B_dev_mean": combined,
        "chosen": chosen,
    }, indent=2))


if __name__ == "__main__":
    main()
