#!/usr/bin/env python3
"""Unblind scores and report paired useful/loop rates. Run only after scores exist."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

JUDGE = Path(__file__).resolve().parent


def main() -> int:
    key = {row["id"]: row for row in json.loads((JUDGE / "sealed" / "key.json").read_text(encoding="utf-8"))}
    scores = [json.loads(line) for line in (JUDGE / "scores.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(scores) != len(key):
        raise SystemExit(f"scores {len(scores)} != items {len(key)}")
    fields = ("useful", "grammatical", "specific", "loop")
    by_prompt: dict[tuple[str, str], dict] = {}
    for row in scores:
        if row["id"] not in key:
            raise SystemExit(f"unknown id {row['id']}")
        for field in fields:
            value = row[field]
            if value not in (0, 1, 2):
                raise SystemExit(f"bad {field} on {row['id']}")
        meta = key[row["id"]]
        by_prompt[(meta["prompt_id"], meta["system"])] = {**row, **meta}
    prompts = sorted({prompt for prompt, _system in by_prompt})
    systems = ("flamew", "letter")
    if any((prompt, system) not in by_prompt for prompt in prompts for system in systems):
        raise SystemExit("missing a system on some prompt")

    def flag(prompt: str, system: str, field: str, cutoff: int) -> int:
        return int(by_prompt[(prompt, system)][field] >= cutoff)

    rng = np.random.default_rng(20260929)
    index = np.arange(len(prompts))
    out = {"n_prompts": len(prompts), "n_items": len(scores), "systems": {}}
    for system in systems:
        block = {}
        for field in fields:
            values = np.array([by_prompt[(prompt, system)][field] for prompt in prompts], dtype=float)
            block[field] = {
                "mean": float(values.mean()),
                "ge1": float((values >= 1).mean()),
                "eq2": float((values == 2).mean()),
            }
        out["systems"][system] = block
    paired = {}
    draws = rng.choice(index, size=(5000, len(prompts)), replace=True)
    for field, cutoff in (("useful", 1), ("loop", 1), ("grammatical", 1), ("specific", 1)):
        diff = np.array([
            flag(prompt, "flamew", field, cutoff) - flag(prompt, "letter", field, cutoff)
            for prompt in prompts
        ], dtype=float)
        samples = diff[draws].mean(axis=1)
        paired[f"{field}_ge1_flame_minus_letter"] = {
            "diff": float(diff.mean()),
            "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        }
    for field in fields:
        diff = np.array([
            by_prompt[(prompt, "flamew")][field] - by_prompt[(prompt, "letter")][field]
            for prompt in prompts
        ], dtype=float)
        samples = diff[draws].mean(axis=1)
        paired[f"{field}_mean_flame_minus_letter"] = {
            "diff": float(diff.mean()),
            "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        }
    out["paired"] = paired
    (JUDGE / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
