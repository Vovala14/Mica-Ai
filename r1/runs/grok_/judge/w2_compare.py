#!/usr/bin/env python3
"""Paired W2 gate. Opens the W2 key only after scores.jsonl exists."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

JUDGE = Path(__file__).resolve().parent
W2 = JUDGE / "w2"


def main() -> int:
    key = {row["id"]: row for row in json.loads((W2 / "sealed" / "key.json").read_text(encoding="utf-8"))}
    scores = [json.loads(line) for line in (W2 / "scores.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(scores) != len(key):
        raise SystemExit(f"scores {len(scores)} != {len(key)}")
    control = {}
    for line in (JUDGE / "assignments.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["system"] == "letter":
            control[row["prompt_id"]] = row
    arm = {}
    for row in scores:
        meta = key[row["id"]]
        for field in ("useful", "grammatical", "specific", "loop"):
            if row[field] not in (0, 1, 2):
                raise SystemExit(f"bad {field} on {row['id']}")
        arm[meta["prompt_id"]] = {**row, **meta}
    prompts = sorted(control)
    if sorted(arm) != prompts or len(prompts) != 100:
        raise SystemExit(f"prompt mismatch control {len(control)} arm {len(arm)}")
    rng = np.random.default_rng(20260929)
    index = np.arange(len(prompts))
    draws = rng.choice(index, size=(5000, len(prompts)), replace=True)
    out = {"n": 100, "fields": {}}
    for field in ("useful", "grammatical", "specific", "loop"):
        diff = np.array([
            int(arm[prompt][field] >= 1) - int(control[prompt][field] >= 1)
            for prompt in prompts
        ], dtype=float)
        samples = diff[draws].mean(axis=1)
        ci = [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
        out["fields"][field] = {
            "control_ge1": float((diff * 0 + np.array([control[p][field] >= 1 for p in prompts])).mean()),
            "arm_ge1": float(np.array([arm[p][field] >= 1 for p in prompts]).mean()),
            "diff_arm_minus_control": float(diff.mean()),
            "ci95": ci,
        }
    loop_ci = out["fields"]["loop"]["ci95"]
    useful_ci = out["fields"]["useful"]["ci95"]
    out["gate"] = {
        "loop_pass": loop_ci[1] < 0,
        "useful_pass": useful_ci[1] >= 0,
    }
    out["gate"]["pass"] = out["gate"]["loop_pass"] and out["gate"]["useful_pass"]
    (W2 / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
