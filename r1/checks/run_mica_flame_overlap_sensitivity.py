#!/usr/bin/env python3
"""Read-only sensitivity of exact Flame scores to v1 train/val overlaps."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from common import paired_bootstrap


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
OUT = R1 / "runs/codex_flame_chat_20260928_retry1"
TRAIN_V1 = R1 / "runs/codex_corpus_v2_20260928/v1_snapshot/train.jsonl"
VAL_V1 = R1 / "runs/codex_corpus_v2_20260928/v1_snapshot/val.jsonl"
EMBER = R1 / "runs/codex_ember_refit_20260928"
MATURE = R1 / "runs/codex_ember_chat_20260928"
SETS = {
    "chat_dev1000": R1 / "data/chat/dev1000.jsonl",
    "chat_clean_val1000": R1 / "data/eval_clean/chat_val1000.jsonl",
    "everyday_clean_val1000": R1 / "data/eval_clean/val1000.jsonl",
}


def key(raw: bytes) -> bytes:
    text = raw[:256].decode("utf-8", errors="replace").casefold()
    return hashlib.blake2b(re.sub(r"\s+", " ", text).strip().encode(),
                           digest_size=16).digest()


def score(row: dict, keep: list[int]) -> float:
    return sum(row["nats"][i] for i in keep) / math.log(2) / sum(
        row["counts"][i] for i in keep)


def main() -> None:
    rows = {name: [bytes.fromhex(line.strip()) for line in path.read_text(
        encoding="ascii").splitlines() if line.strip()]
            for name, path in SETS.items()}
    if any(len(items) != 1000 for items in rows.values()):
        raise RuntimeError("expected 1000 records in every evaluation set")
    keyed: dict[bytes, list[tuple[str, int]]] = {}
    for name, items in rows.items():
        for i, raw in enumerate(items):
            keyed.setdefault(key(raw), []).append((name, i))
    overlapped = {name: set() for name in SETS}
    copies = {split: {name: 0 for name in SETS} for split in ("train", "val")}
    seen = {split: {name: set() for name in SETS} for split in ("train", "val")}
    for split, path in (("train", TRAIN_V1), ("val", VAL_V1)):
        with path.open(encoding="ascii") as stream:
            for line in stream:
                if matches := keyed.get(key(bytes.fromhex(line.strip()))):
                    for name, index in matches:
                        overlapped[name].add(index)
                        seen[split][name].add(index)
                        copies[split][name] += 1
    result = {"v1_train": str(TRAIN_V1), "v1_val": str(VAL_V1),
              "protocol": "drop normalized train-or-checkpoint-validation-overlap records from previously exact-integer-scored per-record nats; pooled bytes+EOS from BOS; paired 5000-resample record bootstrap",
              "sets": {}}
    for name in SETS:
        flame = json.loads((OUT / f"eval_{name}.json").read_text(
            encoding="utf-8"))["flame"]
        ember = json.loads((EMBER / f"eval_{name}.json").read_text(
            encoding="utf-8"))["candidate"]
        mature = json.loads((MATURE / f"eval_{name}.json").read_text(
            encoding="utf-8"))["old_best"]
        if any(row["counts"] != flame["counts"] for row in (ember, mature)):
            raise RuntimeError(f"record count alignment failed for {name}")
        keep = [i for i in range(1000) if i not in overlapped[name]]
        filtered = {}
        for label, comparator in (("ember_refit", ember),
                                  ("mature_everyday", mature)):
            filtered[label] = {
                "bits": score(comparator, keep),
                "flame_minus": list(paired_bootstrap(
                    [flame["nats"][i] for i in keep],
                    [comparator["nats"][i] for i in keep],
                    [flame["counts"][i] for i in keep], n_boot=5000)),
            }
        result["sets"][name] = {
            "excluded_eval_indices_zero_based": sorted(overlapped[name]),
            "train_copies": copies["train"][name],
            "val_copies": copies["val"][name],
            "train_eval_records": len(seen["train"][name]),
            "val_eval_records": len(seen["val"][name]),
            "retained_records": len(keep),
            "retained_targets": sum(flame["counts"][i] for i in keep),
            "flame_bits": score(flame, keep), "comparators": filtered,
        }
    path = OUT / "overlap_sensitivity.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["sets"], indent=2), flush=True)


if __name__ == "__main__":
    main()
