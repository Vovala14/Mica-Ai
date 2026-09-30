#!/usr/bin/env python3
"""Blind the W2 letter arm. Prints count and sha only."""
from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

W2 = Path(__file__).resolve().parent / "w2"


def main() -> int:
    rows = []
    for path in sorted((W2 / "gen").glob("letter.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 100 or len({row["id"] for row in rows}) != 100:
        raise SystemExit(f"expected 100 unique, got {len(rows)}")
    if any(not row.get("continuation") for row in rows):
        raise SystemExit("empty continuation")
    secrets.SystemRandom().shuffle(rows)
    blind = []
    key = []
    for index, row in enumerate(rows):
        anon = f"L{index:04d}"
        blind.append({"id": anon, "context": row["prompt"], "suggestion": row["continuation"]})
        key.append({
            "id": anon,
            "prompt_id": row["id"],
            "model_sha256": row["model_sha256"],
            "decoding": row["decoding"],
            "ids": row["ids"],
        })
    (W2 / "blind").mkdir(parents=True, exist_ok=True)
    (W2 / "sealed").mkdir(parents=True, exist_ok=True)
    blind_path = W2 / "blind" / "items.jsonl"
    blind_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind), encoding="utf-8")
    (W2 / "sealed" / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"blind {len(blind)} sha256 {hashlib.sha256(blind_path.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
