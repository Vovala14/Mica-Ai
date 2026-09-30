#!/usr/bin/env python3
"""Merge generations, drop model labels, write a sealed key. Prints counts only."""
from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

JUDGE = Path(__file__).resolve().parent
GEN = JUDGE / "gen"


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def main() -> int:
    flame = load_jsonl(GEN / "flamew.jsonl")
    letter_paths = sorted(GEN.glob("letter.shard*.jsonl"))
    if letter_paths:
        letter = [row for path in letter_paths for row in load_jsonl(path)]
    else:
        letter = load_jsonl(GEN / "letter.jsonl")
    if len(flame) != 100 or len(letter) != 100:
        raise SystemExit(f"expected 100/100, got flame {len(flame)} letter {len(letter)}")
    flame_ids = [row["id"] for row in flame]
    letter_ids = [row["id"] for row in letter]
    if flame_ids != letter_ids and sorted(flame_ids) != sorted(letter_ids):
        raise SystemExit("prompt ids differ")
    if len(set(flame_ids)) != 100 or len(set(letter_ids)) != 100:
        raise SystemExit("duplicate prompt ids")
    by_letter = {row["id"]: row for row in letter}
    items = []
    key = []
    for row in flame:
        other = by_letter[row["id"]]
        for system, source in (("flamew", row), ("letter", other)):
            items.append({
                "context": source["prompt"],
                "suggestion": source["continuation"],
                "prompt_id": source["id"],
                "system": system,
                "ids": source["ids"],
                "model_sha256": source["model_sha256"],
                "decoding": source["decoding"],
            })
    secret = secrets.SystemRandom()
    secret.shuffle(items)
    blind_dir = JUDGE / "blind"
    sealed = JUDGE / "sealed"
    blind_dir.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)
    blind_rows = []
    key_rows = []
    for index, item in enumerate(items):
        anon = f"B{index:04d}"
        blind_rows.append({
            "id": anon,
            "context": item["context"],
            "suggestion": item["suggestion"],
        })
        key_rows.append({
            "id": anon,
            "prompt_id": item["prompt_id"],
            "system": item["system"],
            "model_sha256": item["model_sha256"],
            "decoding": item["decoding"],
            "ids": item["ids"],
        })
    blind_path = blind_dir / "items.jsonl"
    key_path = sealed / "key.json"
    blind_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind_rows),
        encoding="utf-8",
    )
    key_path.write_text(json.dumps(key_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(blind_path.read_bytes()).hexdigest()
    print(f"blind {len(blind_rows)} sha256 {digest}")
    print(f"key {key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
