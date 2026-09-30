#!/usr/bin/env python3
"""Evaluate the optional train-corpus retrieval decoder on fixed prompts.

This is a qualitative/coverage check, not a language-model likelihood test.
Each suggestion includes its train-line provenance. Prompts should be held
separately from the train split used to construct the retrieval index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))


def file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_prompts(path: Path, limit: int) -> list[dict]:
    source = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(source)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in source.splitlines() if line.strip()]
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("prompts", payload.get("rows"))
    else:
        items = None
    if not isinstance(items, list):
        raise ValueError("prompt file must be a list or contain a prompts/rows list")
    selected = []
    for item in items[:limit]:
        if isinstance(item, str):
            selected.append({"id": None, "prompt": item})
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            selected.append({"id": item.get("id"), "prompt": item["prompt"]})
        else:
            raise ValueError("each prompt must be a string or object with a prompt string")
    if not selected:
        raise ValueError("prompt file contains no selected prompts")
    return selected


def result_payload(config: dict, rows: list[dict]) -> dict:
    return {
        "config": config,
        "summary": {"prompts": len(rows),
                    "usable": sum(not row["abstained"] for row in rows),
                    "abstained": sum(row["abstained"] for row in rows),
                    "total_seconds": sum(row["seconds"] for row in rows)},
        "rows": rows,
    }


def save_result(path: Path, config: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result_payload(config, rows),
                                    ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--mode", choices=("short", "sentence"),
                        default="sentence")
    parser.add_argument("--max-words", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    info = json.loads((args.run / "run_info.json").read_text(encoding="utf-8"))
    os.environ.update({key: str(value) for key, value in info["env"].items()})
    from mica_r1 import serialize
    from mica_r1.retrieval_suggest import SentenceIndex, suggest_retrieval
    from mica_r1.suggest import MicaScorer

    prompts = load_prompts(args.prompts, args.limit)
    config = {"model_sha256": file_sha(args.run / "best.mica"),
              "index_sha256": file_sha(args.index / "sentences.jsonl"),
              "prompts_sha256": file_sha(args.prompts),
              "decoder_sha256": file_sha(R1 / "mica_r1/retrieval_suggest.py"),
              "mode": args.mode, "max_words": args.max_words,
              "limit": args.limit}
    if args.out.exists():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        if previous.get("config") != config:
            raise ValueError("existing result uses a different evaluation configuration")
        rows = previous.get("rows")
        if not isinstance(rows, list) or len(rows) > len(prompts):
            raise ValueError("existing result has invalid rows")
        for existing, expected in zip(rows, prompts):
            if existing.get("id") != expected["id"] or existing.get("prompt") != expected["prompt"]:
                raise ValueError("existing result does not match the prompt order")
    else:
        rows = []
    if len(rows) < len(prompts):
        index = SentenceIndex.load(args.index)
        scorer = MicaScorer(serialize.load(args.run / "best.mica"))
    for item in prompts[len(rows):]:
        prompt = item["prompt"]
        start = time.perf_counter()
        result = suggest_retrieval(scorer, index, prompt, mode=args.mode,
                                   max_words=args.max_words)
        rows.append({"id": item["id"],
                     "prompt": prompt, "abstained": result.abstained,
                     "reason": result.reason,
                     "seconds": time.perf_counter() - start,
                     "suggestions": [s.__dict__ for s in result.suggestions]})
        save_result(args.out, config, rows)
    payload = result_payload(config, rows)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
