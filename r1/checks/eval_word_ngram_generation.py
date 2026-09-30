#!/usr/bin/env python3
"""Archived word-count comparison; not a MICA inference path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normal(text: str) -> str:
    return " ".join(text.casefold().split())


def exact_training_matches(train: Path, sentences: set[str]) -> set[str]:
    """Scan complete train records and punctuation-delimited sentences once."""
    wanted = {normal(sentence) for sentence in sentences}
    found: set[str] = set()
    boundary = re.compile(r"(?<=[.!?])\s+(?=[\"'\u201c\u2018(]*[A-Za-z])")
    with train.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = line.strip()
            for part in (record, *boundary.split(record)):
                norm = normal(part)
                if norm in wanted:
                    found.add(norm)
            if found == wanted:
                break
    return found


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-sentences", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=14)
    parser.add_argument("--mica-weight", type=float, default=0.35)
    args = parser.parse_args()
    if min(args.max_sentences, args.seeds, args.max_words) < 1:
        parser.error("limits must be positive")
    info = json.loads((args.run / "run_info.json").read_text(encoding="utf-8"))
    os.environ.update({key: str(value) for key, value in info["env"].items()})
    from mica_r1 import serialize
    from mica_r1.suggest import MicaScorer
    from research_baselines.word_ngram import WordNgram

    prompts_payload = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompts = prompts_payload["rows"]
    config = {
        "model_sha256": sha(args.run / "best.mica"),
        "index_sha256": sha(args.index_file),
        "train_sha256": sha(args.train_text),
        "prompts_sha256": sha(args.prompts),
        "generator_sha256": sha(R1 / "research_baselines/word_ngram.py"),
        "max_sentences": args.max_sentences, "seeds": args.seeds,
        "max_words": args.max_words, "mica_weight": args.mica_weight,
    }
    if args.out.exists():
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        if payload.get("config") != config:
            raise ValueError("existing output has a different configuration")
    else:
        payload = {"config": config, "rows": []}
    start = time.perf_counter()
    word_model = WordNgram.from_index(args.index_file,
                                      max_sentences=args.max_sentences)
    train_seconds = time.perf_counter() - start
    scorer = MicaScorer(serialize.load(args.run / "best.mica"))
    jobs = [(item["id"], item["prompt"], seed)
            for item in prompts for seed in range(args.seeds)]
    if len(payload["rows"]) > len(jobs):
        raise ValueError("existing output has too many rows")
    for row, job in zip(payload["rows"], jobs):
        if (row["id"], row["prompt"], row["seed"]) != job:
            raise ValueError("existing output does not match prompt order")
    print(f"[word-gen] trained on {word_model.sentences} sentences in "
          f"{train_seconds:.1f}s; resuming {len(payload['rows'])}/{len(jobs)}",
          flush=True)
    for id_, prompt, seed in jobs[len(payload["rows"]):]:
        begin = time.perf_counter()
        sample = word_model.generate(scorer, prompt, seed=seed,
                                     max_words=args.max_words,
                                     mica_weight=args.mica_weight)
        payload["rows"].append({
            "id": id_, "prompt": prompt, "seed": seed,
            "seconds": time.perf_counter() - begin,
            "text": sample.text, "continuation": sample.continuation,
            "complete": sample.complete,
            "exact_index_match": sample.exact_index_match,
            "generated_words": sample.generated_words,
        })
        save(args.out, payload)
        print(f"[word-gen] {len(payload['rows'])}/{len(jobs)} "
              f"{sample.text!r}", flush=True)
    found = exact_training_matches(
        args.train_text, {row["text"] for row in payload["rows"]})
    for row in payload["rows"]:
        row["exact_train_match"] = normal(row["text"]) in found
    payload["summary"] = {
        "outputs": len(payload["rows"]),
        "distinct": len({row["text"] for row in payload["rows"]}),
        "model_completed": sum(row["complete"] for row in payload["rows"]),
        "exact_train_matches": sum(row["exact_train_match"]
                                   for row in payload["rows"]),
        "training_seconds": train_seconds,
        "generation_seconds": sum(row["seconds"] for row in payload["rows"]),
    }
    save(args.out, payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
