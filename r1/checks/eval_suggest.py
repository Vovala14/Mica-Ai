#!/usr/bin/env python3
"""Run the optional word suggester on the fixed MICA completion prompts.

The vocabulary comes only from the training split. Results are checkpointed
after each prompt so an interrupted benchmark can resume without repeating it.
This measures output variety, context-pair differences, and latency; it does
not assign a semantic usefulness score to the suggestions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(rows: list[dict]) -> dict:
    usable = [r for r in rows if not r["abstained"] and r["suggestions"]]
    texts = [r["suggestions"][0]["continuation"] for r in usable]
    pairs: dict[int, dict[str, str]] = {}
    usable_pairs: dict[int, dict[str, str]] = {}
    for row in rows:
        if row.get("category") == "pair":
            text = (row["suggestions"][0]["continuation"]
                    if row["suggestions"] else "")
            pairs.setdefault(row["pair"], {})[row["side"]] = text
            if not row["abstained"] and text:
                usable_pairs.setdefault(row["pair"], {})[row["side"]] = text
    complete_pairs = [p for p in pairs.values() if "A" in p and "B" in p]
    complete_usable_pairs = [p for p in usable_pairs.values()
                             if "A" in p and "B" in p]
    times = sorted(r["seconds"] for r in rows)
    return {
        "completed": len(rows), "usable": len(usable),
        "abstained": len(rows) - len(usable),
        "distinct_top_continuations": len(set(texts)),
        "most_common": Counter(texts).most_common(10),
        "context_pairs_compared": len(complete_pairs),
        "context_pairs_different": sum(p["A"] != p["B"] for p in complete_pairs),
        "context_pairs_both_usable": len(complete_usable_pairs),
        "usable_context_pairs_different": sum(
            p["A"] != p["B"] for p in complete_usable_pairs),
        "median_seconds": statistics.median(times) if times else None,
        "p95_seconds": times[min(len(times) - 1, round(.95 * (len(times) - 1)))]
        if times else None,
    }


def save(path: Path, payload: dict) -> None:
    payload["summary"] = summary(payload["rows"])
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--prompts", type=Path,
                    default=R1 / "runs/checks/everyday_dev_greedy.json")
    ap.add_argument("--corpus", type=Path,
                    default=R1 / "data/everyday/train.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--category", choices=("all", "everyday", "pair"),
                    default="all", help="select a prompt category before limit")
    ap.add_argument("--max-vocab", type=int, default=5000)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--beam-width", type=int, default=2)
    ap.add_argument("--max-words", type=int, default=3)
    ap.add_argument("--min-words", type=int, default=1)
    ap.add_argument("--require-terminal-punctuation", action="store_true")
    ap.add_argument("--min-confidence", type=float, default=0.55)
    ap.add_argument("--prefix-completion", action="store_true",
                    help="allow extending the prompt's last word; default "
                         "treats benchmark prompts as complete words")
    args = ap.parse_args()
    if args.limit < 1:
        ap.error("--limit must be positive")

    info = json.loads((args.run / "run_info.json").read_text(encoding="utf-8"))
    os.environ.update({key: str(value) for key, value in info["env"].items()})
    from mica_r1 import serialize
    from mica_r1.suggest import MicaScorer, Vocabulary, suggest

    model_path = args.run / "best.mica"
    source_bytes = args.prompts.read_bytes()
    prompts = json.loads(source_bytes)["rows"]
    if args.category != "all":
        prompts = [row for row in prompts if row["category"] == args.category]
    prompts = prompts[:args.limit]
    config = {
        "model_sha256": file_sha256(model_path),
        "decoder_sha256": file_sha256(R1 / "mica_r1/suggest.py"),
        "prompts_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": file_sha256(args.corpus), "limit": args.limit,
        "category": args.category,
        "max_vocab": args.max_vocab, "min_count": args.min_count,
        "beam_width": args.beam_width, "max_words": args.max_words,
        "min_words": args.min_words,
        "require_terminal_punctuation": args.require_terminal_punctuation,
        "min_confidence": args.min_confidence,
        "complete_final_word": not args.prefix_completion,
    }
    if args.out.exists():
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        if payload["config"] != config:
            raise ValueError("existing result has a different model, prompts, or settings")
    else:
        payload = {"config": config, "rows": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scorer = MicaScorer(serialize.load(model_path))
    vocab = Vocabulary.from_corpus(args.corpus, max_words=args.max_vocab,
                                   min_count=args.min_count)
    print(f"[suggest] vocabulary: {vocab.size} words; "
          f"resuming at {len(payload['rows'])}/{len(prompts)}", flush=True)
    for item in prompts[len(payload["rows"]):]:
        start = time.perf_counter()
        result = suggest(scorer, vocab, item["prompt"],
                         beam_width=args.beam_width, max_words=args.max_words,
                         min_words=args.min_words,
                         require_terminal_punctuation=args.require_terminal_punctuation,
                         min_confidence=args.min_confidence,
                         complete_final_word=not args.prefix_completion)
        row = {k: item[k] for k in ("id", "category", "prompt")}
        for key in ("pair", "side"):
            if key in item:
                row[key] = item[key]
        row.update({"seconds": time.perf_counter() - start,
                    "abstained": result.abstained, "reason": result.reason,
                    "suggestions": [s.__dict__ for s in result.suggestions]})
        payload["rows"].append(row)
        save(args.out, payload)
        if len(payload["rows"]) % 10 == 0:
            print(f"[suggest] {len(payload['rows'])}/{len(prompts)} "
                  f"{payload['summary']}", flush=True)
    save(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
