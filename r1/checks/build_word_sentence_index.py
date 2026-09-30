#!/usr/bin/env python3
"""Sample complete sentences from training text for a word-generation pilot."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import re


BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'\u201c\u2018(]*[A-Z])")
WORD = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*")


def candidates(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            for part in BOUNDARY.split(line.strip()):
                sentence = part.strip(' \t\r\n\"\u201c\u201d')
                if not sentence or sentence[-1] not in ".!?":
                    continue
                if not sentence[0].isupper():
                    continue
                words = WORD.findall(sentence)
                if not 5 <= len(words) <= 35:
                    continue
                if sum(map(len, words)) < 0.68 * len(sentence):
                    continue
                yield sentence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-sentences", type=int, default=200_000)
    args = parser.parse_args()
    if args.max_sentences < 1:
        parser.error("--max-sentences must be positive")

    # Keep the lowest stable hashes, so source order does not bias the sample.
    heap: list[tuple[int, str]] = []
    seen: set[int] = set()
    eligible = 0
    for sentence in candidates(args.train_text):
        eligible += 1
        digest = int.from_bytes(hashlib.sha256(
            sentence.casefold().encode("utf-8")).digest()[:16], "big")
        if digest in seen:
            continue
        entry = (-digest, sentence)
        if len(heap) < args.max_sentences:
            heapq.heappush(heap, entry)
            seen.add(digest)
        elif digest < -heap[0][0]:
            removed = heapq.heapreplace(heap, entry)
            seen.remove(-removed[0])
            seen.add(digest)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as stream:
        for _, sentence in sorted(heap, reverse=True):
            stream.write(json.dumps({"text": sentence}, ensure_ascii=False) + "\n")
    print(json.dumps({"eligible_occurrences": eligible,
                      "sampled_unique_sentences": len(heap),
                      "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
