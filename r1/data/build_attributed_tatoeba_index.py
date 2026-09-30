#!/usr/bin/env python3
"""Build a source-linked, TRAIN-only Tatoeba sentence index for MICA.

The local everyday corpus discards source IDs when writing train.txt. This
builder recovers only Tatoeba sentences that occur *exactly* in that file,
requires their sentence-ID split to be train, and excludes normalized exact
matches in val.txt and test.txt. It emits the same sentence fields consumed by
SentenceIndex.load without changing the retrieval decoder.

This is an attribution preparation artifact, not a publication clearance.
The locally available TSV has ID/language/text but no author, and the only
local license assertion is the everyday corpus builder's source comment.
The manifest therefore always says publication_ready=false.
"""

from __future__ import annotations

import argparse
import bz2
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import re
import sys


WORD = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'\u201c\u2018(]*[A-Za-z])")
TERMINAL = re.compile(r"[.!?][\"'\u201d\u2019)]*$")
SPACE = re.compile(r"\s+")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sentences(record: str):
    for part in BOUNDARY.split(record.strip()):
        sentence = SPACE.sub(" ", part).strip()
        if sentence:
            yield sentence


def normalized(sentence: str) -> str:
    return SPACE.sub(" ", sentence).strip().casefold()


def quality(sentence: str) -> bool:
    words = WORD.findall(sentence)
    if not 7 <= len(words) <= 40 or len(sentence.encode("utf-8")) > 256:
        return False
    if not sentence[0].isupper() or not TERMINAL.search(sentence):
        return False
    low = sentence.lower()
    if "http" in low or "www." in low:
        return False
    if any(mark in sentence for mark in ("{", "}", "|", "\\", "=", "`")):
        return False
    if sum(ch.isalpha() for ch in sentence) / len(sentence) < 0.60:
        return False
    if sum(ch.isdigit() for ch in sentence) / len(sentence) > 0.10:
        return False
    return True


def split_of_tatoeba_id(sentence_id: str) -> str:
    digest = hashlib.sha256(("tatoeba:" + sentence_id).encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 90 else ("val" if bucket < 95 else "test")


def build(train_file: Path, tsv_file: Path, out_dir: Path, max_sentences: int,
          detailed_file: Path | None = None) -> dict:
    train = train_file.resolve()
    tsv = tsv_file.resolve()
    out = out_dir.resolve()
    if train.name != "train.txt":
        raise ValueError("source must be an everyday train.txt")
    if max_sentences < 1:
        raise ValueError("max_sentences must be positive")
    heldout = [train.with_name("val.txt"), train.with_name("test.txt")]
    for path in (train, *heldout, tsv):
        if not path.is_file():
            raise FileNotFoundError(path)
    detailed = detailed_file.resolve() if detailed_file is not None else None
    if detailed is not None and not detailed.is_file():
        raise FileNotFoundError(detailed)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"index directory is not empty: {out}")

    excluded = set()
    for path in heldout:
        with path.open("r", encoding="utf-8") as stream:
            for record in stream:
                excluded.update(normalized(s) for s in sentences(record))

    # Exact source text and train line matter: the candidate must truly have
    # appeared in this training split, rather than merely resemble a source.
    train_sentences: dict[str, tuple[int, int]] = {}
    counts = Counter()
    with train.open("r", encoding="utf-8") as stream:
        for line_no, record in enumerate(stream, 1):
            counts["train_records_scanned"] += 1
            for ordinal, sentence in enumerate(sentences(record)):
                if quality(sentence) and normalized(sentence) not in excluded:
                    train_sentences.setdefault(sentence, (line_no, ordinal))
    counts["eligible_unique_train_sentences"] = len(train_sentences)

    # Lowest sentence-content SHA-256 priorities give a bounded, stable sample
    # across the whole local Tatoeba export, independent of its row order.
    heap: list[tuple[int, str, str, int, int, int]] = []
    selected: set[str] = set()
    with bz2.open(tsv, "rt", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            counts["tatoeba_rows_scanned"] += 1
            parts = raw.rstrip("\n").split("\t", 2)
            if len(parts) != 3 or not parts[0].isdigit() or parts[1] != "eng":
                counts["malformed_or_non_english"] += 1
                continue
            sentence_id, _language, text = parts
            sentence = SPACE.sub(" ", text).strip()
            if split_of_tatoeba_id(sentence_id) != "train":
                counts["non_train_id_split"] += 1
                continue
            if not quality(sentence):
                counts["source_quality_rejected"] += 1
                continue
            if normalized(sentence) in excluded:
                counts["heldout_rejected"] += 1
                continue
            location = train_sentences.get(sentence)
            if location is None:
                counts["not_in_train_text"] += 1
                continue
            key = hashlib.sha256(normalized(sentence).encode("utf-8")).hexdigest()
            if key in selected:
                counts["duplicate_selected"] += 1
                continue
            priority = int(key[:16], 16)
            item = (-priority, key, sentence, int(sentence_id), location[0], location[1])
            if len(heap) < max_sentences:
                heapq.heappush(heap, item)
                selected.add(key)
            elif item > heap[0]:
                evicted = heapq.heapreplace(heap, item)
                selected.remove(evicted[1])
                selected.add(key)

    # The detailed export is independently downloaded and may be a later
    # weekly snapshot. Keep only exact ID-and-text matches so the source link
    # cannot silently point to a revised or removed sentence.
    owners: dict[int, str | None] = {}
    if detailed is not None:
        wanted = {row[3]: row[2] for row in heap}
        with bz2.open(detailed, "rt", encoding="utf-8", errors="replace") as stream:
            for raw in stream:
                counts["detailed_rows_scanned"] += 1
                parts = raw.rstrip("\n").split("\t")
                if len(parts) != 6 or not parts[0].isdigit() or parts[1] != "eng":
                    counts["detailed_malformed_or_non_english"] += 1
                    continue
                sid = int(parts[0])
                if sid not in wanted:
                    continue
                text = SPACE.sub(" ", parts[2]).strip()
                if text != wanted[sid]:
                    counts["detailed_text_mismatch"] += 1
                    continue
                username = parts[3].strip()
                owners[sid] = username if username and username != "\\N" else None
                counts["detailed_exact_matches"] += 1
        counts["detailed_selected_not_found_or_changed"] = len(wanted) - len(owners)
        counts["detailed_missing_owner"] = sum(owner is None for owner in owners.values())
        heap = [row for row in heap if row[3] in owners]

    out.mkdir(parents=True, exist_ok=True)
    index_file = out / "sentences.jsonl"
    attribution_file = out / "attribution.jsonl"
    with index_file.open("w", encoding="utf-8", newline="\n") as index_stream, \
         attribution_file.open("w", encoding="utf-8", newline="\n") as attr_stream:
        for _, key, sentence, sentence_id, train_line, ordinal in sorted(heap,
                key=lambda row: (row[1], row[3])):
            url = f"https://tatoeba.org/en/sentences/show/{sentence_id}"
            index_stream.write(json.dumps({
                "text": sentence,
                "train_line": train_line,
                "sentence_ordinal": ordinal,
                "normalized_sha256": key,
                "source_url": url,
                "source_title": f"Tatoeba sentence #{sentence_id}",
                "source_license": "CC BY 2.0 FR",
            }, ensure_ascii=False) + "\n")
            attr_stream.write(json.dumps({
                "normalized_sha256": key,
                "source_id": sentence_id,
                "source_owner": owners.get(sentence_id),
                "source_author": None,
                "source_url": url,
                "license_evidence": "https://tatoeba.org/en/downloads (export-level CC BY 2.0 FR; subset also CC0 1.0)",
                "source_text_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
            }, ensure_ascii=False) + "\n")

    manifest = {
        "format": "mica-retrieval-sentences-v1",
        "source_split": "train",
        "publication_ready": False,
        "source_provenance_ready": detailed is not None and bool(heap),
        "public_display_requirements": [
            "Label each completion as adapted from a Tatoeba sentence and ranked by MICA",
            "Link each displayed completion to its source sentence URL",
            "Credit Tatoeba with a link to https://tatoeba.org",
            "Link the CC BY 2.0 FR license at https://creativecommons.org/licenses/by/2.0/fr/",
        ],
        "attribution_status": ("Source ID/URL/title and detailed-export owner available; owner is not necessarily author; export-level license verified, individual CC0 status unavailable"
                               if detailed is not None else
                               "Source ID/URL/title available; owner absent from basic TSV; export-level license verified, individual CC0 status unavailable"),
        "license_evidence_url": "https://tatoeba.org/en/downloads",
        "attribution_guidance_url": "https://en.wiki.tatoeba.org/articles/show/faq",
        "license_scope": "Official Tatoeba downloads page says exports are CC BY 2.0 FR and a subset is also CC0 1.0; per-sentence attribution author is missing locally",
        "source_train_file": str(train),
        "source_train_sha256": sha256_file(train),
        "source_tatoeba_file": str(tsv),
        "source_tatoeba_sha256": sha256_file(tsv),
        "source_detailed_file": str(detailed) if detailed is not None else None,
        "source_detailed_url": ("https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_detailed.tsv.bz2"
                                if detailed is not None else None),
        "source_detailed_sha256": sha256_file(detailed) if detailed is not None else None,
        "excluded_files": [{"path": str(p), "sha256": sha256_file(p)} for p in heldout],
        "excluded_unique_sentences": len(excluded),
        "sampling": "minimum normalized sentence SHA-256 priority among exact Tatoeba train members",
        "max_sentences": max_sentences,
        "selected_sentences": len(heap),
        "index_sha256": sha256_file(index_file),
        "attribution_sha256": sha256_file(attribution_file),
        "counts": dict(counts),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--tatoeba", type=Path, required=True)
    ap.add_argument("--detailed", type=Path,
                    help="official English detailed export, for exact ID/text owner join")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-sentences", type=int, default=200_000)
    args = ap.parse_args()
    print(json.dumps(build(args.train, args.tatoeba, args.out,
                           args.max_sentences, args.detailed), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
