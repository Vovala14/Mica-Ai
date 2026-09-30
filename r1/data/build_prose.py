#!/usr/bin/env python3
"""Build bounded, English prose records from explicit local Parquet shards.

Example (use the Python environment with pyarrow installed)::

    python r1/data/build_prose.py \
      --simple-wiki ../data_raw/wikimedia__wikipedia/20231101.simple/20231101.simple/train-00000-of-00001.parquet \
      --fineweb ../data_raw/HuggingFaceFW__fineweb/sample/sample/10BT/000_00000.parquet \
      --out r1/data/prose_v1

The input is scanned in small Arrow batches. All limits are explicit; even the
defaults stop well before exhausting the downloaded shards. A document gets
one deterministic split based on normalized content. Exact normalized
documents and records are deduplicated globally in a temporary SQLite file.
Each output line is the hex encoding expected by MICA's trainers. A readable
text copy and a provenance/filter manifest are also written.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import time
import unicodedata


SPLITS = ("train", "val", "test")
WHITESPACE = re.compile(r"\s+")
BLANK_LINES = re.compile(r"\n\s*\n+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]*[A-Za-z0-9])")
URL = re.compile(r"(?:https?://|www\.|\b\w+@\w+\.\w+\b)", re.I)
MATH = re.compile(
    r"(?:\\(?:frac|sum|int|sqrt|begin|end)\b|\$[^$]{1,120}\$|"
    r"\b(?:solve|calculate|simplify|evaluate|compute)\b.{0,50}[=+*/^]|"
    r"\b\d+(?:\.\d+)?\s*[=+*/^]\s*\d+(?:\.\d+)?\b)", re.I
)
BOILERPLATE = re.compile(
    r"(?:cookie(?:s)? (?:policy|settings|consent)|privacy policy|"
    r"terms (?:of use|and conditions|of service)|subscribe to (?:our|the) newsletter|"
    r"sign (?:in|up) to (?:continue|read)|all rights reserved|"
    r"click here to (?:read|learn|subscribe)|javascript (?:is required|disabled)|"
    r"advertisement|skip to (?:main )?content)", re.I
)
LIST_LINE = re.compile(r"^(?:[-*•]\s|\d{1,3}[.)]\s|[#=]{2,}|\|)")
CANON = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                       "–": "-", "—": "-", "\u00a0": " "})


def normalized(text: str) -> str:
    return WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)
                          .translate(CANON)).strip()


def digest(text: str) -> bytes:
    return hashlib.sha256(normalized(text).casefold().encode("utf-8")).digest()


def split_of(document_hash: bytes) -> str:
    n = int.from_bytes(document_hash[:8], "big")
    if n < (1 << 64) * 90 // 100:
        return "train"
    if n < (1 << 64) * 95 // 100:
        return "val"
    return "test"


def sampled(document_hash: bytes, rate: float) -> bool:
    """Content-based sampling independent of the bytes used for split assignment."""
    return int.from_bytes(document_hash[8:16], "big") < (1 << 64) * rate


def paragraph_status(paragraph: str) -> str | None:
    """Return a reject reason, or None for likely ordinary English prose."""
    if len(paragraph.encode("utf-8")) < 48:
        return "short"
    if URL.search(paragraph):
        return "url_or_email"
    if BOILERPLATE.search(paragraph):
        return "boilerplate"
    if MATH.search(paragraph) or paragraph.count("$") >= 2:
        return "math"
    if "<" in paragraph and ">" in paragraph:
        return "markup"
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if lines and sum(bool(LIST_LINE.match(line)) for line in lines) * 2 >= len(lines):
        return "list"
    plain = normalized(paragraph)
    if not re.search(r"[.!?](?:[\"'”’)]*)$", plain):
        return "no_sentence_end"
    if len(re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", plain)) < 9:
        return "few_words"
    if len(re.findall(r"[A-Za-z]", plain)) / max(len(plain), 1) < 0.55:
        return "low_letter_ratio"
    if len(re.findall(r"\d", plain)) / max(len(plain), 1) > 0.12:
        return "number_heavy"
    return None


def records_of(paragraph: str, limit: int = 256, counts: Counter | None = None):
    """Pack complete sentences within one paragraph, skipping overlong ones."""
    clean = normalized(paragraph)
    pending = ""
    for sentence in SENTENCE_END.split(clean):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence.encode("utf-8")) > limit:
            if pending:
                yield pending
                pending = ""
            if counts is not None:
                counts["long_sentences_rejected"] += 1
            continue
        candidate = f"{pending} {sentence}" if pending else sentence
        if len(candidate.encode("utf-8")) <= limit:
            pending = candidate
        else:
            yield pending
            pending = sentence
    if pending:
        yield pending


def without_headings(paragraph: str) -> str:
    """Remove short standalone headings before joining wrapped prose lines."""
    kept = []
    for line in paragraph.splitlines():
        line = line.strip()
        if not line:
            continue
        words = re.findall(r"\b[A-Za-z]+\b", line)
        if (len(line) <= 72 and 1 <= len(words) <= 5 and
                line[0].isupper() and not re.search(r"[.!?;:,]$", line)):
            continue
        kept.append(line)
    return " ".join(kept)


def usable_records(text: str, limit: int, counts: Counter):
    """Yield candidate records without retaining the full document's output."""
    for paragraph in BLANK_LINES.split(text.replace("\r\n", "\n")):
        if not paragraph.strip():
            continue
        counts["paragraphs_seen"] += 1
        reason = paragraph_status(paragraph)
        if reason:
            counts[f"paragraph_reject_{reason}"] += 1
            continue
        counts["paragraphs_kept"] += 1
        for record in records_of(without_headings(paragraph), limit, counts):
            if len(record.encode("utf-8")) >= 40:
                yield record
            else:
                counts["short_record_rejected"] += 1


class RecordWriter:
    def __init__(self, out: Path):
        self.hex_files = {s: (out / f"{s}.jsonl").open("w", encoding="ascii")
                          for s in SPLITS}
        self.txt_files = {s: (out / f"{s}.txt").open("w", encoding="utf-8")
                          for s in SPLITS}
        self.hashes = {s: hashlib.sha256() for s in SPLITS}
        self.counts = {s: Counter() for s in SPLITS}

    def write(self, split: str, records: list[str]):
        self.counts[split]["documents"] += 1
        for record in records:
            data = record.encode("utf-8")
            line = data.hex().encode("ascii") + b"\n"
            self.hex_files[split].write(line.decode("ascii"))
            self.txt_files[split].write(record + "\n")
            self.hashes[split].update(line)
            self.counts[split]["records"] += 1
            self.counts[split]["bytes"] += len(data)

    def close(self):
        for file in [*self.hex_files.values(), *self.txt_files.values()]:
            file.close()

    def flush(self):
        for file in [*self.hex_files.values(), *self.txt_files.values()]:
            file.flush()


def source_info(path: Path, kind: str):
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(path)
    if "openmath" in str(path).lower():
        raise ValueError(f"math dataset is not a prose source: {path}")
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    required = {"id", "text"}
    if kind == "fineweb":
        required |= {"language", "language_score"}
    if not required <= names:
        raise ValueError(f"{kind} needs columns {sorted(required)}: {path}")
    stat = path.stat()
    info = {"kind": kind, "path": str(path.resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "parquet_rows": parquet.metadata.num_rows,
            "row_groups": parquet.num_row_groups, "columns_read": sorted(required)}
    return parquet, info


def build(args) -> dict:
    if not args.simple_wiki and not args.fineweb:
        raise ValueError("provide --simple-wiki and/or --fineweb")
    if not 1 <= args.record_limit <= 256:
        raise ValueError("--record-limit must be between 1 and 256")
    for name in ("max_source_rows", "max_source_documents", "max_records",
                 "max_output_mb", "max_document_bytes", "max_document_records",
                 "batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("simple_wiki_sample_rate", "fineweb_sample_rate"):
        if not 0 < getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output folder is not empty: {out}")
    sources = [("simple_wiki", Path(p)) for p in args.simple_wiki]
    sources += [("fineweb", Path(p)) for p in args.fineweb]
    opened = [(kind, *source_info(path, kind)) for kind, path in sources]
    out.mkdir(parents=True, exist_ok=True)
    writer = RecordWriter(out)
    source_reports = []
    total_records = total_bytes = 0
    max_bytes = int(args.max_output_mb * 1024 * 1024)
    global_stop = False
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="mica_prose_dedup_") as temp:
            db = sqlite3.connect(str(Path(temp) / "seen.sqlite"))
            try:
                db.execute("PRAGMA journal_mode=OFF")  # disposable working index
                db.execute("PRAGMA synchronous=OFF")
                db.execute("PRAGMA cache_size=-16384")  # 16 MiB cache limit
                db.execute("CREATE TABLE documents (hash BLOB PRIMARY KEY)")
                db.execute("CREATE TABLE records (hash BLOB PRIMARY KEY)")
                for kind, parquet, info in opened:
                    counts = Counter()
                    columns = info["columns_read"]
                    stop = "source_exhausted"
                    for batch in parquet.iter_batches(batch_size=args.batch_size,
                                                      columns=columns):
                        for row in batch.to_pylist():
                            if counts["rows_seen"] >= args.max_source_rows:
                                stop = "row_limit"
                                break
                            if counts["documents_kept"] >= args.max_source_documents:
                                stop = "document_limit"
                                break
                            counts["rows_seen"] += 1
                            if counts["rows_seen"] % 10_000 == 0:
                                writer.flush()
                                print(
                                    f"[prose] {kind}: rows={counts['rows_seen']:,}/"
                                    f"{info['parquet_rows']:,} docs={counts['documents_kept']:,} "
                                    f"records={total_records:,} text={total_bytes / 2**20:.1f} MiB "
                                    f"elapsed={time.monotonic() - started:.0f}s",
                                    file=sys.stderr, flush=True,
                                )
                            if kind == "fineweb":
                                if (row.get("language") != "en" or
                                        (row.get("language_score") or 0) < args.min_language_score):
                                    counts["row_reject_language"] += 1
                                    continue
                            text = row.get("text")
                            if not isinstance(text, str) or not text.strip():
                                counts["row_reject_empty"] += 1
                                continue
                            size = len(text.encode("utf-8"))
                            if size > args.max_document_bytes:
                                counts["row_reject_large"] += 1
                                continue
                            doc_hash = digest(text)
                            rate = (args.fineweb_sample_rate if kind == "fineweb"
                                    else args.simple_wiki_sample_rate)
                            if not sampled(doc_hash, rate):
                                counts["row_reject_sampling"] += 1
                                continue
                            if db.execute("SELECT 1 FROM documents WHERE hash=?",
                                          (doc_hash,)).fetchone():
                                counts["row_reject_duplicate_document"] += 1
                                continue
                            candidates = list(islice(
                                usable_records(text, args.record_limit, counts),
                                args.max_document_records + 1,
                            ))
                            if not candidates:
                                counts["row_reject_no_prose"] += 1
                                continue
                            if len(candidates) > args.max_document_records:
                                candidates = candidates[:args.max_document_records]
                                counts["documents_truncated_at_record_cap"] += 1
                            fresh = []
                            hashes = set()
                            for record in candidates:
                                h = digest(record)
                                if h in hashes or db.execute(
                                    "SELECT 1 FROM records WHERE hash=?", (h,)
                                ).fetchone():
                                    counts["duplicate_records_rejected"] += 1
                                    continue
                                hashes.add(h)
                                fresh.append(record)
                            if not fresh:
                                counts["row_reject_no_new_records"] += 1
                                continue
                            byte_count = sum(len(r.encode("utf-8")) for r in fresh)
                            if total_records + len(fresh) > args.max_records:
                                stop = "global_record_limit"
                                global_stop = True
                                break
                            if total_bytes + byte_count > max_bytes:
                                stop = "global_byte_limit"
                                global_stop = True
                                break
                            db.execute("INSERT INTO documents VALUES (?)", (doc_hash,))
                            db.executemany("INSERT INTO records VALUES (?)",
                                           [(h,) for h in hashes])
                            split = split_of(doc_hash)
                            writer.write(split, fresh)
                            counts["documents_kept"] += 1
                            counts["records_kept"] += len(fresh)
                            counts["bytes_kept"] += byte_count
                            counts[f"documents_{split}"] += 1
                            total_records += len(fresh)
                            total_bytes += byte_count
                        if stop != "source_exhausted":
                            break
                    db.commit()
                    source_reports.append({**info, "stop_reason": stop,
                                           "counts": dict(sorted(counts.items()))})
                    if global_stop:
                        break
            finally:
                db.close()
    finally:
        writer.close()
    report = {
        "format": "one UTF-8 record as lowercase hex per JSONL line",
        "split_rule": "SHA-256 of NFKC/whitespace-normalized casefolded whole document; 90/5/5",
        "dedup_rule": "global SHA-256 of NFKC/whitespace-normalized casefolded documents and records",
        "filter_count_scope": "paragraph counters cover material inspected through the first max_document_records+1 usable records per sampled document",
        "limits": {"max_source_rows": args.max_source_rows,
                   "max_source_documents": args.max_source_documents,
                   "max_records": args.max_records,
                   "max_output_bytes": max_bytes,
                   "max_document_bytes": args.max_document_bytes,
                   "max_document_records": args.max_document_records,
                   "record_limit_bytes": args.record_limit,
                   "batch_size": args.batch_size,
                   "min_fineweb_language_score": args.min_language_score,
                   "simple_wiki_sample_rate": args.simple_wiki_sample_rate,
                   "fineweb_sample_rate": args.fineweb_sample_rate},
        "sources": source_reports,
        "splits": {s: {**dict(writer.counts[s]),
                       "jsonl_sha256": writer.hashes[s].hexdigest(),
                       "jsonl_path": str(out / f"{s}.jsonl"),
                       "text_path": str(out / f"{s}.txt")}
                   for s in SPLITS},
        "totals": {"records": total_records, "text_bytes": total_bytes},
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n",
                                       encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simple-wiki", action="append", default=[], metavar="PARQUET",
                    help="explicit Simple English Wikipedia shard; repeatable")
    ap.add_argument("--fineweb", action="append", default=[], metavar="PARQUET",
                    help="explicit FineWeb shard; repeatable")
    ap.add_argument("--out", required=True, help="new or empty output directory")
    ap.add_argument("--max-source-rows", type=int, default=100_000)
    ap.add_argument("--max-source-documents", type=int, default=10_000)
    ap.add_argument("--max-records", type=int, default=300_000)
    ap.add_argument("--max-output-mb", type=float, default=64)
    ap.add_argument("--max-document-bytes", type=int, default=50_000)
    ap.add_argument("--max-document-records", type=int, default=16)
    ap.add_argument("--record-limit", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--min-language-score", type=float, default=0.90)
    ap.add_argument("--simple-wiki-sample-rate", type=float, default=1.0,
                    help="deterministically keep this fraction of SimpleWiki documents")
    ap.add_argument("--fineweb-sample-rate", type=float, default=1.0,
                    help="deterministically keep this fraction of FineWeb documents")
    args = ap.parse_args()
    report = build(args)
    print(json.dumps({"totals": report["totals"],
                      "sources": [{"kind": s["kind"], "stop_reason": s["stop_reason"],
                                   "counts": s["counts"]} for s in report["sources"]],
                      "splits": {s: {k: v for k, v in d.items()
                                     if k in ("documents", "records", "bytes")}
                                 for s, d in report["splits"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
