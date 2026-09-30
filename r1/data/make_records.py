#!/usr/bin/env python3
"""Build the R1 record set. Section 13, "Data boundaries".

  "Deduplicate before assigning whole documents to training, validation, and
   locked test sets in proportions 90,5,5. Split long training documents into
   at most 256-byte records at UTF boundaries."

Documents come from data/corpus (already deduplicated, with a per-document
index and SHA-256). This script only re-splits them 90/5/5 by document and
cuts each document into records at UTF-8 character boundaries.
"""

from __future__ import annotations

import argparse, hashlib, json, os, random, sys
from array import array
from pathlib import Path


def utf8_safe_cut(buf: bytes, limit: int) -> int:
    """Largest cut <= limit that lands on a UTF-8 character boundary."""
    if len(buf) <= limit:
        return len(buf)
    k = limit
    while k > 0 and (buf[k] & 0xC0) == 0x80:      # never cut inside a character
        k -= 1
    return k or limit


def records_of(doc: bytes, limit: int) -> list[bytes]:
    out, i = [], 0
    while i < len(doc):
        k = utf8_safe_cut(doc[i:], limit)
        chunk = doc[i:i + k]
        if chunk.strip():
            out.append(chunk)
        i += k
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="r1/data/records")
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    corpus, out = Path(args.corpus), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    docs = []
    for split in ("train", "val"):
        blob = (corpus / f"{split}.bin").read_bytes()
        index = json.loads((corpus / f"{split}_docs.json").read_text())
        for d in index:
            docs.append((blob[d["offset"]:d["offset"] + d["length"]], d))
    print(f"[records] {len(docs)} documents, {sum(len(b) for b, _ in docs)/1e6:.2f} MB")

    rng = random.Random(args.seed)
    rng.shuffle(docs)
    n = len(docs)
    n_tr, n_va = int(n * 0.90), int(n * 0.05)
    parts = {"train": docs[:n_tr], "val": docs[n_tr:n_tr + n_va],
             "test": docs[n_tr + n_va:]}

    summary = {"seed": args.seed, "record_limit": args.limit,
               "split_rule": "whole documents assigned 90/5/5 (section 13)",
               "splits": {}}
    for name, group in parts.items():
        recs = []
        for blob, _meta in group:
            recs.extend(records_of(blob, args.limit))
        path = out / f"{name}.jsonl"
        with open(path, "w") as fh:
            for r in recs:
                fh.write(r.hex() + "\n")
        total = sum(len(r) for r in recs)
        summary["splits"][name] = {
            "documents": len(group), "records": len(recs), "bytes": total,
            "mean_record_bytes": round(total / max(len(recs), 1), 1),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"[records] {name}: {len(group)} docs -> {len(recs):,} records, "
              f"{total/1e6:.2f} MB")
    (out / "records_manifest.json").write_text(json.dumps(summary, indent=2))
    return 0


def load_records(path) -> list[bytes] | list[array]:
    """Load byte records, or little-endian uint16 symbol records for Flame-W.

    The byte return type stays unchanged for existing callers and checkpoints.
    For word records ``len(record)`` counts symbols, including in slicing by
    ``--record-bytes`` (which means symbols under MICA_SYMBOLS > 258).
    """
    n_symbols = int(os.environ.get("MICA_SYMBOLS", "258"))
    if n_symbols == 258:
        with open(path) as fh:
            return [bytes.fromhex(line.strip()) for line in fh if line.strip()]
    if not 258 < n_symbols <= 65535:
        raise ValueError("MICA_SYMBOLS must be 258 or a uint16 word alphabet")
    records = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = bytes.fromhex(line.strip())
            if len(raw) % 2:
                raise ValueError("uint16 record has an odd byte count")
            record = array("H")
            record.frombytes(raw)
            if sys.byteorder != "little":
                record.byteswap()
            if any(symbol >= n_symbols - 2 for symbol in record):
                raise ValueError("record includes BOS, EOS or an invalid symbol")
            records.append(record)
    return records


if __name__ == "__main__":
    raise SystemExit(main())
