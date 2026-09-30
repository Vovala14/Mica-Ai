#!/usr/bin/env python3
"""Turn GSM8K into MICA records: one grade-school word problem per record.

Why a single problem per record, rather than packing the corpus into fixed
chunks the way data/make_records.py does for prose:

  The task is "given a question, produce a worked solution". A record that
  starts mid-problem teaches the model to continue arbitrary text; a record
  that holds exactly one problem teaches it that a record BEGINS with a
  question and ENDS after the answer. The EOS prediction then carries real
  information, and generation has a natural stopping point.

Record format, after stripping GSM8K's <<48/2=24>> calculator annotations
(they are an artefact of how the dataset was built, not part of the answer):

    Q: Natalia sold clips to 48 of her friends in April, ...
    A: Natalia sold 48/2 = 24 clips in May.
    Natalia sold 48+24 = 72 clips altogether in April and May.
    #### 72

The "#### n" line is GSM8K's own final-answer marker and is what the accuracy
harness parses, so it is kept verbatim.

Splits are by PROBLEM, and GSM8K's own test.jsonl is passed through untouched
as the locked test set, so nothing in it can reach training.
"""

from __future__ import annotations

import argparse, json, re
from pathlib import Path

CALC = re.compile(r"<<[^>]*>>")
FINAL = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def render(row: dict) -> str:
    q = row["question"].strip()
    a = CALC.sub("", row["answer"]).strip()
    return f"Q: {q}\nA: {a}\n"


def final_answer(text: str) -> str | None:
    m = FINAL.search(text)
    return m.group(1).replace(",", "") if m else None


def utf8_safe_cut(buf: bytes, limit: int) -> int:
    if len(buf) <= limit:
        return len(buf)
    k = limit
    while k > 0 and (buf[k] & 0xC0) == 0x80:
        k -= 1
    return k or limit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k", required=True,
                    help="directory holding GSM8K train.jsonl and test.jsonl")
    ap.add_argument("--out", default="r1/data/gsm8k")
    ap.add_argument("--record-bytes", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--drop-oversize", action="store_true",
                    help="drop problems longer than the record length instead "
                         "of truncating them. Truncating teaches the model to "
                         "stop mid-solution, so this is usually what you want.")
    args = ap.parse_args()

    src = Path(args.gsm8k)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = __import__("random").Random(args.seed)

    train_rows = [json.loads(l) for l in (src / "train.jsonl").open()]
    test_rows = [json.loads(l) for l in (src / "test.jsonl").open()]

    # Deduplicate on the question text. GSM8K is clean, but a duplicate that
    # straddles train and validation would quietly invalidate every number.
    seen, uniq = set(), []
    for r in train_rows:
        k = " ".join(r["question"].split()).lower()
        if k not in seen:
            seen.add(k); uniq.append(r)
    test_keys = {" ".join(r["question"].split()).lower() for r in test_rows}
    leaked = [r for r in uniq if
              " ".join(r["question"].split()).lower() in test_keys]
    uniq = [r for r in uniq if
            " ".join(r["question"].split()).lower() not in test_keys]

    rng.shuffle(uniq)
    n_val = int(len(uniq) * args.val_frac)
    splits = {"val": uniq[:n_val], "train": uniq[n_val:], "test": test_rows}

    stats = {}
    for name, rows in splits.items():
        recs, dropped, truncated = [], 0, 0
        for r in rows:
            b = render(r).encode()
            if len(b) > args.record_bytes:
                if args.drop_oversize:
                    dropped += 1
                    continue
                b = b[:utf8_safe_cut(b, args.record_bytes)]
                truncated += 1
            recs.append(b)
        (out / f"{name}.jsonl").write_text(
            "".join(r.hex() + "\n" for r in recs))
        stats[name] = {"records": len(recs), "bytes": sum(len(r) for r in recs),
                       "dropped": dropped, "truncated": truncated}

    # The accuracy harness needs the question alone and the gold final answer.
    graded = []
    for r in test_rows:
        fa = final_answer(r["answer"])
        if fa is not None:
            graded.append({"prompt": f"Q: {r['question'].strip()}\nA:",
                           "final": fa})
    (out / "test_prompts.jsonl").write_text(
        "".join(json.dumps(g) + "\n" for g in graded))

    manifest = {"source": str(src), "record_bytes": args.record_bytes,
                "duplicates_removed": len(train_rows) - len(uniq) - len(leaked),
                "train_rows_also_in_test": len(leaked),
                "splits": stats, "graded_test_prompts": len(graded)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for name, st in stats.items():
        print(f"  {name:>5}: {st['records']:>6,} problems  "
              f"{st['bytes']/1e6:>6.2f} MB  "
              f"dropped {st['dropped']}  truncated {st['truncated']}")
    print(f"  graded test prompts: {len(graded):,}")
    if leaked:
        print(f"  removed {len(leaked)} training rows that also appear in test")
    total = sum(st["bytes"] for st in stats.values())
    print(f"\n  total {total/1e6:.2f} MB of text.")
    if total < 50e6:
        print("  WARNING: a multi-megabyte model wants hundreds of MB of text.")
        print("  GSM8K alone is a format rehearsal, not a training set. Scale")
        print("  up with AI-MO/NuminaMath-CoT (1.23 GB, Apache-2.0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
