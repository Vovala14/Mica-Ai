"""Build exact-deduplicated everyday evaluation samples without training data.

The original everyday split groups COCO captions by image, but the same
caption can describe multiple images. This removes case/whitespace-normalized
record duplicates across train, validation, and test before selecting fixed
evaluation samples. It does not rewrite the training data or model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

R1 = Path(__file__).resolve().parents[1]


def key(raw: bytes) -> bytes:
    text = raw.decode("utf-8", errors="replace").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def records(path: Path):
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield bytes.fromhex(line)[:256]


def clean_pool(path: Path, excluded: set[bytes], *, skip: int = 0):
    pool: list[bytes] = []
    seen: set[bytes] = set()
    excluded_count = duplicate_count = 0
    for index, raw in enumerate(records(path)):
        digest = key(raw)
        if digest in excluded:
            excluded_count += 1
        elif digest in seen:
            duplicate_count += 1
        elif index >= skip:
            pool.append(raw)
        seen.add(digest)
    return pool, seen, {"excluded": excluded_count,
                        "within_split_duplicates": duplicate_count,
                        "eligible": len(pool)}


def choose(pool: list[bytes], n: int) -> list[bytes]:
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} clean records; requested {n}")
    return pool[::max(1, len(pool) // n)][:n]


def write(path: Path, sample: list[bytes]) -> str:
    path.write_text("".join(raw.hex() + "\n" for raw in sample),
                    encoding="ascii")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=R1 / "data/everyday/train.jsonl")
    ap.add_argument("--val", type=Path, default=R1 / "data/everyday/val.jsonl")
    ap.add_argument("--test", type=Path, default=R1 / "data/everyday/test.jsonl")
    ap.add_argument("--out", type=Path, default=R1 / "data/eval_clean")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--val-skip", type=int, default=256)
    a = ap.parse_args()
    if a.n < 1 or a.val_skip < 0:
        ap.error("--n must be positive and --val-skip nonnegative")

    train_keys = {key(raw) for raw in records(a.train)}
    val_pool, val_keys, val_info = clean_pool(a.val, train_keys,
                                               skip=a.val_skip)
    test_pool, _, test_info = clean_pool(a.test, train_keys | val_keys)
    val_sample = choose(val_pool, a.n)
    test_sample = choose(test_pool, a.n)
    a.out.mkdir(parents=True, exist_ok=True)
    val_hash = write(a.out / f"val{a.n}.jsonl", val_sample)
    test_hash = write(a.out / f"test{a.n}.jsonl", test_sample)
    assert not ({key(r) for r in val_sample} & train_keys)
    assert not ({key(r) for r in test_sample} & (train_keys | val_keys))
    manifest = {
        "normalization": "UTF-8 replacement, casefold, whitespace collapse, strip",
        "train_file": str(a.train.resolve()),
        "val_file": str(a.val.resolve()),
        "test_file": str(a.test.resolve()), "train_unique": len(train_keys),
        "val_skip": a.val_skip, "val": val_info, "test": test_info,
        "selected_records_per_split": a.n,
        "val_sha256": val_hash, "test_sha256": test_hash,
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
