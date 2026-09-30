#!/usr/bin/env python3
"""Choose disjoint, reproducible prose evaluation records across whole splits.

Run after build_prose.py has written its manifest, for example::

    python r1/checks/make_prose_eval.py --corpus r1/data/prose_fullscan_320mb_r8 --out r1/data/prose_eval_v1

The checkpoint file contains 256 validation records sampled across the entire
validation split. The two audit files contain 1,000 other validation records
and 1,000 test records. The source corpus is read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_split(corpus: Path, split: str, declared: dict) -> tuple[list[bytes], dict]:
    """Verify the builder's logical-LF hash and return byte records."""
    path = corpus / f"{split}.jsonl"
    file_bytes = path.read_bytes()
    if file_bytes and not file_bytes.endswith(b"\n"):
        raise ValueError(f"{path}: final record lacks a newline")
    lines = file_bytes.splitlines()
    records = []
    logical_hash = hashlib.sha256()
    for index, line in enumerate(lines):
        try:
            raw = bytes.fromhex(line.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid hex at record {index}") from exc
        if not raw or len(raw) > 256 or raw.hex().encode("ascii") != line:
            raise ValueError(f"{path}: invalid byte record at index {index}")
        records.append(raw)
        logical_hash.update(line + b"\n")
    logical_sha = logical_hash.hexdigest()
    if len(records) != declared.get("records"):
        raise ValueError(f"{path}: record count differs from builder manifest")
    if logical_sha != declared.get("jsonl_sha256"):
        raise ValueError(f"{path}: content hash differs from builder manifest")
    return records, {"file": str(path), "records": len(records),
                     "file_sha256": sha256(file_bytes),
                     "builder_logical_sha256": logical_sha}


def pick(total: int, count: int, seed: str) -> list[int]:
    if count < 1 or total < count:
        raise ValueError(f"need {count} records but split has {total}")
    return random.Random(seed).sample(range(total), count)


def describe_sample(filename: str, indices: list[int], source: list[bytes],
                    out: Path) -> dict:
    # Sort by original index for stable human inspection. Selection itself is
    # uniform across the entire split, including its later source blocks.
    indices = sorted(indices)
    records = [source[i] for i in indices]
    payload = b"".join(raw.hex().encode("ascii") + b"\n" for raw in records)
    (out / filename).write_bytes(payload)
    return {"file": filename, "records": len(records),
            "text_bytes": sum(map(len, records)),
            "targets_with_eos": sum(len(raw) + 1 for raw in records),
            "file_sha256": sha256(payload), "source_indices": indices,
            "record_sha256": [sha256(raw) for raw in records]}


def build(corpus: Path, out: Path, *, seed: int = 20260926,
          checkpoint_n: int = 256, val_audit_n: int = 1000,
          test_audit_n: int = 1000) -> dict:
    corpus, out = corpus.resolve(), out.resolve()
    if min(checkpoint_n, val_audit_n, test_audit_n) < 1:
        raise ValueError("each sample size must be positive")
    if out == corpus or corpus in out.parents:
        raise ValueError("output must be outside the source corpus")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    manifest_path = corpus / "manifest.json"
    builder_manifest_bytes = manifest_path.read_bytes()
    builder_manifest = json.loads(builder_manifest_bytes)
    splits = builder_manifest["splits"]
    val, val_info = load_split(corpus, "val", splits["val"])
    test, test_info = load_split(corpus, "test", splits["test"])

    val_indices = pick(len(val), checkpoint_n + val_audit_n,
                       f"{seed}:val")
    checkpoint_indices = val_indices[:checkpoint_n]
    audit_val_indices = val_indices[checkpoint_n:]
    audit_test_indices = pick(len(test), test_audit_n, f"{seed}:test")
    groups = ({val[i] for i in checkpoint_indices},
              {val[i] for i in audit_val_indices},
              {test[i] for i in audit_test_indices})
    if (any(len(group) != count for group, count in zip(
            groups, (checkpoint_n, val_audit_n, test_audit_n))) or
            groups[0] & groups[1] or groups[0] & groups[2] or
            groups[1] & groups[2]):
        raise ValueError("selected evaluation records overlap")

    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "checkpoint_val": describe_sample(f"checkpoint_val{checkpoint_n}.jsonl",
                                          checkpoint_indices, val, out),
        "audit_val": describe_sample(f"audit_val{val_audit_n}.jsonl",
                                     audit_val_indices, val, out),
        "audit_test": describe_sample(f"audit_test{test_audit_n}.jsonl",
                                      audit_test_indices, test, out),
    }
    report = {"seed": seed, "selection": "uniform seeded sample of indices",
              "builder_manifest_sha256": sha256(builder_manifest_bytes),
              "source_corpus": str(corpus),
              "inputs": {"val": val_info, "test": test_info},
              "outputs": outputs, "pairwise_overlap": 0}
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--checkpoint-n", type=int, default=256)
    ap.add_argument("--val-audit-n", type=int, default=1000)
    ap.add_argument("--test-audit-n", type=int, default=1000)
    a = ap.parse_args()
    report = build(a.corpus, a.out, seed=a.seed,
                   checkpoint_n=a.checkpoint_n,
                   val_audit_n=a.val_audit_n, test_audit_n=a.test_audit_n)
    print(json.dumps({"inputs": report["inputs"],
                      "outputs": {name: {k: value for k, value in info.items()
                                         if k not in ("source_indices", "record_sha256")}
                                  for name, info in report["outputs"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
