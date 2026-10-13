#!/usr/bin/env python3
"""Mix corrections-only site records into a base corpus for readout fitting.

Consumes an ingest_site_logs.py --corrections-only output directory. Records
stay in the existing hex format. The cap applies to records AND targets (bytes
for Ember, word symbols for Flame-W, plus EOS). No correction is oversampled.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def records(path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield bytes.fromhex(line.strip()).hex()


def targets(record, model):
    size = len(record) // 2
    if model == "flamew" and size % 2:
        raise ValueError("Flame-W records must contain whole uint16 symbols")
    return size // (2 if model == "flamew" else 1) + 1


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mix(base, site, model, out, heldout=(), fraction="0.05", seed=0):
    fraction = Fraction(str(fraction))
    if not 0 < fraction <= Fraction("0.05"):
        raise ValueError("Correction fraction must be greater than zero and at most 0.05")
    site = Path(site)
    manifest_path = site / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("corrections_only") is not True:
        raise ValueError("Re-ingest the export with --corrections-only")
    train, val = (site / model / f"{split}.jsonl" for split in ("train", "val"))
    for path in (train, val):
        if sha(path) != manifest["files"][f"{model}/{path.name}"]["sha256"]:
            raise ValueError("Site record hash differs from ingestion manifest")
    heldout = [val, *map(Path, heldout)]
    out = Path(out)
    inputs = [Path(base), manifest_path, train, *heldout]
    if out.resolve() in {p.resolve() for p in inputs} or out.with_suffix(".manifest.json").resolve() in {p.resolve() for p in inputs}:
        raise ValueError("Output must not overwrite an input")
    candidates = set(records(train))
    for path in heldout:
        candidates.difference_update(records(path))
    base_records = base_targets = 0
    for record in records(base):
        base_records += 1
        base_targets += targets(record, model)
        candidates.discard(record)
    if not base_records:
        raise ValueError("Base training corpus is empty")
    ratio = fraction / (1 - fraction)
    record_budget = int(base_records * ratio)
    target_budget = int(base_targets * ratio)
    selected, correction_targets = [], 0
    ordered = sorted(candidates, key=lambda r: (hashlib.sha256(f"{seed}:{r}".encode()).digest(), r))
    for record in ordered:
        cost = targets(record, model)
        if len(selected) >= record_budget:
            break
        if correction_targets + cost <= target_budget:
            selected.append(record)
            correction_targets += cost
    out.parent.mkdir(parents=True, exist_ok=True)
    # Spread the correction rows through the original corpus deterministically.
    with out.open("w", encoding="utf-8") as destination:
        cursor = 0
        for index, record in enumerate(records(base), 1):
            destination.write(record + "\n")
            while cursor < len(selected) and (cursor + 1) * base_records <= index * len(selected):
                destination.write(selected[cursor] + "\n")
                cursor += 1
    report = {"model": model, "seed": seed, "max_fraction": float(fraction),
              "base_records": base_records, "base_targets": base_targets,
              "correction_records": len(selected), "correction_targets": correction_targets,
              "eligible_corrections": len(candidates),
              "inputs": [{"path": str(p), "sha256": sha(p)} for p in inputs],
              "output_sha256": sha(out)}
    out.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--model", choices=("ember", "flamew"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, action="append", default=[])
    parser.add_argument("--max-fraction", default="0.05")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(mix(args.base, args.site, args.model, args.out,
                         args.heldout, args.max_fraction, args.seed), indent=2))


if __name__ == "__main__":
    main()
