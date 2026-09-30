#!/usr/bin/env python3
"""One bounded low-rate native Flame VSET/readout refit on mix-A training.

Prepare writes two fresh, disjoint development sets and a predeclared gate.
Training never reads development or clean evaluation records.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

from run_mica_ember_balanced import EVALS, key, records
from run_mica_page_block_refit import digest, write_json


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
MIX = R1 / "data/mix/v02a"
SOURCE = R1 / "runs/codex_flame_balanced_v02a_20260928/train"
SOURCE_SHA = "46be93f265743baacd391698aac9721a6bfba2b616180c00eaa02e421dac2e92"
OUT = R1 / "runs/codex_flame_refit_20260928"
DEV_SOURCES = {
    "chat": R1 / "data/chat/val.jsonl",
    "everyday": R1 / "data/everyday/val.jsonl",
}
GATE = (
    "candidate minus source exact integer equal-weight chat/everyday fresh "
    "dev1000 mean <= -0.005 bits/target with paired 95% CI upper < 0; "
    "neither individual-domain point difference > +0.002; "
    "clean chat/everyday val1000 report only after acceptance"
)


def status(stage: str, detail: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "status.json", {
        "stage": stage, "detail": detail, "updated": time.time()})


class TrainingBloom:
    """No-false-negative filter of normalized training records (8 MiB)."""

    MASK = (1 << 26) - 1

    def __init__(self) -> None:
        self.bits = bytearray(1 << 23)

    def indexes(self, fingerprint: bytes):
        first = int.from_bytes(fingerprint[:8], "little")
        second = int.from_bytes(fingerprint[8:], "little") | 1
        for offset in range(4):
            yield (first + offset * second) & self.MASK

    def add(self, fingerprint: bytes) -> None:
        for index in self.indexes(fingerprint):
            self.bits[index >> 3] |= 1 << (index & 7)

    def contains(self, fingerprint: bytes) -> bool:
        return all(self.bits[index >> 3] & (1 << (index & 7))
                   for index in self.indexes(fingerprint))


def reservoir(path: Path, excluded: set[bytes], bloom: TrainingBloom, seed: int,
              size: int = 1000) -> list[bytes]:
    rng = random.Random(seed)
    rows: list[bytes] = []
    seen: set[bytes] = set()
    eligible = 0
    for row in records(path):
        fingerprint = key(row)
        if (fingerprint in excluded or fingerprint in seen or
                bloom.contains(fingerprint)):
            continue
        seen.add(fingerprint)
        eligible += 1
        if len(rows) < size:
            rows.append(row)
        else:
            index = rng.randrange(eligible)
            if index < size:
                rows[index] = row
    if len(rows) != size:
        raise RuntimeError(f"only {len(rows)} eligible records in {path}")
    return rows


def prepare() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing populated output directory: {OUT}")
    status("preparing")
    if digest(SOURCE / "best.mica") != SOURCE_SHA or not (SOURCE / "FINISHED").is_file():
        raise RuntimeError("mature native Flame source changed or unfinished")
    old = json.loads((R1 / "runs/codex_ember_balanced_v02a_20260928/preflight.json")
                     .read_text(encoding="utf-8"))
    if (digest(MIX / "manifest.json") != old["mix_manifest_sha256"] or
            digest(MIX / "train.jsonl") != old["mix_file_sha256"]["mix/v02a/train.jsonl"] or
            digest(MIX / "val.jsonl") != old["mix_file_sha256"]["mix/v02a/val.jsonl"]):
        raise RuntimeError("verified balanced mix A changed")
    excluded: set[bytes] = set()
    reserved = [MIX / "val.jsonl", *EVALS.values(),
                R1 / "data/ctxprobe/chat_long600.jsonl",
                R1 / "data/ctxprobe/everyday_long600.jsonl"]
    for path in reserved:
        excluded.update(key(row) for row in records(path))
    # Some common dialogue phrases occur in both source splits. A Bloom
    # filter excludes them before sampling, with no false negatives.
    bloom = TrainingBloom()
    training_count = 0
    for row in records(MIX / "train.jsonl"):
        bloom.add(key(row))
        training_count += 1
    status("preparing", f"indexed {training_count} normalized train records")
    fresh: dict[str, list[bytes]] = {}
    for index, (domain, path) in enumerate(DEV_SOURCES.items()):
        rows = reservoir(path, excluded, bloom, 20260928 + index)
        fresh[domain] = rows
        excluded.update(key(row) for row in rows)
    chosen = {key(row) for rows in fresh.values() for row in rows}
    if len(chosen) != 2000:
        raise RuntimeError("fresh development domains overlap after normalisation")
    # Exact audit still checks the final chosen records against all training.
    hits: set[bytes] = set()
    for row in records(MIX / "train.jsonl"):
        fingerprint = key(row)
        if fingerprint in chosen:
            hits.add(fingerprint)
    if hits:
        raise RuntimeError(f"Bloom filter missed {len(hits)} train overlaps")
    for domain, rows in fresh.items():
        (OUT / f"{domain}_dev2_1000.jsonl").write_text(
            "".join(row.hex() + "\n" for row in rows), encoding="ascii")
    manifest = json.loads((MIX / "manifest.json").read_text(encoding="utf-8"))
    if training_count != manifest["splits"]["train"]["total"]["records"]:
        raise RuntimeError("training mixture record count changed")
    write_json(OUT / "protocol.json", {
        "source": str(SOURCE / "best.mica"), "source_sha256": SOURCE_SHA,
        "mix_manifest_sha256": digest(MIX / "manifest.json"),
        "mix_train_sha256": digest(MIX / "train.jsonl"),
        "fresh_dev": {domain: {
            "path": str(OUT / f"{domain}_dev2_1000.jsonl"),
            "sha256": digest(OUT / f"{domain}_dev2_1000.jsonl"),
            "records": len(rows)} for domain, rows in fresh.items()},
        "fresh_dev_sources_sha256": {domain: digest(path)
                                     for domain, path in DEV_SOURCES.items()},
        "excluded": [str(path) for path in reserved],
        "training_records_verified": training_count,
        "training_bloom_bits": 1 << 26,
        "exact_training_overlap_count": len(hits),
        "train_sample_records": 40000, "train_sample_seed": 20260930,
        "fit_seed": 20260930, "steps": 1500, "max_records": 20000,
        "check_every": 250, "lr_scale": 0.01,
        "selection": GATE,
        "architecture": "native learned integer-rule cellular automaton; selectors fixed; VSET immediates and integer readout refitted",
    })
    status("prepared")


def sample_training(size: int, seed: int) -> list[bytes]:
    rng = random.Random(seed)
    rows: list[bytes] = []
    count = 0
    for row in records(MIX / "train.jsonl"):
        count += 1
        if len(rows) < size:
            rows.append(row)
        else:
            index = rng.randrange(count)
            if index < size:
                rows[index] = row
    if len(rows) != size:
        raise RuntimeError("not enough mix-A training records")
    return rows


def train() -> None:
    protocol_path = OUT / "protocol.json"
    if not protocol_path.is_file():
        raise RuntimeError("prepare must pass before training")
    if json.loads((OUT / "status.json").read_text(encoding="utf-8"))["stage"] != "prepared":
        raise RuntimeError("refusing duplicate or unfinished refit")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (digest(SOURCE / "best.mica") != SOURCE_SHA or
            digest(MIX / "manifest.json") != protocol["mix_manifest_sha256"] or
            digest(MIX / "train.jsonl") != protocol["mix_train_sha256"]):
        raise RuntimeError("source or approved training mixture changed")
    for item in protocol["fresh_dev"].values():
        if digest(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("fresh development set changed")
    if (ROOT.parent / "STOP_MICA").exists():
        raise RuntimeError("STOP_MICA blocks training")
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    for name in list(os.environ):
        if name.startswith("MICA_"):
            del os.environ[name]
    os.environ.update({name: str(value) for name, value in info["env"].items()})
    os.environ["PYTHONUTF8"] = "1"
    sys.path.insert(0, str(R1))
    from mica_r1 import fit, serialize, spec
    from mica_r1.discretise import to_integer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for native Flame refit")
    if spec.N_CANDIDATES != 1024 or spec.N_PHASE != 16 or spec.VSET_WIDTH != 6:
        raise RuntimeError("unexpected native F1 Flame geometry")
    status("sampling_training")
    rows = sample_training(40000, 20260930)
    status("refitting")
    torch.manual_seed(20260930)
    model = fit.from_integer(serialize.load(SOURCE / "best.mica"),
                             spec.MAX_TICKS).to(torch.device("cuda"))
    serialize.save(to_integer(model), OUT / "initial.mica")
    if digest(OUT / "initial.mica") != SOURCE_SHA:
        raise RuntimeError("soft conversion changed source integer model")
    with (OUT / "refit.log").open("w", encoding="utf-8") as log:
        fit.fit_readout_and_rules(
            model, rows, torch.device("cuda"), spec.MAX_TICKS,
            steps=1500, max_records=20000, check_every=250, lr_scale=0.01,
            log=lambda line: print(line, file=log, flush=True))
    candidate = OUT / "candidate"
    candidate.mkdir()
    serialize.save(to_integer(model), candidate / "best.mica")
    shutil.copy2(SOURCE / "run_info.json", candidate / "run_info.json")
    status("training_complete", digest(candidate / "best.mica"))


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train"), required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out.resolve()
    try:
        {"prepare": prepare, "train": train}[args.stage]()
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
