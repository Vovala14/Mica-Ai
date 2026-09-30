#!/usr/bin/env python3
"""Build frozen, human-readable prompts from the clean everyday eval splits.

Development prompts come only from validation records. Holdout B prompts come
only from test records and are reserved for one final model/configuration run.
The record IDs below were chosen for varied everyday prose and scene language
before viewing any decoder output for these sets. No random sampling is used.

Run from any directory::

    python r1/checks/make_sentence_quality_sets.py

An existing output directory is verified byte for byte instead of overwritten.
The pre-existing sentence_holdout_20260926.json is deliberately not read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


R1 = Path(__file__).resolve().parents[1]
SOURCE = R1 / "data/eval_clean"
OUT = R1 / "data/sentence_quality_20260926"
TRAIN = R1 / "data/everyday/train.jsonl"

# Indices in eval_clean/val1000.jsonl, divided by editorial prompt style.
# Both lists are sorted together before prefix lengths are assigned.
DEV_GENERAL = (
    6, 57, 65, 77, 99, 127, 170, 188, 200, 211,
    220, 225, 235, 330, 332, 344, 529, 543, 546, 600,
    618, 621, 642, 706, 714, 731, 739, 831, 930, 936,
)
DEV_SCENE = (
    5, 9, 17, 37, 58, 85, 87, 91, 104, 122,
    138, 174, 222, 240, 316, 338, 405, 408, 412, 628,
)
HOLDOUT_GENERAL = (
    30, 68, 70, 87, 97, 106, 173, 232, 248,
    307, 330, 420, 448, 539, 603, 649, 731, 842,
)
HOLDOUT_SCENE = (
    16, 27, 66, 84, 148, 205, 271, 338, 446,
    619, 740, 841,
)
EXPECTED_CLEAN_SHA256 = {
    "val": "c6107c8e80f28156c0314db180576e7311ff3f7a9e2843d711709c9a7d2f4d26",
    "test": "d11e7f8883c5c87280735ae499610532787cea10a82bcb5dadab16829a471c17",
}
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]*[A-Za-z])")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(text: str) -> str:
    """Use the same key semantics as make_clean_eval.py."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def source_records(split: str, source: Path) -> tuple[list[bytes], str]:
    path = source / f"{split}1000.jsonl"
    payload = path.read_bytes()
    file_hash = sha256(payload)
    if file_hash != EXPECTED_CLEAN_SHA256[split]:
        raise ValueError(f"unexpected clean evaluation source hash: {path}")
    lines = payload.splitlines()
    if len(lines) != 1000 or not payload.endswith(b"\n"):
        raise ValueError(f"unexpected clean evaluation source shape: {path}")
    return [bytes.fromhex(line.decode("ascii")) for line in lines], file_hash


def selected_rows(split: str, source: list[bytes], general: tuple[int, ...],
                  scene: tuple[int, ...], label: str) -> list[dict]:
    tags = {index: "general" for index in general}
    if len(tags) != len(general) or len(set(scene)) != len(scene) or set(scene) & tags.keys():
        raise ValueError(f"duplicate curated {label} indices")
    tags.update({index: "scene" for index in scene})
    rows = []
    for ordinal, index in enumerate(sorted(tags)):
        raw = source[index]
        reference = raw.decode("utf-8", errors="strict")
        words = list(re.finditer(r"\S+", reference))
        if not 6 <= len(words) <= 40 or not reference.endswith((".", "?", "!")):
            raise ValueError(f"unsuitable source record: {split}:{index}")
        # Cycle through short and long prompts. Keep two words after the cut.
        requested = 2 + ordinal % 7
        initial_words = min(requested, len(words) - 2)
        options = []
        for count in range(initial_words, len(words) - 1):
            # The source has a space after each eligible cut. Preserve it so
            # prompt + reference_suffix reconstructs the exact source text.
            cut = words[count - 1].end() + 1
            prompt = reference[:cut]
            suffix = reference[cut:]
            if prompt + suffix != reference or not prompt.endswith(" "):
                raise ValueError(f"invalid word cut: {split}:{index}")
            options.append((count, prompt, suffix))
        rows.append({
            "id": f"{label}-{ordinal + 1:02d}",
            "style": tags[index],
            "source_split": split,
            "clean_eval_index": index,
            "source_record_sha256": sha256(raw),
            "reference": reference,
            "_options": options,
        })
    return rows


def training_collisions(train: Path, rows: list[dict]) -> tuple[set[str], str]:
    reference_keys = {normalized(row["reference"]) for row in rows}
    keys = set(reference_keys)
    keys.update(normalized(prompt) for row in rows
                for _, prompt, _ in row["_options"])
    hits: set[str] = set()
    digest = hashlib.sha256()
    with train.open("rb") as stream:
        for line in stream:
            digest.update(line)
            raw = bytes.fromhex(line.strip().decode("ascii"))[:256]
            text = raw.decode("utf-8", errors="replace")
            key = normalized(text)
            if key in keys:
                hits.add(key)
            # Everyday prose records can contain several sentences. An exact
            # reference hidden inside one must be treated as training overlap.
            if any(mark in text for mark in (". ", "? ", "! ")):
                for sentence in SENTENCE_BOUNDARY.split(text):
                    sentence_key = normalized(sentence)
                    if sentence_key in reference_keys:
                        hits.add(sentence_key)
    return hits, digest.hexdigest()


def assign_prompts(rows: list[dict], collisions: set[str]) -> dict:
    references: set[str] = set()
    prompts: set[str] = set()
    extensions = 0
    for row in rows:
        ref_key = normalized(row["reference"])
        if ref_key in collisions or ref_key in references:
            raise ValueError(f"training or selected reference duplicate: {row['id']}")
        references.add(ref_key)
        first_count = row["_options"][0][0]
        for count, prompt, suffix in row.pop("_options"):
            key = normalized(prompt)
            if key not in collisions and key not in prompts:
                prompts.add(key)
                row["prompt"] = prompt
                row["reference_suffix"] = suffix
                row["prompt_words"] = count
                extensions += count - first_count
                break
        else:
            raise ValueError(f"no unique, train-free prompt: {row['id']}")
    return {"selected_reference_count": len(references),
            "selected_prompt_count": len(prompts),
            "prefix_words_added_for_uniqueness": extensions,
            "train_reference_duplicates": 0,
            "train_prompt_duplicates": 0}


def stable_json(data: object) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def jsonl(rows: list[dict]) -> bytes:
    return b"".join(json.dumps(row, ensure_ascii=False,
                               separators=(",", ":")).encode("utf-8") + b"\n"
                    for row in rows)


def build(source: Path = SOURCE, train: Path = TRAIN, out: Path = OUT) -> dict:
    val, val_hash = source_records("val", source)
    test, test_hash = source_records("test", source)
    dev = selected_rows("val", val, DEV_GENERAL, DEV_SCENE, "dev")
    holdout = selected_rows("test", test, HOLDOUT_GENERAL, HOLDOUT_SCENE,
                            "holdout-b")
    rows = dev + holdout
    collisions, train_hash = training_collisions(train, rows)
    checks = assign_prompts(rows, collisions)
    assert len(dev) == 50 and len(holdout) == 30
    assert len({row["source_record_sha256"] for row in rows}) == 80
    assert all(row["prompt"] + row["reference_suffix"] == row["reference"]
               for row in rows)

    dev_payload = jsonl(dev)
    holdout_payload = jsonl(holdout)
    source_manifest = source / "manifest.json"
    manifest = {
        "purpose": "Qualitative English continuation prompts for next MICA decoder iteration",
        "holdout_b_policy": "Freeze model, configuration, and evaluation procedure before one final run on holdout B; do not use its outputs for development.",
        "construction": "Manual selection of clean everyday evaluation record indices, then deterministic 2-8 word prefix cycle in source order; extend a prefix only to avoid a duplicate or training match. No RNG or seed.",
        "source": {
            "clean_eval_manifest_sha256": sha256(source_manifest.read_bytes()),
            "val1000_sha256": val_hash,
            "test1000_sha256": test_hash,
            "everyday_train_jsonl_sha256": train_hash,
            "normalization": "UTF-8 replacement, casefold, whitespace collapse, strip",
        },
        "outputs": {
            "dev50.jsonl": {"records": len(dev), "sha256": sha256(dev_payload),
                            "general": len(DEV_GENERAL), "scene": len(DEV_SCENE)},
            "holdout_b30.jsonl": {"records": len(holdout),
                                   "sha256": sha256(holdout_payload),
                                   "general": len(HOLDOUT_GENERAL),
                                   "scene": len(HOLDOUT_SCENE)},
        },
        "checks": checks,
    }
    output_files = {"dev50.jsonl": dev_payload,
                    "holdout_b30.jsonl": holdout_payload,
                    "manifest.json": stable_json(manifest)}
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in output_files.items():
        path = out / name
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"existing output differs; refusing overwrite: {path}")
        else:
            path.write_bytes(payload)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    report = build(args.source, args.train, args.out)
    print(stable_json(report).decode("utf-8"))


if __name__ == "__main__":
    main()
