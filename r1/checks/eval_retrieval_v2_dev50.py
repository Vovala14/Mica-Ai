#!/usr/bin/env python3
"""Reproduce the opt-in v2 comparison on sentence_quality_20260926/dev50.

This intentionally uses one named development set and verifies that the
checkpoint, retrieval index, and prompts match the existing baseline run.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time


R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
RUN = R1 / "runs/sweep/everyday_c256"
INDEX = R1 / "data/everyday/attributed_tatoeba_index_1m_detailed"
PROMPTS = R1 / "data/sentence_quality_20260926/dev50.jsonl"
BASELINE = R1 / "runs/checks/retrieval_dev50_attributed_baseline_sentence.json"
OUTPUT = R1 / "runs/checks/retrieval_dev50_attributed_v2_sentence.json"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    expected = baseline["config"]
    actual = {
        "model_sha256": file_sha(RUN / "best.mica"),
        "index_sha256": file_sha(INDEX / "sentences.jsonl"),
        "prompts_sha256": file_sha(PROMPTS),
    }
    for key, value in actual.items():
        if expected[key] != value:
            raise ValueError(f"{key} differs from the dev50 baseline")
    if expected["mode"] != "sentence" or expected["max_words"] != 8:
        raise ValueError("unexpected baseline decoder settings")
    prompts = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines()]
    if len(prompts) != 50 or len(baseline["rows"]) != 50:
        raise ValueError("dev50 or baseline row count differs from 50")
    for prompt, row in zip(prompts, baseline["rows"]):
        if prompt["id"] != row["id"] or prompt["prompt"] != row["prompt"]:
            raise ValueError("baseline and dev50 prompt order differ")

    run_info = json.loads((RUN / "run_info.json").read_text(encoding="utf-8"))
    os.environ.update({key: str(value) for key, value in run_info["env"].items()})
    from mica_r1 import serialize
    from mica_r1.retrieval_suggest import SentenceIndex
    from mica_r1.retrieval_v2 import suggest_retrieval_v2
    from mica_r1.suggest import MicaScorer

    index = SentenceIndex.load(INDEX)
    scorer = MicaScorer(serialize.load(RUN / "best.mica"))
    rows = []
    for item in prompts:
        start = time.perf_counter()
        result = suggest_retrieval_v2(scorer, index, item["prompt"],
                                      mode="sentence", max_words=8)
        rows.append({
            "id": item["id"], "prompt": item["prompt"],
            "abstained": result.abstained, "reason": result.reason,
            "seconds": time.perf_counter() - start,
            "suggestions": [suggestion.__dict__ for suggestion in result.suggestions],
        })
    config = {**actual,
              "decoder_sha256": file_sha(R1 / "mica_r1/retrieval_v2.py"),
              "compat_sha256": file_sha(R1 / "mica_r1/retrieval_compat.py"),
              "mode": "sentence", "max_words": 8, "limit": 50}
    payload = {
        "config": config,
        "summary": {
            "prompts": len(rows),
            "usable": sum(not row["abstained"] for row in rows),
            "abstained": sum(row["abstained"] for row in rows),
            "total_seconds": sum(row["seconds"] for row in rows),
        },
        "rows": rows,
    }
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, OUTPUT)
    changed = sum(
        (row["suggestions"][0]["continuation"] if row["suggestions"] else None) !=
        (base["suggestions"][0]["continuation"] if base["suggestions"] else None)
        for row, base in zip(rows, baseline["rows"])
    )
    print(json.dumps({"baseline": baseline["summary"], "v2": payload["summary"],
                      "changed_top1": changed, "output": str(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
