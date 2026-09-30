#!/usr/bin/env python3
"""Development-only end-to-end retrieval length sweep on dev50."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
RUN = R1 / "runs/sweep/everyday_c256"
INDEX = R1 / "data/everyday/attributed_tatoeba_index_1m_detailed"
PROMPTS = R1 / "data/sentence_quality_20260926/dev50.jsonl"
BASELINE = R1 / "runs/checks/retrieval_dev50_attributed_baseline_sentence.json"
OUTPUT = R1 / "runs/checks/retrieval_dev50_mica_length_full_shortlist.json"
LENGTHS = (8, 24, 32, 1024)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    expected = baseline["config"]
    paths = {"model_sha256": RUN / "best.mica",
             "index_sha256": INDEX / "sentences.jsonl",
             "prompts_sha256": PROMPTS,
             "decoder_sha256": R1 / "mica_r1/retrieval_suggest.py"}
    for key, path in paths.items():
        if expected[key] != sha256(path):
            raise ValueError(f"{key} differs from the dev50 baseline")
    prompts = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines()]
    if len(prompts) != len(baseline["rows"]):
        raise ValueError("dev50 and baseline row counts differ")
    for prompt, row in zip(prompts, baseline["rows"]):
        if (prompt["id"], prompt["prompt"]) != (row["id"], row["prompt"]):
            raise ValueError("dev50 prompt order differs")

    info = json.loads((RUN / "run_info.json").read_text(encoding="utf-8"))
    os.environ.update({key: str(value) for key, value in info["env"].items()})
    from mica_r1 import serialize
    from mica_r1.retrieval_suggest import SentenceIndex, suggest_retrieval
    from mica_r1.suggest import MicaScorer

    index = SentenceIndex.load(INDEX)
    scorer = MicaScorer(serialize.load(RUN / "best.mica"))
    rows = [{"id": item["id"], "prompt": item["prompt"], "variants": {}}
            for item in prompts]
    summary = {}
    for length in LENGTHS:
        key = "full" if length == 1024 else str(length)
        for row in rows:
            start = time.perf_counter()
            result = suggest_retrieval(scorer, index, row["prompt"],
                                       mode="sentence", max_words=8,
                                       model_scored_bytes=length)
            elapsed = time.perf_counter() - start
            row["variants"][key] = {
                "seconds": elapsed, "abstained": result.abstained,
                "reason": result.reason,
                "suggestions": [suggestion.__dict__ for suggestion in result.suggestions],
            }
        if length == 8:
            for row, old in zip(rows, baseline["rows"]):
                fresh = row["variants"]["8"]
                if (fresh["reason"] != old["reason"] or
                        [s["continuation"] for s in fresh["suggestions"]] !=
                        [s["continuation"] for s in old["suggestions"]]):
                    raise ValueError(f"8-byte output changed for {row['id']}")
        seconds = [row["variants"][key]["seconds"] for row in rows]
        summary[key] = {
            "usable": sum(not row["variants"][key]["abstained"] for row in rows),
            "total_seconds": sum(seconds), "mean_seconds": statistics.mean(seconds),
            "median_seconds": statistics.median(seconds),
            "changed_top1_vs_8": sum(
                (row["variants"][key]["suggestions"][0]["continuation"]
                 if row["variants"][key]["suggestions"] else None) !=
                (row["variants"]["8"]["suggestions"][0]["continuation"]
                 if row["variants"]["8"]["suggestions"] else None)
                for row in rows),
        }
        payload = {"config": {**expected, "lengths": [8, 24, 32, "full"],
                              "full_cap_bytes": 1024,
                              "candidate_scope": "full standard 48-match shortlist"},
                   "summary": summary, "rows": rows}
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        print(f"{key}: {summary[key]}", flush=True)
    print(f"output: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
