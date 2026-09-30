#!/usr/bin/env python3
"""Development-only probe of MICA continuation scoring length.

Re-score the three candidates already returned by the frozen baseline. This
keeps retrieval, filtering, semantic terms, and candidate provenance fixed.
Only the model's mean byte log likelihood is replaced in the rank score.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
RUN = R1 / "runs/sweep/everyday_c256"
BASELINE = R1 / "runs/checks/retrieval_dev50_attributed_baseline_sentence.json"
RATINGS = R1 / "runs/checks/retrieval_dev50_attributed_baseline_sentence_ratings.json"
PROMPTS = R1 / "data/sentence_quality_20260926/dev50.jsonl"
OUTPUT = R1 / "runs/checks/retrieval_dev50_mica_length_fixed_top3.json"
LENGTHS = (8, 24, 32, "full")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    ratings = json.loads(RATINGS.read_text(encoding="utf-8"))
    prompts = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines()]
    expected = baseline["config"]
    if expected["model_sha256"] != sha256(RUN / "best.mica"):
        raise ValueError("checkpoint differs from the dev50 baseline")
    if expected["prompts_sha256"] != sha256(PROMPTS):
        raise ValueError("prompts differ from the dev50 baseline")
    if len(prompts) != len(baseline["rows"]) != len(ratings["ratings"]):
        raise ValueError("dev50 and baseline row counts differ")
    for prompt, row, rating in zip(prompts, baseline["rows"], ratings["ratings"]):
        if ((prompt["id"], prompt["prompt"]) != (row["id"], row["prompt"])
                or (row["id"], row["prompt"]) != (rating["id"], rating["prompt"])):
            raise ValueError("dev50 prompt order differs")

    info = json.loads((RUN / "run_info.json").read_text(encoding="utf-8"))
    os.environ.update({key: str(value) for key, value in info["env"].items()})
    from mica_r1 import serialize
    from mica_r1.retrieval_suggest import _mean_logp
    from mica_r1.suggest import MicaScorer

    scorer = MicaScorer(serialize.load(RUN / "best.mica"))
    rows: list[dict] = []
    aggregate_seconds = {str(length): 0.0 for length in LENGTHS}
    for row, rating in zip(baseline["rows"], ratings["ratings"]):
        candidates = row["suggestions"]
        initial = scorer.start(row["prompt"].encode("utf-8")) if candidates else None
        variants: dict[str, dict] = {}
        for length in LENGTHS:
            key = str(length)
            scores = []
            start = time.perf_counter()
            for i, candidate in enumerate(candidates):
                max_bytes = (len(candidate["continuation"].encode("utf-8"))
                             if length == "full" else length)
                mean, nbytes = _mean_logp(scorer, initial, candidate["continuation"],
                                          max_bytes)
                # All non-model terms in the frozen baseline rank score remain fixed.
                score = candidate["rank_score"] - candidate["model_mean_logp"] + mean
                scores.append({"candidate_index": i, "mean_logp": mean,
                               "scored_bytes": nbytes, "rank_score": score})
                if length == 8:
                    if not math.isclose(mean, candidate["model_mean_logp"], abs_tol=1e-10):
                        raise ValueError(f"8-byte score changed for {row['id']} #{i}")
            elapsed = time.perf_counter() - start
            aggregate_seconds[key] += elapsed
            scores.sort(key=lambda x: (-x["rank_score"],
                                        -candidates[x["candidate_index"]]["matched_words"],
                                        candidates[x["candidate_index"]]["source_train_line"],
                                        candidates[x["candidate_index"]]["continuation"]))
            variants[key] = {"seconds": elapsed, "ranking": scores,
                             "top_candidate_index": (scores[0]["candidate_index"]
                                                     if scores else None)}
        if variants["8"]["top_candidate_index"] not in (None, 0):
            raise ValueError(f"8-byte rank changed for {row['id']}")
        rows.append({"id": row["id"], "prompt": row["prompt"],
                     "baseline_accepted": rating["accepted"],
                     "candidates": [{"continuation": c["continuation"],
                                     "full_text": c["full_text"],
                                     "source_url": c["source_url"],
                                     "baseline_rank_score": c["rank_score"],
                                     "baseline_mean_logp": c["model_mean_logp"],
                                     "full_bytes": len(c["continuation"].encode("utf-8"))}
                                    for c in candidates],
                     "variants": variants})

    summary = {}
    for length in LENGTHS:
        key = str(length)
        seconds = [row["variants"][key]["seconds"] for row in rows]
        changed = [row for row in rows if row["variants"][key]["top_candidate_index"]
                   not in (None, 0)]
        summary[key] = {"changed_top1": len(changed),
                        "changed_ids": [row["id"] for row in changed],
                        "score_seconds_total": aggregate_seconds[key],
                        "score_seconds_median_per_prompt": statistics.median(seconds),
                        "score_seconds_mean_per_prompt": statistics.mean(seconds)}
    result = {"config": {"model_sha256": expected["model_sha256"],
                          "index_sha256": expected["index_sha256"],
                          "prompts_sha256": expected["prompts_sha256"],
                          "baseline_sha256": sha256(BASELINE),
                          "ratings_sha256": sha256(RATINGS),
                          "candidate_scope": "frozen baseline top three",
                          "score": "mean UTF-8 byte log likelihood",
                          "lengths": list(LENGTHS)},
              "summary": summary, "rows": rows}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
