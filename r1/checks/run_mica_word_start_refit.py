#!/usr/bin/env python3
"""Matched native MICA refits with different first-byte-of-word weights.

The exported integer cellular engine is evaluated at every byte and EOS. A
separate exact metric tracks the first letter/digit after BOS or a space.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from run_mica_page_block_refit import digest, sample_train, write_json


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
SOURCE = R1 / "runs/codex_page_block_refit_20260927_v2/control"
OUT = R1 / "runs/codex_word_start_refit_20260927"
WEIGHTS = (1, 2, 4)


def fresh_dev(path: Path, clean: set[bytes], train: set[bytes],
              skip: int = 512, n: int = 256) -> list[bytes]:
    rows, seen = [], set()
    with path.open(encoding="ascii") as src:
        for line in src:
            record = bytes.fromhex(line.strip())[:256]
            if not record or record in clean or record in train or record in seen:
                continue
            seen.add(record)
            if len(seen) <= skip:
                continue
            rows.append(record)
            if len(rows) == n:
                break
    if len(rows) != n:
        raise ValueError("not enough disjoint development records")
    return rows


def exact_metrics(model, records: list[bytes], batch, spec,
                  chunk: int = 32) -> dict:
    """Same integer probe/ingest/eligible-softmax path as batch.evaluate."""
    ms = batch.ModelStack([model], score_backend="c")
    eligible = np.r_[np.arange(256), spec.EOS]
    all_nats, all_counts = [], []
    word_nats, word_counts, word_top1 = [], [], []
    for start in range(0, len(records), chunk):
        part = records[start:start + chunk]
        n = len(part)
        lengths = np.asarray([len(r) for r in part], dtype=np.int32)
        maxlen = int(lengths.max())
        padded = np.full((n, maxlen), -1, dtype=np.int32)
        for i, row in enumerate(part):
            padded[i, :len(row)] = np.frombuffer(row, dtype=np.uint8)
        st = batch.BatchSession(n, np.zeros(n, dtype=np.int32))
        batch.ingest_batch(ms, st, np.full(n, spec.BOS, dtype=np.int32))
        rec_nats = np.zeros(n, dtype=np.float64)
        rec_counts = np.zeros(n, dtype=np.int64)
        rec_word_nats = np.zeros(n, dtype=np.float64)
        rec_word_counts = np.zeros(n, dtype=np.int64)
        rec_word_top1 = np.zeros(n, dtype=np.int64)
        for t in range(maxlen + 1):
            scores = batch.probe_batch(ms, st).astype(np.float64) / spec.LOGIT_DIVISOR
            target = np.where(t < lengths, padded[:, min(t, maxlen - 1)], spec.EOS)
            alive = t <= lengths
            elig = scores[:, eligible]
            mx = elig.max(axis=1)
            logsum = mx + np.log(np.exp(elig - mx[:, None]).sum(axis=1))
            nll = logsum - scores[np.arange(n), target]
            rec_nats += np.where(alive, nll, 0.0)
            rec_counts += alive
            if t < maxlen:
                letter = (((target >= 65) & (target <= 90)) |
                          ((target >= 97) & (target <= 122)) |
                          ((target >= 48) & (target <= 57)) |
                          ((target >= 128) & (target <= 255)))
                previous_space = (t == 0) if t == 0 else padded[:, t - 1] == 32
                word_start = (t < lengths) & previous_space & letter
                rec_word_nats += np.where(word_start, nll, 0.0)
                rec_word_counts += word_start
                rec_word_top1 += word_start & (eligible[elig.argmax(axis=1)] == target)
                feed = np.where(t < lengths, padded[:, t], spec.BOS)
                batch.ingest_batch(ms, st, feed.astype(np.int32))
        all_nats.extend(rec_nats.tolist())
        all_counts.extend(rec_counts.tolist())
        word_nats.extend(rec_word_nats.tolist())
        word_counts.extend(rec_word_counts.tolist())
        word_top1.extend(rec_word_top1.tolist())
    return {
        "records": len(records), "targets": int(sum(all_counts)),
        "bits": sum(all_nats) / sum(all_counts) / math.log(2),
        "nats": all_nats, "counts": all_counts,
        "word_start_targets": int(sum(word_counts)),
        "word_start_bits": sum(word_nats) / sum(word_counts) / math.log(2),
        "word_start_top1": sum(word_top1) / sum(word_counts),
        "word_start_nats": word_nats,
        "word_start_counts": word_counts,
        "word_start_correct": word_top1,
    }


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"refusing nonempty output directory: {OUT}")
    OUT.mkdir(parents=True)
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    for key in list(os.environ):
        if key.startswith("MICA_"):
            del os.environ[key]
    os.environ.update({k: str(v) for k, v in info["env"].items()})
    os.environ["PYTHONUTF8"] = "1"
    sys.path[:0] = [str(R1), str(R1 / "checks")]
    from mica_r1 import batch, fit, serialize, spec
    from mica_r1.discretise import to_integer
    from common import paired_bootstrap
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for matched refits")
    device = torch.device("cuda")
    source_sha = digest(SOURCE / "best.mica")
    if source_sha != "d6cedad89ba210efe4ee50c2b87d6989f9fff241b73c75718b401d14b6e127cc":
        raise ValueError(f"unexpected source SHA: {source_sha}")
    source = serialize.load(SOURCE / "best.mica")
    clean = [bytes.fromhex(s.strip())[:256] for s in
             (R1 / "data/eval_clean/val1000.jsonl").open(encoding="ascii")
             if s.strip()]
    clean_set = set(clean)
    train = sample_train(R1 / "data/everyday/train.jsonl", 40000, clean_set)
    dev = fresh_dev(R1 / "data/everyday/val.jsonl", clean_set, set(train))
    dev_path = OUT / "dev_fresh256.jsonl"
    dev_path.write_text("".join(row.hex() + "\n" for row in dev), encoding="ascii")
    write_json(OUT / "protocol.json", {
        "source": str(SOURCE / "best.mica"), "source_sha256": source_sha,
        "train": "r1/data/everyday/train.jsonl", "train_records": len(train),
        "dev": str(dev_path), "dev_sha256": digest(dev_path),
        "dev_rule": "256 unique disjoint val records after skipping first 512 eligible",
        "clean": "r1/data/eval_clean/val1000.jsonl",
        "weights": WEIGHTS, "steps_per_arm": 1500, "lr_scale": 0.01,
        "fit_record_limit": 20000,
        "word_start": "letter/digit/high-byte after BOS or ASCII space",
        "accept": "dev word-start gain >=0.02 bits with paired CI upper<0; total dev degradation <=0.015 bits",
        "engine": "exact exported integer cellular engine, C scorer",
    })

    def status(stage: str, detail: str = "") -> None:
        write_json(OUT / "status.json", {
            "stage": stage, "detail": detail, "updated": time.time()})

    try:
        batch.MAX_TICKS = spec.MAX_TICKS
        status("refitting")
        for weight in WEIGHTS:
            label = f"weight{weight}"
            status("refitting", label)
            torch.manual_seed(20260927)
            sm = fit.from_integer(source, spec.MAX_TICKS).to(device)
            if digest(SOURCE / "best.mica") != source_sha:
                raise RuntimeError("source changed during run")
            with (OUT / f"{label}_refit.log").open("w", encoding="utf-8") as log:
                fit.fit_readout_and_rules(
                    sm, train, device, spec.MAX_TICKS, steps=1500,
                    max_records=20000, check_every=250, lr_scale=0.01,
                    word_start_weight=float(weight),
                    log=lambda line: print(line, file=log, flush=True))
            folder = OUT / label
            folder.mkdir()
            serialize.save(to_integer(sm), folder / "best.mica")
            shutil.copy2(SOURCE / "run_info.json", folder / "run_info.json")
            del sm
            torch.cuda.empty_cache()

        status("development_acceptance")
        dev_metrics = {"source": exact_metrics(source, dev, batch, spec)}
        for weight in WEIGHTS:
            label = f"weight{weight}"
            model = serialize.load(OUT / label / "best.mica")
            dev_metrics[label] = exact_metrics(model, dev, batch, spec)
        baseline = dev_metrics["weight1"]
        proposals = {}
        for weight in (2, 4):
            label = f"weight{weight}"
            cand = dev_metrics[label]
            wdiff = paired_bootstrap(cand["word_start_nats"],
                                     baseline["word_start_nats"],
                                     baseline["word_start_counts"])
            adiff = paired_bootstrap(cand["nats"], baseline["nats"],
                                     baseline["counts"])
            proposals[label] = {
                "word_start_difference": wdiff,
                "all_target_difference": adiff,
                "passes_dev": bool(wdiff[0] <= -0.02 and wdiff[2] < 0 and
                                   adiff[0] <= 0.015),
            }
        qualified = [label for label, data in proposals.items()
                     if data["passes_dev"]]
        selected = min(qualified,
                       key=lambda label: proposals[label]["word_start_difference"][0]) \
                   if qualified else "weight1"
        result = {"source_sha256": source_sha,
                  "dev": {label: {"file": str(SOURCE / "best.mica") if label == "source"
                                  else str(OUT / label / "best.mica"),
                                  "sha256": source_sha if label == "source"
                                  else digest(OUT / label / "best.mica"),
                                  "bits": metrics["bits"],
                                  "word_start_bits": metrics["word_start_bits"],
                                  "word_start_top1": metrics["word_start_top1"],
                                  "targets": metrics["targets"],
                                  "word_start_targets": metrics["word_start_targets"]}
                          for label, metrics in dev_metrics.items()},
                  "proposals": proposals, "selected_on_dev": selected,
                  "clean_val1000": "evaluation pending"}
        write_json(OUT / "results.json", result)
        status("evaluating_clean")
        clean_metrics = {}
        for label in ("source", "weight1", "weight2", "weight4"):
            model = source if label == "source" else serialize.load(
                OUT / label / "best.mica")
            metrics = exact_metrics(model, clean, batch, spec)
            clean_metrics[label] = metrics
            write_json(OUT / f"{label}_val1000.json", metrics)
            status("evaluating_clean", label)
        previous = json.loads((R1 /
            "runs/codex_page_block_refit_20260927_v2/control_val1000.json")
            .read_text(encoding="utf-8"))["control"]
        if clean_metrics["source"]["counts"] != previous["counts"] or abs(
                clean_metrics["source"]["bits"] - previous["bits"]) > 1e-9:
            raise AssertionError("exact evaluator disagrees with prior eval_int")
        clean_summary = {}
        for label, metrics in clean_metrics.items():
            wdiff = paired_bootstrap(metrics["word_start_nats"],
                                     clean_metrics["source"]["word_start_nats"],
                                     clean_metrics["source"]["word_start_counts"])
            adiff = paired_bootstrap(metrics["nats"],
                                     clean_metrics["source"]["nats"],
                                     clean_metrics["source"]["counts"])
            clean_summary[label] = {
                "bits": metrics["bits"],
                "word_start_bits": metrics["word_start_bits"],
                "word_start_top1": metrics["word_start_top1"],
                "targets": metrics["targets"],
                "word_start_targets": metrics["word_start_targets"],
                "difference_vs_source": adiff,
                "word_start_difference_vs_source": wdiff,
            }
        result["clean_val1000"] = clean_summary
        write_json(OUT / "results.json", result)
        status("generating")
        for weight in WEIGHTS:
            label = f"weight{weight}"
            command = [sys.executable,
                       str(R1 / "checks/eval_mica_raw_generation.py"),
                       "--run", str(OUT / label), "--prompts",
                       str(R1 / "checks/generation_dev10.json"),
                       "--limit", "10", "--byte-limit", "100",
                       "--out", str(OUT / f"{label}_generation.json")]
            with (OUT / f"{label}_generation.log").open("w", encoding="utf-8") as stream:
                proc = subprocess.run(command, cwd=ROOT, env=dict(os.environ),
                                      stdout=stream, stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError(f"generation failed: {label} {proc.returncode}")
        status("complete")
        print(json.dumps({"selected": selected, "clean": clean_summary},
                         indent=2), flush=True)
        return 0
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
