#!/usr/bin/env python3
"""Train one phase's hard integer rule thresholds on next-byte information."""
from __future__ import annotations

import copy
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
from run_mica_word_start_refit import exact_metrics, fresh_dev


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
SOURCE = R1 / "runs/codex_page_block_refit_20260927_v2/control"
OUT = R1 / "runs/codex_phase_bias_info_20260927"
PHASE = 7
PROPOSAL_RECORDS = 3000
SCREEN_RECORDS = 1000
DELTAS = (8, 16, 32, 64)
UPDATES_PER_PAGE = 2


def conditional_bits(winners: np.ndarray, targets: np.ndarray,
                     base: np.ndarray, candidates: int) -> tuple[float, np.ndarray]:
    """Smoothed in-sample conditional entropy and rule/byte counts."""
    vocab = len(base)
    hist = np.bincount(winners.astype(np.int64) * vocab + targets,
                       minlength=candidates * vocab).reshape(candidates, vocab)
    posterior = (hist + 16 * base[None, :]) / (hist.sum(1)[:, None] + 16)
    return float(-np.log(posterior[winners, targets]).mean() / math.log(2)), hist


def screen_bits(winners_train: np.ndarray, targets_train: np.ndarray,
                winners_screen: np.ndarray, targets_screen: np.ndarray,
                base: np.ndarray, candidates: int) -> float:
    _, hist = conditional_bits(winners_train, targets_train, base, candidates)
    posterior = (hist + 16 * base[None, :]) / (hist.sum(1)[:, None] + 16)
    return float(-np.log(posterior[winners_screen, targets_screen]).mean() /
                 math.log(2))


def optimize_page(raw_train: np.ndarray, targets_train: np.ndarray,
                  raw_screen: np.ndarray, targets_screen: np.ndarray,
                  original_bias: np.ndarray) -> tuple[np.ndarray, dict]:
    """Two coordinate steps, each choosing the best train-information split."""
    candidates = raw_train.shape[1]
    bias = original_bias.astype(np.int32).copy()
    base = (np.bincount(targets_train, minlength=258).astype(np.float64) + 0.5)
    base /= base.sum()
    scores = raw_train.astype(np.int32) + bias[None, :]
    winners = scores.argmax(axis=1)
    screen_scores = raw_screen.astype(np.int32) + bias[None, :]
    screen_winners = screen_scores.argmax(axis=1)
    before_train, _ = conditional_bits(winners, targets_train, base, candidates)
    before_screen = screen_bits(winners, targets_train, screen_winners,
                                targets_screen, base, candidates)
    changes = []
    for _ in range(UPDATES_PER_PAGE):
        best_score = scores[np.arange(len(winners)), winners]
        near = (scores >= best_score[:, None] - 64).sum(axis=0)
        pool = np.argsort(-near, kind="stable")[:48]
        current_bits, _ = conditional_bits(winners, targets_train,
                                            base, candidates)
        best = None
        for candidate in pool:
            candidate = int(candidate)
            for delta in DELTAS:
                new_score = scores[:, candidate] + delta
                stolen = ((new_score > best_score) |
                          ((new_score == best_score) & (candidate < winners)))
                stolen &= winners != candidate
                moved = int(stolen.sum())
                if moved < 8 or moved > len(winners) // 4:
                    continue
                new_winners = winners.copy()
                new_winners[stolen] = candidate
                proposal_bits, _ = conditional_bits(
                    new_winners, targets_train, base, candidates)
                if proposal_bits < current_bits - 1e-5 and (
                        best is None or proposal_bits < best[0]):
                    best = (proposal_bits, candidate, delta, moved)
        if best is None:
            break
        _, candidate, delta, moved = best
        bias[candidate] = min(32767, bias[candidate] + delta)
        scores[:, candidate] = raw_train[:, candidate] + bias[candidate]
        winners = scores.argmax(axis=1)
        screen_scores[:, candidate] = raw_screen[:, candidate] + bias[candidate]
        screen_winners = screen_scores.argmax(axis=1)
        changes.append({"candidate": candidate, "delta": delta,
                        "moved_training_rows": moved})
    after_train, _ = conditional_bits(winners, targets_train, base, candidates)
    after_screen = screen_bits(winners, targets_train, screen_winners,
                               targets_screen, base, candidates)
    return bias, {"train_bits_before": before_train,
                  "train_bits_after": after_train,
                  "screen_bits_before": before_screen,
                  "screen_bits_after": after_screen,
                  "changed": changes, "train_rows": len(targets_train),
                  "screen_rows": len(targets_screen)}


def command(label: str, argv: list[str], env: dict[str, str]) -> None:
    with (OUT / f"{label}.log").open("w", encoding="utf-8") as stream:
        rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{label} exited {rc}; inspect its log")


def status(stage: str, detail: str = "") -> None:
    write_json(OUT / "status.json", {"stage": stage, "detail": detail,
                                     "updated": time.time()})


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"refusing nonempty output directory: {OUT}")
    if (ROOT.parent / "STOP_MICA").exists():
        raise SystemExit("STOP_MICA is present")
    OUT.mkdir(parents=True)
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    for key in list(os.environ):
        if key.startswith("MICA_"):
            del os.environ[key]
    os.environ.update({k: str(v) for k, v in info["env"].items()})
    os.environ["PYTHONUTF8"] = "1"
    env = dict(os.environ)
    sys.path[:0] = [str(R1), str(R1 / "checks")]
    from mica_r1 import batch, fit, serialize, spec
    from mica_r1.discretise import to_integer
    from common import paired_bootstrap
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for the matched refit")
    if spec.WINDOW != 1 or spec.VSET_WIDTH != 6:
        raise RuntimeError("unexpected MICA geometry")
    device = torch.device("cuda")
    source_sha = digest(SOURCE / "best.mica")
    if source_sha != "d6cedad89ba210efe4ee50c2b87d6989f9fff241b73c75718b401d14b6e127cc":
        raise RuntimeError("source checkpoint changed")
    original = serialize.load(SOURCE / "best.mica")
    clean = [bytes.fromhex(line.strip())[:256] for line in
             (R1 / "data/eval_clean/val1000.jsonl").open(encoding="ascii")
             if line.strip()]
    train = sample_train(R1 / "data/everyday/train.jsonl", 40000, set(clean))
    dev = fresh_dev(R1 / "data/everyday/val.jsonl", set(clean),
                    set(train), skip=8192, n=1000)
    dev_file = OUT / "dev_fresh1000.jsonl"
    dev_file.write_text("".join(r.hex() + "\n" for r in dev), encoding="ascii")
    write_json(OUT / "protocol.json", {
        "source": str(SOURCE / "best.mica"), "source_sha256": source_sha,
        "phase": PHASE, "proposal_training_records": PROPOSAL_RECORDS,
        "proposal_screen_training_records": SCREEN_RECORDS,
        "rule_change": "two positive integer scoring-bias coordinate updates per page, selected on training conditional next-byte entropy",
        "train": "r1/data/everyday/train.jsonl, 40000 records for matched VSET/readout refit",
        "fit_steps": 1500, "fit_lr_scale": 0.01,
        "dev": str(dev_file), "dev_sha256": digest(dev_file),
        "dev_selection": "1000 disjoint everyday val records after 8192 eligible",
        "accept": "proposal minus matched control <= -0.001 all-target bits/target, paired 95% upper < 0, word-start loss degradation <= +0.01",
        "clean": "r1/data/eval_clean/val1000.jsonl; never used for learning or acceptance",
        "engine": "exact exported integer MICA, bytes plus EOS from BOS",
    })
    try:
        status("learning_thresholds")
        sample = train[:PROPOSAL_RECORDS + SCREEN_RECORDS]
        sm = fit.from_integer(original, spec.MAX_TICKS).to(device)
        pages, raw = fit._page_score_rows(sm, sample, device,
                                          cells=((1, PHASE),))
        _, target = fit.contexts(sample, list(range(1, 10)), device)
        n_proposal = sum(len(r) + 1 for r in sample[:PROPOSAL_RECORDS])
        page_array = pages.cpu().numpy()
        raw_array = raw.cpu().numpy().astype(np.int16)
        target_array = target.cpu().numpy().astype(np.int64)
        del sm, pages, raw, target
        torch.cuda.empty_cache()
        candidate = copy.deepcopy(original)
        diagnostics = {}
        for page in range(PHASE * spec.PAGE_STRIDE,
                          (PHASE + 1) * spec.PAGE_STRIDE):
            a = np.flatnonzero(page_array[:n_proposal] == page)
            b = np.flatnonzero(page_array[n_proposal:] == page) + n_proposal
            if len(a) < 150 or len(b) < 30:
                diagnostics[str(page)] = {"skipped": "insufficient training rows",
                                          "train_rows": len(a), "screen_rows": len(b)}
                continue
            new_bias, report = optimize_page(raw_array[a], target_array[a],
                                             raw_array[b], target_array[b],
                                             original.sc_bias[page])
            candidate.sc_bias[page] = new_bias.astype(candidate.sc_bias.dtype)
            diagnostics[str(page)] = report
        candidate.validate()
        write_json(OUT / "proposal.json", {
            "pages_with_changes": sum(bool(row.get("changed")) for row in
                                      diagnostics.values()),
            "bias_entries_changed": int(np.count_nonzero(
                candidate.sc_bias != original.sc_bias)),
            "diagnostics": diagnostics})
        prefit = OUT / "prefit"
        prefit.mkdir()
        serialize.save(candidate, prefit / "best.mica")
        shutil.copy2(SOURCE / "run_info.json", prefit / "run_info.json")
        del raw_array, page_array, target_array
        for label, seed in (("control", original), ("proposal", candidate)):
            status("refitting", label)
            torch.manual_seed(20260927)
            model = fit.from_integer(seed, spec.MAX_TICKS).to(device)
            with (OUT / f"{label}_refit.log").open("w", encoding="utf-8") as log:
                fit.fit_readout_and_rules(
                    model, train, device, spec.MAX_TICKS, steps=1500,
                    max_records=20000, check_every=250, lr_scale=0.01,
                    log=lambda line: print(line, file=log, flush=True))
            folder = OUT / label
            folder.mkdir()
            serialize.save(to_integer(model), folder / "best.mica")
            shutil.copy2(SOURCE / "run_info.json", folder / "run_info.json")
            del model
            torch.cuda.empty_cache()
        batch.MAX_TICKS = spec.MAX_TICKS
        status("evaluating_development")
        models = {label: serialize.load(OUT / label / "best.mica")
                  for label in ("control", "proposal")}
        dev_scores = {label: exact_metrics(model, dev, batch, spec)
                      for label, model in models.items()}
        for label, metrics in dev_scores.items():
            write_json(OUT / f"{label}_dev1000.json", metrics)
        baseline, trial = dev_scores["control"], dev_scores["proposal"]
        diff = paired_bootstrap(trial["nats"], baseline["nats"],
                                baseline["counts"], n_boot=5000)
        word_diff = paired_bootstrap(trial["word_start_nats"],
                                     baseline["word_start_nats"],
                                     baseline["word_start_counts"], n_boot=5000)
        accepted = bool(diff[0] <= -0.001 and diff[2] < 0 and
                        word_diff[0] <= 0.01)
        result = {"accepted_on_dev": accepted,
                  "dev": {label: {"bits": score["bits"],
                                   "word_start_bits": score["word_start_bits"],
                                   "word_start_top1": score["word_start_top1"],
                                   "sha256": digest(OUT / label / "best.mica")}
                          for label, score in dev_scores.items()},
                  "proposal_minus_control_dev": diff,
                  "word_start_proposal_minus_control_dev": word_diff}
        write_json(OUT / "results.json", result)
        status("evaluating_clean")
        for label in models:
            command(label + "_clean", [
                sys.executable, str(R1 / "checks/eval_int.py"),
                "--run", f"{label}={OUT / label}", "--set", "val1000",
                "--val-file", str(R1 / "data/eval_clean/val1000.jsonl"),
                "--val-skip", "0", "--out", str(OUT / f"{label}_val1000.json")], env)
        clean_scores = {label: json.loads((OUT / f"{label}_val1000.json")
                                         .read_text(encoding="utf-8"))[label]
                        for label in models}
        baseline, trial = clean_scores["control"], clean_scores["proposal"]
        clean_diff = paired_bootstrap(trial["nats"], baseline["nats"],
                                      baseline["counts"], n_boot=5000)
        result["clean"] = {"control_bits": baseline["bits"],
                           "proposal_bits": trial["bits"],
                           "difference": clean_diff,
                           "records": trial["records"],
                           "targets": trial["targets"]}
        result["new_best_evidence"] = bool(
            accepted and trial["bits"] < 1.7423033175727414 and
            clean_diff[2] < 0)
        write_json(OUT / "results.json", result)
        status("generating")
        for label in models:
            command(label + "_generation", [
                sys.executable, str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(OUT / label), "--prompts",
                str(R1 / "checks/generation_dev10.json"), "--limit", "10",
                "--byte-limit", "100", "--out",
                str(OUT / f"{label}_generation.json")], env)
        status("complete")
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
