#!/usr/bin/env python3
"""Split one busy MICA rule bucket by an integer context predicate.

A dormant rule first copies its parent's VSET program. One score term is
replaced with a training-selected longer-context read. After verifying that
the child mostly steals the parent's rows, both arms refit VSET immediates and
readout on identical training records. Inference remains the native integer CA.
"""
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
OUT = R1 / "runs/codex_bucket_split_20260927"


def choose_split(model, train, fit, spec, device):
    """Train-only information gain; score and purity use hard integer rules."""
    import torch
    sample = train[:2000]
    soft = fit.from_integer(model, spec.MAX_TICKS).to(device)
    page, raw = fit._page_score_rows(soft, sample, device,
                                    cells=((1, 0),))
    ctx, tgt = fit.contexts(sample, list(range(1, 10)), device)
    route = sum(int(model.inj_delta[32, ch] >=
                    model.inj_delta[32, ch + spec.ROUTING_PAIR_OFFSET]) << i
                for i, ch in enumerate(spec.ROUTING_CHANNELS))
    page_id = route
    keep = page == page_id
    page_ctx = ctx[keep].cpu().numpy()
    page_tgt = tgt[keep].cpu().numpy()
    scores = raw[keep].cpu().numpy().astype(np.int32)
    scores += model.sc_bias[page_id].astype(np.int32)
    del soft, page, raw, ctx, tgt
    torch.cuda.empty_cache()
    wins = scores.argmax(axis=1)
    max_score = scores[np.arange(len(wins)), wins]
    usage = np.bincount(wins, minlength=spec.N_CANDIDATES)
    dormant = np.flatnonzero(usage == 0)
    if not len(dormant):
        raise RuntimeError("no dormant candidate to split an active bucket")
    child = int(dormant.max())
    code = np.vstack([model.inj_delta[:, :spec.TAPE_CHANNELS],
                      np.zeros((1, spec.TAPE_CHANNELS), dtype=np.int8)]).astype(np.int32)
    letter = (((page_tgt >= 65) & (page_tgt <= 90)) |
              ((page_tgt >= 97) & (page_tgt <= 122)) |
              ((page_tgt >= 48) & (page_tgt <= 57)) |
              ((page_tgt >= 128) & (page_tgt <= 255)))
    starts = (page_ctx[:, 0] == 32) & letter
    chosen = None
    for parent in np.argsort(-usage)[:8]:
        parent = int(parent)
        if parent == child or usage[parent] < 80:
            continue
        parent_rows = wins == parent
        word_rows = parent_rows & starts
        n = int(word_rows.sum())
        if n < 40:
            continue
        target = page_tgt[word_rows]
        hist = np.bincount(target, minlength=spec.N_SYMBOLS).astype(float)
        base = (hist + 0.5) / (n + 0.5 * spec.N_SYMBOLS)
        def term_variance(j):
            ch = int(model.sc_ch[page_id, parent, j])
            dist = -spec.OFFSETS[int(model.sc_nb[page_id, parent, j])]
            if ch >= spec.TAPE_CHANNELS or dist >= page_ctx.shape[1]:
                return float("inf")
            return float(np.var(code[page_ctx[word_rows, dist], ch]))

        term_order = sorted(range(spec.N_SCORE_TERMS), key=term_variance)
        for term in term_order[:2]:
            old_dist = -spec.OFFSETS[int(model.sc_nb[page_id, parent, term])]
            old_ch = int(model.sc_ch[page_id, parent, term])
            old_co = int(model.sc_co[page_id, parent, term])
            if old_ch >= spec.TAPE_CHANNELS or old_dist >= page_ctx.shape[1]:
                continue
            old_value = old_co * code[page_ctx[:, old_dist], old_ch]
            for dist in (3, 6, 8):
                new_nb = spec.OFFSETS.index(-dist)
                for channel in range(spec.TAPE_CHANNELS):
                    new_value = code[page_ctx[:, dist], channel]
                    for sign in (-1, 1):
                        change = sign * new_value - old_value
                        for quantile in (0.50, 0.60, 0.70, 0.80):
                            threshold = int(np.quantile(change[word_rows], quantile))
                            child_score = scores[:, parent] + change - threshold
                            steals = ((child_score > max_score) |
                                      ((child_score == max_score) & (child < wins)))
                            n_steal = int(steals.sum())
                            if n_steal == 0:
                                continue
                            parent_steal = int((steals & parent_rows).sum())
                            purity = parent_steal / n_steal
                            mask = steals[word_rows]
                            take = int(mask.sum())
                            if purity < 0.90 or take < 20 or take > 0.6 * n:
                                continue
                            yes = np.bincount(target[mask],
                                              minlength=spec.N_SYMBOLS).astype(float)
                            no = hist - yes
                            yes_p = (yes + 12 * base) / (take + 12)
                            no_p = (no + 12 * base) / (n - take + 12)
                            gain = float((yes * np.log(yes_p / base) +
                                          no * np.log(no_p / base)).sum())
                            merit = gain * purity
                            if chosen is None or merit > chosen["merit"]:
                                chosen = {"page": page_id, "parent": parent,
                                          "child": child, "term": term,
                                          "new_nb": new_nb,
                                          "new_dist": dist,
                                          "new_ch": channel, "new_co": sign,
                                          "threshold": threshold,
                                          "parent_usage": int(usage[parent]),
                                          "word_parent": n,
                                          "word_split": take,
                                          "child_wins": n_steal,
                                          "parent_stolen": parent_steal,
                                          "purity": purity,
                                          "gain_nats_training": gain,
                                          "merit": merit}
    if chosen is None:
        raise RuntimeError("no parent-preserving split met support and purity gates")
    return chosen


def install_split(model, split):
    trial = copy.deepcopy(model)
    p, parent, child = (split[k] for k in ("page", "parent", "child"))
    for field in ("sc_nb", "sc_ch", "sc_co", "sc_bias", "op_code", "op_d",
                  "op_n", "op_c", "op_a", "op_b", "op_u", "op_v"):
        arr = getattr(trial, field)
        arr[p, child] = arr[p, parent].copy()
    term = split["term"]
    trial.sc_nb[p, child, term] = split["new_nb"]
    trial.sc_ch[p, child, term] = split["new_ch"]
    trial.sc_co[p, child, term] = split["new_co"]
    trial.sc_bias[p, child] = int(trial.sc_bias[p, parent]) - split["threshold"]
    trial.validate()
    return trial


def main():
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
        raise RuntimeError("GPU required")
    device = torch.device("cuda")
    source_sha = digest(SOURCE / "best.mica")
    if source_sha != "d6cedad89ba210efe4ee50c2b87d6989f9fff241b73c75718b401d14b6e127cc":
        raise ValueError("unexpected source SHA")
    source = serialize.load(SOURCE / "best.mica")
    clean = [bytes.fromhex(s.strip())[:256] for s in
             (R1 / "data/eval_clean/val1000.jsonl").open(encoding="ascii")
             if s.strip()]
    train = sample_train(R1 / "data/everyday/train.jsonl", 40000, set(clean))
    dev = fresh_dev(R1 / "data/everyday/val.jsonl", set(clean), set(train),
                    skip=2048, n=256)
    (OUT / "dev_fresh256.jsonl").write_text(
        "".join(r.hex() + "\n" for r in dev), encoding="ascii")
    write_json(OUT / "protocol.json", {
        "source": str(SOURCE / "best.mica"), "source_sha256": source_sha,
        "train_records": 40000, "selection_records": 2000,
        "fit_steps": 1500, "lr_scale": 0.01,
        "dev_file": str(OUT / "dev_fresh256.jsonl"),
        "dev_sha256": digest(OUT / "dev_fresh256.jsonl"),
        "dev_skip_eligible": 2048,
        "accept": "candidate minus matched control <=-0.002 all bits/target, paired CI upper<0, word-start degradation <=0.01",
        "engine": "native exported integer MICA, C scorer, bytes plus EOS from BOS",
    })

    def status(stage, detail=""):
        write_json(OUT / "status.json", {"stage": stage,
                                         "detail": detail, "updated": time.time()})

    try:
        status("selecting_split")
        split = choose_split(source, train, fit, spec, device)
        write_json(OUT / "split.json", split)
        candidate = install_split(source, split)
        prefit = OUT / "prefit"
        prefit.mkdir()
        serialize.save(candidate, prefit / "best.mica")
        shutil.copy2(SOURCE / "run_info.json", prefit / "run_info.json")
        for label, seed in (("control", source), ("split", candidate)):
            status("refitting", label)
            torch.manual_seed(20260927)
            sm = fit.from_integer(seed, spec.MAX_TICKS).to(device)
            with (OUT / f"{label}_refit.log").open("w", encoding="utf-8") as log:
                fit.fit_readout_and_rules(
                    sm, train, device, spec.MAX_TICKS, steps=1500,
                    max_records=20000, check_every=250, lr_scale=0.01,
                    log=lambda line: print(line, file=log, flush=True))
            folder = OUT / label
            folder.mkdir()
            serialize.save(to_integer(sm), folder / "best.mica")
            shutil.copy2(SOURCE / "run_info.json", folder / "run_info.json")
            del sm
            torch.cuda.empty_cache()
        batch.MAX_TICKS = spec.MAX_TICKS
        status("development_acceptance")
        models = {label: serialize.load(OUT / label / "best.mica")
                  for label in ("control", "split")}
        dev_m = {label: exact_metrics(m, dev, batch, spec)
                 for label, m in models.items()}
        a, b = dev_m["split"], dev_m["control"]
        diff = paired_bootstrap(a["nats"], b["nats"], b["counts"])
        word_diff = paired_bootstrap(a["word_start_nats"],
                                     b["word_start_nats"],
                                     b["word_start_counts"])
        accepted = bool(diff[0] <= -0.002 and diff[2] < 0 and
                        word_diff[0] <= 0.01)
        result = {"accepted_on_dev": accepted,
                  "dev": {label: {"bits": m["bits"],
                                   "word_start_bits": m["word_start_bits"],
                                   "word_start_top1": m["word_start_top1"],
                                   "sha256": digest(OUT / label / "best.mica")}
                          for label, m in dev_m.items()},
                  "split_minus_control": diff,
                  "word_start_split_minus_control": word_diff,
                  "clean": "evaluation pending"}
        write_json(OUT / "results.json", result)
        status("evaluating_clean")
        clean_m = {label: exact_metrics(m, clean, batch, spec)
                   for label, m in models.items()}
        for label, m in clean_m.items():
            write_json(OUT / f"{label}_val1000.json", m)
        a, b = clean_m["split"], clean_m["control"]
        clean_diff = paired_bootstrap(a["nats"], b["nats"], b["counts"])
        result["clean"] = {"control_bits": b["bits"],
                           "split_bits": a["bits"],
                           "difference": clean_diff,
                           "records": len(clean),
                           "targets": a["targets"]}
        result["new_best_evidence"] = bool(
            accepted and a["bits"] < 1.74230331757 and clean_diff[2] < 0)
        write_json(OUT / "results.json", result)
        status("generating")
        for label in models:
            command = [sys.executable,
                       str(R1 / "checks/eval_mica_raw_generation.py"),
                       "--run", str(OUT / label), "--prompts",
                       str(R1 / "checks/generation_dev10.json"), "--limit", "10",
                       "--byte-limit", "100", "--out",
                       str(OUT / f"{label}_generation.json")]
            with (OUT / f"{label}_generation.log").open("w", encoding="utf-8") as stream:
                p = subprocess.run(command, cwd=ROOT, env=dict(os.environ),
                                   stdout=stream, stderr=subprocess.STDOUT)
            if p.returncode:
                raise RuntimeError(f"generation failed: {label} {p.returncode}")
        status("complete")
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
