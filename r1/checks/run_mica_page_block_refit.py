#!/usr/bin/env python3
"""Matched MICA page-block selector rewrite with joint integer-rule refit.

Only two existing rule pages for the space byte are changed. A corpus-informed
hard selector book is installed there, then all VSET immediates and readout
weights are refitted on training data in both matched arms. Exact integer
evaluation and greedy generation use the original cellular engine.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
SOURCE = R1 / "runs/overnight/mica_lag64_continuation_20260927"
OUT = R1 / "runs/codex_page_block_refit_20260927"
PYTHON = sys.executable


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sample_train(path: Path, n: int, exclude: set[bytes]) -> list[bytes]:
    """Uniform deterministic reservoir sample from the full training split."""
    rng = random.Random(20260927)
    rows: list[bytes] = []
    eligible = 0
    with path.open(encoding="ascii") as src:
        for line in src:
            record = bytes.fromhex(line.strip())[:256]
            if not record or record in exclude:
                continue
            eligible += 1
            if len(rows) < n:
                rows.append(record)
            else:
                i = rng.randrange(eligible)
                if i < n:
                    rows[i] = record
    if len(rows) < n:
        raise ValueError("not enough training records")
    return rows


def disjoint_dev(path: Path, clean: set[bytes], train: set[bytes],
                 n: int) -> list[bytes]:
    rows = []
    seen = set()
    with path.open(encoding="ascii") as src:
        for line in src:
            record = bytes.fromhex(line.strip())[:256]
            if record and record not in clean and record not in train \
                    and record not in seen:
                rows.append(record)
                seen.add(record)
                if len(rows) == n:
                    break
    if len(rows) != n:
        raise ValueError("not enough disjoint development records")
    return rows


def run(label: str, command: list[str], env: dict[str, str]) -> None:
    with (OUT / f"{label}.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label} exited {result.returncode}")


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()
    if args.lr_scale <= 0:
        parser.error("--lr-scale must be positive")
    if not 1 <= args.block_size <= 256:
        parser.error("--block-size must be between 1 and 256")
    OUT = args.out.resolve()
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
    import numpy as np
    import torch

    if spec.WINDOW != 1 or spec.VSET_WIDTH != 6 or spec.TAPE_CHANNELS != 16:
        raise ValueError("pilot requires original lag64 VSET6 tape geometry")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU needed for matched VSET/readout refit")
    device = torch.device("cuda")
    source_sha = digest(SOURCE / "best.mica")
    original = serialize.load(SOURCE / "best.mica")
    clean = [bytes.fromhex(line.strip())[:256] for line in
             (R1 / "data/eval_clean/val1000.jsonl").open(encoding="ascii")
             if line.strip()]
    clean_set = set(clean)
    train = sample_train(R1 / "data/everyday/train.jsonl", 40000, clean_set)
    dev = disjoint_dev(R1 / "data/everyday/val.jsonl", clean_set,
                       set(train), 128)
    (OUT / "dev_accept128.jsonl").write_text(
        "".join(row.hex() + "\n" for row in dev), encoding="ascii")
    train_hash = hashlib.sha256(b"".join(train)).hexdigest()
    space_code = original.inj_delta[ord(" "), :spec.TAPE_CHANNELS]
    route = sum(int(space_code[ch] >= space_code[ch + spec.ROUTING_PAIR_OFFSET])
                << i for i, ch in enumerate(spec.ROUTING_CHANNELS))
    pages = [route + phase * spec.PAGE_STRIDE for phase in (0, 1)]
    write_json(OUT / "protocol.json", {
        "source": str(SOURCE / "best.mica"), "source_sha256": source_sha,
        "candidate_pages": pages, "space_route": route,
        "changed_rulebook": "hard score triples and biases on two space-route pages",
        "block_size_per_page": args.block_size,
        "slot_choice": "least-used baseline candidates on 150 training records",
        "method": "information-ranked context templates at 3 back, followed by joint VSET/readout refit",
        "train_file": "r1/data/everyday/train.jsonl", "train_records": len(train),
        "train_sample_sha256": train_hash, "refit_steps": 1500,
        "refit_lr_scale": args.lr_scale,
        "refit_sample_limit": 20000,
        "development": "128 disjoint everyday/val records, not clean val1000",
        "clean_validation": "r1/data/eval_clean/val1000.jsonl, exact integer",
    })

    def status(stage: str, detail: str = "") -> None:
        write_json(OUT / "status.json", {"stage": stage, "detail": detail,
                                         "updated": time.time()})

    try:
        status("proposing")
        template = fit.from_integer(original, spec.MAX_TICKS).to(device)
        fit.template_rules(template, train, device,
                           max_back=tuple([3] * spec.N_PHASE),
                           max_records=5000, ranking="information",
                           phases=(0, 1))
        templated = to_integer(template)
        del template
        candidate = serialize.load(SOURCE / "best.mica")
        selected = {}
        if args.block_size == spec.N_CANDIDATES:
            for field in ("sc_nb", "sc_ch", "sc_co", "sc_bias"):
                getattr(candidate, field)[pages] = getattr(templated, field)[pages]
        else:
            baseline_soft = fit.from_integer(original, spec.MAX_TICKS).to(device)
            page_ids, raw_scores = fit._page_score_rows(
                baseline_soft, train[:150], device, cells=((1, 0), (1, 1)))
            c = spec.N_CANDIDATES
            bias = baseline_soft.sc_bias.detach().round().int()
            tie = torch.arange(c, device=device, dtype=torch.int64)
            winners = ((raw_scores.long() + bias[page_ids]) * (c + 1) -
                       tie).argmax(-1)
            uses = torch.bincount(page_ids * c + winners,
                                  minlength=spec.N_PAGES * c).reshape(
                                      spec.N_PAGES, c).cpu().numpy()
            for page in pages:
                order = sorted(range(c), key=lambda k: (int(uses[page, k]), -k))
                slots = order[:args.block_size]
                selected[str(page)] = {
                    "slots": slots,
                    "baseline_wins_in_sample": [int(uses[page, k]) for k in slots],
                    "page_observations": int(uses[page].sum()),
                }
                for field in ("sc_nb", "sc_ch", "sc_co", "sc_bias"):
                    getattr(candidate, field)[page, slots] = getattr(
                        templated, field)[page, :args.block_size]
            del baseline_soft, page_ids, raw_scores, winners
            torch.cuda.empty_cache()
        candidate.validate()
        prefit = OUT / "block_prefit"
        prefit.mkdir()
        serialize.save(candidate, prefit / "best.mica")
        shutil.copy2(SOURCE / "run_info.json", prefit / "run_info.json")
        write_json(OUT / "proposal.json", {
            "pages": pages, "selectors_changed": {
                field: int(np.count_nonzero(getattr(candidate, field)[pages] !=
                                             getattr(original, field)[pages]))
                for field in ("sc_nb", "sc_ch", "sc_co", "sc_bias")},
            "prefit_sha256": digest(prefit / "best.mica"),
            "selected": selected,
        })

        for arm, seed_model in (("control", original), ("block", candidate)):
            status("refitting", arm)
            torch.manual_seed(20260927)
            sm = fit.from_integer(seed_model, spec.MAX_TICKS).to(device)
            initial = to_integer(sm)
            check = OUT / f"{arm}_initial.mica"
            serialize.save(initial, check)
            expected = (SOURCE / "best.mica") if arm == "control" else \
                (prefit / "best.mica")
            if digest(check) != digest(expected):
                raise AssertionError(f"{arm} SoftMica conversion changed hard model")
            with (OUT / f"{arm}_refit.log").open("w", encoding="utf-8") as log:
                fit.fit_readout_and_rules(
                    sm, train, device, spec.MAX_TICKS, steps=1500,
                    max_records=20000, check_every=250,
                    lr_scale=args.lr_scale,
                    log=lambda line: print(line, file=log, flush=True))
            arm_dir = OUT / arm
            arm_dir.mkdir()
            serialize.save(to_integer(sm), arm_dir / "best.mica")
            shutil.copy2(SOURCE / "run_info.json", arm_dir / "run_info.json")
            del sm
            torch.cuda.empty_cache()

        batch.MAX_TICKS = spec.MAX_TICKS

        def dev_score(model):
            stack = batch.ModelStack([model], score_backend="c")
            _, _, _, per_record, _ = batch.evaluate(stack, dev, per_record=True)
            counts = np.asarray([len(row) + 1 for row in dev], dtype=np.int64)
            nats = np.asarray(per_record[0]) * counts
            return float(nats.sum() / counts.sum() / math.log(2)), nats, counts

        status("development_acceptance")
        control = serialize.load(OUT / "control/best.mica")
        block = serialize.load(OUT / "block/best.mica")
        control_bits, control_nats, counts = dev_score(control)
        block_bits, block_nats, block_counts = dev_score(block)
        if not np.array_equal(counts, block_counts):
            raise AssertionError("development target counts differ")
        difference, lo, hi = paired_bootstrap(block_nats, control_nats, counts)
        accepted = difference <= -0.005 and hi < 0
        result = {"accepted": accepted,
                  "control": {"file": str(OUT / "control/best.mica"),
                              "sha256": digest(OUT / "control/best.mica"),
                              "dev128_bits": control_bits},
                  "block": {"file": str(OUT / "block/best.mica"),
                            "sha256": digest(OUT / "block/best.mica"),
                            "dev128_bits": block_bits},
                  "block_minus_control": difference,
                  "paired_95ci": [lo, hi], "min_gain": 0.005,
                  "clean_val1000": "not used for acceptance"}
        write_json(OUT / "results.json", result)
        status("evaluating_clean")
        env = dict(os.environ)
        for arm in ("control", "block"):
            arm_dir = OUT / arm
            run(arm + "_clean", [PYTHON, str(R1 / "checks/eval_int.py"),
                "--run", f"{arm}={arm_dir}", "--set", "val1000",
                "--val-file", str(R1 / "data/eval_clean/val1000.jsonl"),
                "--val-skip", "0", "--out", str(OUT / f"{arm}_val1000.json")], env)
            run(arm + "_generation", [PYTHON,
                str(R1 / "checks/eval_mica_raw_generation.py"),
                "--run", str(arm_dir), "--prompts",
                str(R1 / "checks/generation_dev10.json"), "--limit", "10",
                "--byte-limit", "100", "--out",
                str(OUT / f"{arm}_generation.json")], env)
        rows = {arm: json.loads((OUT / f"{arm}_val1000.json")
                                .read_text(encoding="utf-8"))[arm]
                for arm in ("control", "block")}
        if rows["control"]["counts"] != rows["block"]["counts"]:
            raise AssertionError("clean target counts differ")
        c_diff, c_lo, c_hi = paired_bootstrap(
            rows["block"]["nats"], rows["control"]["nats"],
            rows["block"]["counts"])
        result["clean_val1000"] = {
            "control_bits": rows["control"]["bits"],
            "block_bits": rows["block"]["bits"],
            "difference": c_diff, "paired_95ci": [c_lo, c_hi],
            "records": rows["block"]["records"],
            "targets": rows["block"]["targets"]}
        result["new_best_evidence"] = bool(
            accepted and rows["block"]["bits"] < 1.76562195856
            and c_hi < 0)
        write_json(OUT / "results.json", result)
        status("complete")
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
