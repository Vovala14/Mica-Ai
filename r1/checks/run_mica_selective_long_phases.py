#!/usr/bin/env python3
"""One-round matched MICA test: extend only four long-context rule phases."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_overnight import train_command
from run_mica_page_block_refit import digest, sample_train, write_json
from run_mica_word_start_refit import exact_metrics, fresh_dev


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
SOURCE = R1 / "runs/ablation/mica_probe64_lag64_full_round1"
OUT = R1 / "runs/codex_selective_long_phases_20260927"
SCHEDULE = "1,2,2,3,2,3,3,4,1,2,3,4,2,3,4,4"
SELECTIVE = "1,2,2,3,2,3,3,6,1,2,3,6,2,3,6,6"


def status(stage: str, detail: str = "") -> None:
    write_json(OUT / "status.json", {"stage": stage, "detail": detail,
                                     "updated": time.time()})


def run_command(label: str, cmd: list[str], env: dict[str, str]) -> None:
    with (OUT / f"{label}.log").open("w", encoding="utf-8") as stream:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                              stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"{label} exited {proc.returncode}; inspect its log")


def main() -> None:
    evaluate_existing = sys.argv[1:] == ["--evaluate-existing"]
    if OUT.exists() and any(OUT.iterdir()) and not evaluate_existing:
        raise SystemExit(f"refusing nonempty output directory: {OUT}")
    if evaluate_existing and not all(
            (OUT / label / "FINISHED").exists()
            for label in ("control", "selective")):
        raise SystemExit("both completed training checkpoints are required")
    if (ROOT.parent / "STOP_MICA").exists():
        raise SystemExit("STOP_MICA is present")
    OUT.mkdir(parents=True, exist_ok=True)
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    if digest(SOURCE / "best.mica") != (
            "24149019cc0a8cf16a83ce0444f60739691d50c813f9fa196219bb3425905679"):
        raise RuntimeError("unexpected source checkpoint")
    if info["args"]["rule_max_back"] != SCHEDULE or info["args"]["steps"] != 1:
        raise RuntimeError("source protocol differs")
    for key in list(os.environ):
        if key.startswith("MICA_"):
            del os.environ[key]
    os.environ.update(info["env"])
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    sys.path[:0] = [str(R1), str(R1 / "checks")]
    from mica_r1 import batch, serialize, spec
    from common import paired_bootstrap

    clean = [bytes.fromhex(s.strip())[:256] for s in
             (R1 / "data/eval_clean/val1000.jsonl").open(encoding="ascii")
             if s.strip()]
    dev_file = OUT / "dev_fresh512.jsonl"
    if evaluate_existing:
        dev = [bytes.fromhex(line.strip()) for line in
               dev_file.open(encoding="ascii") if line.strip()]
        if len(dev) != 512:
            raise RuntimeError("saved development set is incomplete")
    else:
        sampled_train = sample_train(R1 / "data/everyday/train.jsonl", 40000,
                                     set(clean))
        dev = fresh_dev(R1 / "data/everyday/val.jsonl", set(clean),
                        set(sampled_train), skip=4096, n=512)
        dev_file.write_text("".join(r.hex() + "\n" for r in dev),
                            encoding="ascii")
    protocol = {
        "source": str(SOURCE / "best.mica"),
        "source_sha256": digest(SOURCE / "best.mica"),
        "control_schedule": SCHEDULE, "selective_schedule": SELECTIVE,
        "changed_phases": [7, 11, 14, 15],
        "training": "matched seed 1, everyday train, 1 round, 40000 fit records, 1500 steps",
        "development": str(dev_file), "dev_sha256": digest(dev_file),
        "dev_selection": "512 fresh everyday val records after 4096 eligible, excluding clean and sampled train",
        "accept": "selective minus control <= -0.005 bits/target, paired 95% upper < 0, word-start loss degradation <= +0.01",
        "clean": "r1/data/eval_clean/val1000.jsonl; never used for fitting or selection",
        "evaluation": "exact exported integer engine, C scorer, bytes plus EOS from BOS",
    }
    if evaluate_existing:
        if json.loads((OUT / "protocol.json").read_text(
                encoding="utf-8"))["dev_sha256"] != digest(dev_file):
            raise RuntimeError("saved development set hash changed")
    else:
        write_json(OUT / "protocol.json", protocol)
    try:
        if not evaluate_existing:
            for label, schedule in (("control", SCHEDULE),
                                    ("selective", SELECTIVE)):
                status("training", label)
                folder = OUT / label
                cmd = train_command(info, folder, 1)
                cmd[cmd.index("--rule-max-back") + 1] = schedule
                cmd.extend(["--tape-lags", info["args"]["tape_lags"]])
                run_command(label + "_train", cmd, env)
                if not (folder / "FINISHED").exists():
                    raise RuntimeError(f"{label} missing FINISHED marker")
        status("evaluating_development")
        batch.MAX_TICKS = spec.MAX_TICKS
        models = {label: serialize.load(OUT / label / "best.mica")
                  for label in ("control", "selective")}
        dev_metrics = {label: exact_metrics(model, dev, batch, spec)
                       for label, model in models.items()}
        for label, metrics in dev_metrics.items():
            write_json(OUT / f"{label}_dev512.json", metrics)
        control, selective = dev_metrics["control"], dev_metrics["selective"]
        diff = paired_bootstrap(selective["nats"], control["nats"],
                                control["counts"], n_boot=5000)
        word_diff = paired_bootstrap(selective["word_start_nats"],
                                     control["word_start_nats"],
                                     control["word_start_counts"], n_boot=5000)
        accepted = bool(diff[0] <= -0.005 and diff[2] < 0 and
                        word_diff[0] <= 0.01)
        result = {"accepted_on_development": accepted,
                  "dev": {label: {"bits": metrics["bits"],
                                   "word_start_bits": metrics["word_start_bits"],
                                   "word_start_top1": metrics["word_start_top1"],
                                   "sha256": digest(OUT / label / "best.mica")}
                          for label, metrics in dev_metrics.items()},
                  "selective_minus_control_dev": diff,
                  "word_start_selective_minus_control_dev": word_diff}
        write_json(OUT / "results.json", result)
        status("evaluating_clean")
        clean_metrics = {label: exact_metrics(model, clean, batch, spec)
                         for label, model in models.items()}
        for label, metrics in clean_metrics.items():
            write_json(OUT / f"{label}_val1000.json", metrics)
        control, selective = clean_metrics["control"], clean_metrics["selective"]
        clean_diff = paired_bootstrap(selective["nats"], control["nats"],
                                      control["counts"], n_boot=5000)
        result["clean"] = {"control_bits": control["bits"],
                           "selective_bits": selective["bits"],
                           "selective_minus_control": clean_diff,
                           "records": len(clean), "targets": selective["targets"]}
        result["new_best_evidence"] = bool(
            accepted and selective["bits"] < 1.7423033175727414 and
            clean_diff[2] < 0)
        write_json(OUT / "results.json", result)
        status("generating")
        for label in models:
            run_command(label + "_generation", [
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
