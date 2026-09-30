#!/usr/bin/env python3
"""One at a time, measure native MICA Flame geometry scaling pilots.

Every training arm starts from the same seed, corpus and one-round fit
recipe as the original lag-64 control. The clean set is scored only after
the disjoint development set; no clean record enters model selection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_mica_overnight import train_command
from run_mica_page_block_refit import digest, sample_train, write_json
from run_mica_word_start_refit import fresh_dev


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
SOURCE = R1 / "runs/ablation/mica_probe64_lag64_full_round1"
OUT = R1 / "runs/codex_flame_scaling_20260928"
CLEAN = R1 / "data/eval_clean/val1000.jsonl"
SOURCE_SHA = "24149019cc0a8cf16a83ce0444f60739691d50c813f9fa196219bb3425905679"
PILOTS = {
    "F1": {"env": {"MICA_CANDIDATES": "1024"}, "args": {}},
    "F2": {"env": {"MICA_ROUTING_CHANNELS": "0,1,2,3,4,5,6"}, "args": {}},
    "F3": {"env": {"MICA_PROBE": "432"},
           "args": {"work_probes": "384",
                    "work_lags": "1:16,2:16,3:8,4:8,8:8,16:8"}},
    "F4": {"env": {"MICA_PHASES": "32", "MICA_TICKS": "32",
                   "MICA_CHANNELS": "208"}, "args": {"ticks": "32"}},
}


def status(pilot: str, stage: str, detail: str = "") -> None:
    write_json(OUT / pilot / "status.json", {
        "pilot": pilot, "stage": stage, "detail": detail, "updated": time.time()})


def run_logged(label: str, cmd: list[str], env: dict[str, str],
               folder: Path) -> None:
    with (folder / f"{label}.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label} exited {result.returncode}; see log")


def prepare() -> tuple[dict, Path]:
    if digest(SOURCE / "best.mica") != SOURCE_SHA:
        raise RuntimeError("one-round control checkpoint changed")
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    if (info["args"]["steps"] != 1 or
            info["args"]["fit_records"] != 40000 or
            info["args"]["fit_steps"] != 1500):
        raise RuntimeError("one-round control recipe changed")
    OUT.mkdir(parents=True, exist_ok=True)
    dev_file = OUT / "dev_fresh1000.jsonl"
    protocol_file = OUT / "protocol.json"
    if not dev_file.exists():
        clean = {bytes.fromhex(line.strip())[:256] for line in
                 CLEAN.open(encoding="ascii") if line.strip()}
        train = sample_train(R1 / "data/everyday/train.jsonl", 40000, clean)
        dev = fresh_dev(R1 / "data/everyday/val.jsonl", clean,
                        set(train), skip=10240, n=1000)
        dev_file.write_text("".join(row.hex() + "\n" for row in dev),
                            encoding="ascii")
        write_json(protocol_file, {
            "source": str(SOURCE / "best.mica"), "source_sha256": SOURCE_SHA,
            "plan": "docs/2026-09-27-ember-flame-plan.md",
            "training": "same seed 1, everyday train, one fit round, 40000 fit records, 1500 fit steps, round_lr 0.3",
            "pilots": PILOTS, "development": str(dev_file),
            "development_sha256": digest(dev_file),
            "development_selection": "1000 unique everyday validation records after skipping 10240 eligible; exclude clean and sampled training records",
            "nomination_gate": "pilot minus one-round baseline <= -0.020 bits/target on fresh dev1000 with paired 95% CI upper < 0; clean val1000 is reporting only and must independently confirm before a knob is kept",
            "clean": str(CLEAN),
            "engine": "exact exported integer MICA C scorer, pooled bytes plus EOS from BOS",
        })
    elif (not protocol_file.exists() or
          json.loads(protocol_file.read_text(encoding="utf-8"))[
              "development_sha256"] != digest(dev_file)):
        raise RuntimeError("saved development protocol/hash mismatch")
    return info, dev_file


def train_args(info: dict, pilot: str, dest: Path) -> tuple[list[str], dict]:
    config = PILOTS[pilot]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("MICA_")}
    env.update({key: str(value) for key, value in info["env"].items()})
    env.update(config["env"])
    env["PYTHONUTF8"] = "1"
    cmd = train_command(info, dest, 1)
    cmd += ["--tape-lags", info["args"]["tape_lags"]]
    for key, value in config["args"].items():
        flag = "--" + key.replace("_", "-")
        if flag in cmd:
            cmd[cmd.index(flag) + 1] = value
        else:
            cmd += [flag, value]
    return cmd, env


def evaluate(pilot: str, folder: Path, dev_file: Path) -> None:
    from common import paired_bootstrap

    model_dir = folder / "train"
    if not (model_dir / "FINISHED").exists():
        raise RuntimeError("training has not finished")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("MICA_")}
    env["PYTHONUTF8"] = "1"
    scores = {}
    for set_name, path in (("dev1000", dev_file), ("clean_val1000", CLEAN)):
        status(pilot, "evaluating", set_name)
        output = folder / f"{set_name}.json"
        cmd = [sys.executable, str(R1 / "checks/eval_int.py"),
               "--run", f"baseline={SOURCE}", "--run", f"pilot={model_dir}",
               "--set", "val1000", "--val-file", str(path),
               "--val-skip", "0", "--out", str(output)]
        run_logged(set_name, cmd, env, folder)
        result = json.loads(output.read_text(encoding="utf-8"))
        if set(result) != {"baseline", "pilot"}:
            raise RuntimeError(f"{set_name} missing model result")
        control, candidate = result["baseline"], result["pilot"]
        if control["counts"] != candidate["counts"] or control["records"] != 1000:
            raise RuntimeError(f"{set_name} record alignment failed")
        diff = paired_bootstrap(candidate["nats"], control["nats"],
                                control["counts"], n_boot=5000)
        scores[set_name] = {
            "baseline_bits": control["bits"], "pilot_bits": candidate["bits"],
            "pilot_minus_baseline": diff, "records": candidate["records"],
            "targets": candidate["targets"]}
        write_json(folder / "results.json", {
            "pilot": pilot, "model": str(model_dir / "best.mica"),
            "model_sha256": digest(model_dir / "best.mica"),
            "baseline": str(SOURCE / "best.mica"),
            "baseline_sha256": SOURCE_SHA, "scores": scores,
            "nominated_on_dev": bool(
                scores.get("dev1000", {}).get("pilot_minus_baseline", [1, 1, 1])[0]
                <= -0.020 and
                scores.get("dev1000", {}).get("pilot_minus_baseline", [1, 1, 1])[2]
                < 0)})
    status(pilot, "generating")
    run_logged("generation", [
        sys.executable, str(R1 / "checks/eval_mica_raw_generation.py"),
        "--run", str(model_dir), "--prompts",
        str(R1 / "checks/generation_dev10.json"), "--limit", "10",
        "--byte-limit", "100", "--out", str(folder / "generation.json")],
        env, folder)
    status(pilot, "complete")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", required=True, choices=sorted(PILOTS))
    ap.add_argument("--stage", required=True,
                    choices=("dry-run", "train", "evaluate"))
    ap.add_argument("--fit-after-gradient-oom", action="store_true",
                    help="fit-mode pilot only: run after the straight-through "
                         "dry-run OOM at batch 1, which fit mode does not use")
    args = ap.parse_args()
    pilot, stage = args.pilot, args.stage
    info, dev_file = prepare()
    folder = OUT / pilot
    folder.mkdir(exist_ok=True)
    try:
        if stage == "dry-run":
            if (folder / "dry_run.log").exists():
                raise RuntimeError("dry-run already exists; inspect it")
            status(pilot, "dry_running")
            cmd, env = train_args(info, pilot, folder / "dry_run")
            run_logged("dry_run", cmd + ["--dry-run"], env, folder)
            status(pilot, "dry_run_passed")
        elif stage == "train":
            prior = json.loads((folder / "status.json").read_text(
                encoding="utf-8")) if (folder / "status.json").exists() else {}
            dry_run_passed = prior.get("stage") == "dry_run_passed"
            allowed_fit_oom = (args.fit_after_gradient_oom and pilot in ("F1", "F2")
                               and info["args"]["mode"] == "fit"
                               and prior.get("stage") == "failed"
                               and "dry-run:" in prior.get("detail", "")
                               and "[dry-run] out of memory even at batch 1" in
                               (folder / "dry_run.log").read_text(encoding="utf-8"))
            if not (dry_run_passed or allowed_fit_oom):
                raise RuntimeError("successful dry-run or documented fit-mode OOM exception required")
            dest = folder / "train"
            if dest.exists() and any(dest.iterdir()):
                raise RuntimeError("training output already exists")
            if (ROOT.parent / "STOP_MICA").exists():
                raise RuntimeError("STOP_MICA is present")
            status(pilot, "training")
            cmd, env = train_args(info, pilot, dest)
            run_logged("train", cmd, env, folder)
            if not (dest / "FINISHED").exists() or not (dest / "best.mica").exists():
                raise RuntimeError("training exited without FINISHED/best.mica")
            status(pilot, "training_complete")
        else:
            evaluate(pilot, folder, dev_file)
    except Exception as exc:
        status(pilot, "failed", f"{stage}: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
