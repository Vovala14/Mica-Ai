#!/usr/bin/env python3
"""Matched MICA-only tape-readout lag comparison, one fit round per arm.

Both arms use the same exact integer rule engine and corpus. The only changed
setting is where the existing tape probes read recent bytes. No auxiliary
language model participates in training or inference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

R1 = Path(__file__).resolve().parents[1]
ROOT = R1.parent
REFERENCE = R1 / "runs/sweep/everyday_c256/run_info.json"
OUT = R1 / "runs/ablation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-window", type=int, choices=(32, 64),
                        default=32)
    args = parser.parse_args()
    info = json.loads(REFERENCE.read_text(encoding="utf-8"))
    base = info["args"]
    env = os.environ.copy()
    env.update({key: str(value) for key, value in info["env"].items()})
    env["MICA_PROBE_WINDOW"] = str(args.probe_window)
    flags = [
        "--records", base["records"],
        "--val-records", base["val_records"],
        "--steps", "1", "--batch", str(base["batch"]),
        "--record-bytes", str(base["record_bytes"]),
        "--ticks", str(base["ticks"]),
        "--tau", str(base["tau"]),
        "--sel-init", str(base["sel_init"]),
        "--sel-tau", str(base["sel_tau"]),
        "--lr", str(base["lr"]),
        "--int-lr", str(base["int_lr"]),
        "--init", base["init"],
        "--work-probes", str(base["work_probes"]),
        "--bias-init", base["bias_init"],
        "--mode", "fit",
        "--rule-max-back", base["rule_max_back"],
        "--rule-scoring", base["rule_scoring"],
        "--rule-self-terms", str(base["rule_self_terms"]),
        "--rule-state", "none",
        "--fit-records", "40000", "--fit-steps", "1500",
        "--round-lr", str(base["round_lr"]),
        "--warmup", str(base["warmup"]),
        "--clip", str(base["clip"]),
        "--val-records-n", "256",
        "--checkpoint-every", "0",
        "--save-every", "50", "--log-every", "10",
        "--tbptt", str(base["tbptt"]),
        "--min-batch", str(base["min_batch"]),
        "--gpu-mem-fraction", str(base["gpu_mem_fraction"]),
        "--min-free-ram-gb", str(base["min_free_ram_gb"]),
    ]
    if base["segments"]:
        flags.append("--segments")
    if args.probe_window == 32:
        arms = [
            ("mica_tape_default_full_round1", ""),
            ("mica_tape_long32_full_round1",
             "1:16,2:8,3:4,4:4,8:4,16:4,24:4,32:4"),
        ]
    else:
        arms = [
            ("mica_probe64_lag32_full_round1",
             "1:16,2:8,3:4,4:4,8:4,16:4,24:4,32:4"),
            ("mica_probe64_lag64_full_round1",
             "1:16,2:8,3:4,4:4,8:4,16:4,32:4,64:4"),
        ]
    for name, tape_lags in arms:
        path = OUT / name
        if (path / "FINISHED").exists():
            print(f"[mica-lag] {name}: already finished", flush=True)
            continue
        command = [sys.executable, str(R1 / "train_soft.py"), *flags,
                   "--out", str(path)]
        if tape_lags:
            command.extend(["--tape-lags", tape_lags])
        print(f"[mica-lag] start {name}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        print(f"[mica-lag] finished {name}", flush=True)


if __name__ == "__main__":
    main()
