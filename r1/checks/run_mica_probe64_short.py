#!/usr/bin/env python3
"""Short matched MICA-only 32-versus-64-byte tape-probe pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

R1 = Path(__file__).resolve().parents[1]
ROOT = R1.parent
SOURCE = R1 / "runs/ablation/tape_long_short/run_info.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", action="store_true",
                        help="try MICA's cross-byte rule state with lag 64")
    args = parser.parse_args()
    info = json.loads(SOURCE.read_text(encoding="utf-8"))
    a = info["args"]
    env = os.environ.copy()
    env.update({key: str(value) for key, value in info["env"].items()})
    env["MICA_PROBE_WINDOW"] = "64"
    common = [
        "--records", a["records"], "--val-records", a["val_records"],
        "--steps", "3", "--batch", str(a["batch"]),
        "--record-bytes", str(a["record_bytes"]),
        "--ticks", str(a["ticks"]),
        "--tau", str(a["tau"]), "--sel-init", str(a["sel_init"]),
        "--sel-tau", str(a["sel_tau"]),
        "--lr", str(a["lr"]), "--int-lr", str(a["int_lr"]),
        "--init", a["init"], "--work-probes", str(a["work_probes"]),
        "--bias-init", a["bias_init"], "--mode", "fit",
        "--rule-max-back", a["rule_max_back"],
        "--rule-scoring", a["rule_scoring"],
        "--rule-self-terms", str(a["rule_self_terms"]),
        "--rule-state", "stream" if args.stream else "none",
        "--fit-records", "500", "--fit-steps", "30",
        "--round-lr", str(a["round_lr"]),
        "--val-records-n", "64",
        "--checkpoint-every", "1", "--save-every", "50",
        "--gpu-mem-fraction", str(a["gpu_mem_fraction"]),
        "--min-free-ram-gb", str(a["min_free_ram_gb"]),
    ]
    if args.stream:
        arms = [("mica_probe64_stream_lag64_short",
                 "1:16,2:8,3:4,4:4,8:4,16:4,32:4,64:4")]
    else:
        arms = [
            ("mica_probe64_lag32_short",
             "1:16,2:8,3:4,4:4,8:4,16:4,24:4,32:4"),
            ("mica_probe64_lag64_short",
             "1:16,2:8,3:4,4:4,8:4,16:4,32:4,64:4"),
        ]
    for name, lags in arms:
        out = R1 / "runs/ablation" / name
        if (out / "FINISHED").exists():
            print(f"[mica-probe64] {name}: already finished", flush=True)
            continue
        command = [sys.executable, str(R1 / "train_soft.py"), *common,
                   "--tape-lags", lags, "--out", str(out)]
        print(f"[mica-probe64] start {name}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        print(f"[mica-probe64] finished {name}", flush=True)


if __name__ == "__main__":
    main()
