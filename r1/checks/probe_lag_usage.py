#!/usr/bin/env python3
"""Summarize learned integer readout coefficients by MICA probe lag."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    info = json.loads((args.run / "run_info.json").read_text(encoding="utf-8"))
    os.environ.update({key: str(value) for key, value in info["env"].items()})
    from mica_r1 import serialize, spec

    model = serialize.load(args.run / "best.mica")
    lag = spec.N_CELLS - model.pr_cell.astype(np.int32)
    coef = model.pr_co.astype(np.int32)
    tape = model.pr_chan < spec.TAPE_CHANNELS
    rows = []
    for value in sorted(set(lag[tape].tolist())):
        selected = tape & (lag == value)
        absolute = np.abs(coef[selected])
        rows.append({"lag": int(value), "entries": int(selected.sum()),
                     "nonzero_fraction": float(np.mean(absolute > 0)),
                     "mean_absolute_coefficient": float(np.mean(absolute)),
                     "median_absolute_coefficient": float(np.median(absolute))})
    result = {"model": str(args.run / "best.mica"), "tape_lags": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
