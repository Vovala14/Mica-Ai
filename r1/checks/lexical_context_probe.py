#!/usr/bin/env python3
"""Check word-state survival and its effect on MICA's exact next-byte scores."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import run_geometry, worker_env

PAIRS = (
    ("xxxxxxxxabcdefgh ", "yyyyyyyyabcdefgh "),
    ("xxxxxxxxabcdefgh next", "yyyyyyyyabcdefgh next"),
    ("xxxxxxxxabcdefgh next ", "yyyyyyyyabcdefgh next "),
    ("aaaaaaaaconnection ", "bbbbbbbbconnection "),
    ("The ability ", "The disability "),
    ("A payment ", "A repayment "),
)


def worker(args) -> None:
    import numpy as np
    from mica_r1 import engine, serialize, spec

    model = serialize.load(args.mica)
    rows = []
    for left, right in PAIRS:
        width = max(len(left), len(right))
        feed_left, feed_right = left.rjust(width), right.rjust(width)
        states, scores = [], []
        for prompt in (feed_left, feed_right):
            session = engine.new_session(model)
            for byte in prompt.encode("ascii"):
                engine.ingest(model, session, byte)
            head = (session.position - 1) % spec.N_CELLS
            states.append(session.F[head, spec.TAPE_CHANNELS:
                                         spec.TAPE_CHANNELS + 12].tolist())
            scores.append(engine.probe_scores(model, session).astype(np.int64))
        x, y = scores
        eligible = np.r_[:256, spec.EOS]
        px = np.exp((x[eligible] - x[eligible].max()) /
                    spec.LOGIT_DIVISOR)
        py = np.exp((y[eligible] - y[eligible].max()) /
                    spec.LOGIT_DIVISOR)
        px /= px.sum(); py /= py.sum()
        rows.append({"left": left, "right": right,
                     "shared_suffix_bytes": len(os.path.commonprefix(
                         [feed_left[::-1], feed_right[::-1]])),
                     "track0_equal": states[0][:6] == states[1][:6],
                     "track1_equal": states[0][6:] == states[1][6:],
                     "score_equal": bool(np.array_equal(x, y)),
                     "max_score_delta": int(np.max(np.abs(x - y))),
                     "total_variation": float(np.abs(px - py).sum() / 2)})
    Path(args.out).write_text(json.dumps({"model": args.mica,
                                         "rows": rows}, indent=2) + "\n")
    for row in rows:
        print(f"[lexical-probe] suffix={row['shared_suffix_bytes']} "
              f"track0_equal={row['track0_equal']} "
              f"scores_equal={row['score_equal']} "
              f"TV={row['total_variation']:.6f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mica")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    cmd = [sys.executable, __file__, "--worker", "--run", str(args.run),
           "--out", str(args.out), "--mica", str(args.run / "best.mica")]
    raise SystemExit(subprocess.call(cmd, env=worker_env(run_geometry(args.run))))


if __name__ == "__main__":
    main()
