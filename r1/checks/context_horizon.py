#!/usr/bin/env python3
"""Measure whether earlier bytes can change MICA's next-symbol distribution.

For each shared suffix length, compare prompts with different, equal-length
prefixes. Equal prefix lengths keep the rolling cell position aligned. Exact
score equality after a suffix means that this prefix difference is invisible
to the exported integer model at that point. This diagnostic does not judge
whether a completion is useful; run the fixed completion benchmark for that.

    python r1/checks/context_horizon.py --run r1/runs/sweep/everyday_c256 \
        --out r1/runs/checks/everyday_context_horizon.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import run_geometry, worker_env

LENGTHS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
SUFFIX = (b" and then we went home for dinner before writing a short note "
          b"about what happened earlier in the day. ") * 2
PREFIX_PAIRS = (
    (b"The chef chopped onions for dinner tonight. ",
     b"The pilot checked the route for flight today. "),
    (b"A" * 48, b"Z" * 48),
    (b"The chef prepared dinner for the guests. ",
     b"The coach prepared drills for the team. "),
    (b"I bought fresh fruit at the market today. ",
     b"I watched a film at the theater today. "),
    (b"The rain soaked every street in the city. ",
     b"The sun warmed every street in the city. "),
    (b"Please send the report to my work email. ",
     b"Please bring the suitcase to my hotel. "),
)


def aligned(a: bytes, b: bytes) -> tuple[bytes, bytes]:
    """Left-pad prefixes so their differing final words stay near the probe."""
    n = max(len(a), len(b))
    return a.rjust(n, b" "), b.rjust(n, b" ")


def worker(a) -> int:
    import numpy as np
    from mica_r1 import engine, serialize, spec

    m = serialize.load(a.mica)
    result = {"model": str(a.mica), "suffix_lengths": list(LENGTHS),
              "pair_count": len(PREFIX_PAIRS), "rows": []}
    for length in LENGTHS:
        suffix = SUFFIX[:length]
        equal = 0
        max_score_delta = 0
        top_equal = 0
        max_tv = 0.0
        for pa, pb in PREFIX_PAIRS:
            pa, pb = aligned(pa, pb)
            scores = []
            for prompt in (pa + suffix, pb + suffix):
                session = engine.new_session(m)
                for byte in prompt:
                    engine.ingest(m, session, byte)
                scores.append(engine.probe_scores(m, session).astype(np.int64))
            x, y = scores
            equal += int(np.array_equal(x, y))
            max_score_delta = max(max_score_delta, int(np.max(np.abs(x - y))))
            elig = np.r_[:256, spec.EOS]
            top_equal += int(elig[np.argmax(x[elig])] == elig[np.argmax(y[elig])])
            px = np.exp((x[elig] - x[elig].max()) / spec.LOGIT_DIVISOR)
            py = np.exp((y[elig] - y[elig].max()) / spec.LOGIT_DIVISOR)
            px /= px.sum(); py /= py.sum()
            max_tv = max(max_tv, float(np.abs(px - py).sum() / 2))
        row = {"shared_suffix_bytes": length, "exactly_equal_pairs": equal,
               "same_top_symbol_pairs": top_equal,
               "max_score_delta": max_score_delta,
               "max_total_variation": max_tv}
        result["rows"].append(row)
        print(f"[horizon] suffix {length:3d}: {equal}/{len(PREFIX_PAIRS)} "
              f"identical scores, max TV {max_tv:.6f}", flush=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mica")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.worker:
        return worker(a)
    run = Path(a.run)
    cmd = [sys.executable, __file__, "--worker", "--run", str(run),
           "--mica", str(run / "best.mica"), "--out", str(out)]
    return subprocess.call(cmd, env=worker_env(run_geometry(run)))


if __name__ == "__main__":
    raise SystemExit(main())
