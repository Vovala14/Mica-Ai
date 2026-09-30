#!/usr/bin/env python3
"""Verify the exported integer MICA cells implement the intended word state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import run_geometry, worker_env

PROMPTS = ("The ability ", "A repayment ", "xxxxxxxxabcdefgh next ")


def worker(args) -> None:
    from mica_r1 import engine, serialize, spec
    from mica_r1.fit import WORD_BYTES

    model = serialize.load(args.mica)
    K = spec.TAPE_CHANNELS

    def route(symbol: int) -> int:
        code = model.inj_delta[symbol, :K]
        bits = sum((int(code[c] >= code[c + spec.ROUTING_PAIR_OFFSET]) << i)
                   for i, c in enumerate(spec.ROUTING_CHANNELS))
        return bits

    def decode(session, track: int) -> int:
        head = (session.position - 1) % spec.N_CELLS
        values = session.F[head, K + track * 6:K + track * 6 + 6]
        if int(values[5]) != 0:
            raise AssertionError("state padding channel changed")
        state = 0
        for i, value in enumerate(values[:5]):
            if int(value) not in (-127, 0, 127):
                raise AssertionError(f"invalid ternary state value {value}")
            state += (int(value) // 127 + 1) * 3 ** i
        return state

    rows = []
    for prompt in PROMPTS:
        session = engine.new_session(model)
        previous, current = 0, 121  # after BOS
        checked = 0
        for symbol in prompt.encode("ascii"):
            page = route(symbol)
            word = symbol in WORD_BYTES
            old_current = current
            previous = previous if word else old_current % 121
            if not word:
                current = 121 + old_current % 121
            elif old_current >= 121:
                current = page + 1
            else:
                current = (5 * old_current + page + 1) % 121
            engine.ingest(model, session, symbol)
            actual = (decode(session, 0), decode(session, 1))
            expected = (previous, current)
            if actual != expected:
                raise AssertionError(f"{prompt!r} byte {checked}: "
                                     f"actual {actual}, expected {expected}")
            checked += 1
        rows.append({"prompt": prompt, "bytes_checked": checked,
                     "previous_word_state": previous,
                     "current_word_state": current})
    args.out.write_text(json.dumps({"model": args.mica,
                                    "rows": rows}, indent=2) + "\n")
    print(f"[lexical-state] verified {sum(x['bytes_checked'] for x in rows)} "
          "exported integer transitions", flush=True)


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
