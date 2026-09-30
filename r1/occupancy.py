#!/usr/bin/env python3
"""Field occupancy versus position in the record.

Finding 11 in docs/r1-findings.md: what MICA learns is specific to how full the
field is, and none of the diagnostics §14 names would have caught it. This is
the missing instrument. It answers two questions at once:

  * how quickly does the field fill, and does it saturate before the record
    ends — i.e. is the model's working memory already exhausted?
  * how does that change with geometry, which is what decides whether a scale
    study is worth running?

Run it under any geometry:

    MICA_CELLS=768 MICA_CHANNELS=32 python3 r1/occupancy.py
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

import numpy as np
from make_records import load_records
from mica_r1 import spec, engine
from mica_r1.engine import random_model, Session, ingest


def occupancy(model, records, marks) -> dict:
    N, C = spec.N_CELLS, spec.N_CHANNELS
    touched = {m: [] for m in marks}
    written = {m: [] for m in marks}
    updates = {m: [] for m in marks}

    for rec in records:
        s = Session()
        ingest(model, s, spec.BOS)
        prev_upd = s.updates
        for pos, byte in enumerate(rec, start=1):
            ingest(model, s, byte)
            if pos in touched:
                touched[pos].append(float((s.F != 0).any(axis=1).mean()))
                written[pos].append(float((s.F != 0).mean()))
                updates[pos].append((s.updates - prev_upd) / pos)
        prev_upd = prev_upd

    return {m: {"cells_touched": round(float(np.mean(touched[m])), 4),
                "channels_written": round(float(np.mean(written[m])), 4),
                "updates_per_symbol": round(float(np.mean(updates[m])), 1)}
            for m in marks if touched[m]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="r1/data/records/val.jsonl")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--routing", default="r1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    engine.ROUTING_MODE = args.routing

    pool = [r for r in load_records(args.records)[:900] if len(r) == 256][:args.n]
    marks = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256]
    m = random_model(args.seed)
    res = occupancy(m, pool, marks)

    g = spec.describe()
    print(f"geometry: {g['cells']} cells x {g['channels']} channels "
          f"= {g['field_state_bytes']:,} bytes of field; routing {args.routing}")
    print(f"{'after byte':>11}{'cells touched':>16}{'channels written':>19}{'upd/sym':>10}")
    print("-" * 56)
    for k in marks:
        if k not in res:
            continue
        r = res[k]
        bar = "#" * int(r["cells_touched"] * 30)
        print(f"{k:>11}{r['cells_touched']:>15.1%}{r['channels_written']:>19.1%}"
              f"{r['updates_per_symbol']:>10.0f}  {bar}")
    full = [k for k in marks if k in res and res[k]["cells_touched"] > 0.99]
    if full:
        print(f"\nEvery cell has been written by byte {min(full)}. Past that "
              f"point the field is no longer filling, it is overwriting: what\n"
              f"the model remembers is bounded by what the rules choose to "
              f"preserve, not by capacity.")
    else:
        print(f"\nThe field is still filling at 256 bytes "
              f"({res[max(res)]['cells_touched']:.1%} of cells touched).")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"geometry": g, "routing": args.routing, "occupancy": res}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
