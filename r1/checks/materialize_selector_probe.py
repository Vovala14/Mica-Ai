#!/usr/bin/env python3
"""Export a rejected selector probe for independent analysis, without promotion."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--donor", type=Path, required=True)
    ap.add_argument("--search", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    info = json.loads((args.control / "run_info.json").read_text())
    donor_info = json.loads((args.donor / "run_info.json").read_text())
    if info["env"] != donor_info["env"]:
        raise ValueError("geometry mismatch")
    for key in list(os.environ):
        if key.startswith("MICA_"):
            del os.environ[key]
    os.environ.update({k: str(v) for k, v in info["env"].items()})
    from mica_r1 import serialize, spec

    status = json.loads((args.search / "status.json").read_text())
    if status["stage"] != "selection_complete":
        raise ValueError("search has not completed")
    best = status["best_proposal"]
    if not best:
        raise ValueError("no proposal to export")
    fields = best.get("fields", [best.get("field")])
    if any(field not in ("sc_nb", "sc_ch", "sc_co") for field in fields):
        raise ValueError("unexpected selector field")
    control = serialize.load(args.control / "best.mica")
    donor = serialize.load(args.donor / "best.mica")
    page = slice(best["phase"] * spec.PAGE_STRIDE,
                 (best["phase"] + 1) * spec.PAGE_STRIDE)
    for field in fields:
        getattr(control, field)[page] = getattr(donor, field)[page]
    out = args.search / "best_probe"
    out.mkdir(exist_ok=True)
    serialize.save(control, out / "best.mica")
    shutil.copy2(args.control / "run_info.json", out / "run_info.json")
    (out / "proposal.json").write_text(json.dumps(best, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
