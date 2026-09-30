#!/usr/bin/env python3
"""Greedy text from a byte/letter MICA model (such as Ember), R1 §8 decoder.

    python r1/generate_bytes.py RUN_DIR "Prompt text" [--limit 120]

RUN_DIR holds best.mica and run_info.json; the run's MICA_* geometry is set
before the engine is imported.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run", type=Path)
ap.add_argument("prompt")
ap.add_argument("--limit", type=int, default=120, help="maximum output bytes")
a = ap.parse_args()

info = json.loads((a.run / "run_info.json").read_text(encoding="utf-8"))
os.environ.update({k: str(v) for k, v in info["env"].items()})
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mica_r1 import decode, engine, serialize  # noqa: E402  (spec binds MICA_* at import)

model = serialize.load(a.run / "best.mica")
s = engine.new_session(model)
err = decode.feed_prompt(model, s, a.prompt.encode("utf-8"))
if err is not None:
    sys.exit(f"invalid prompt: {err}")
out, status = decode.generate(model, s, a.limit)
print(a.prompt + out.decode("utf-8"))
print(f"[{status}]", file=sys.stderr)
