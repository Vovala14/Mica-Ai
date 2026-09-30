#!/usr/bin/env python3
"""Generate directly from MICA's exact integer engine and reference decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def repeated_fourgram_fraction(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text.casefold())
    phrases = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    return (len(phrases) - len(set(phrases))) / len(phrases) if phrases else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--byte-limit", type=int, default=120)
    args = parser.parse_args()
    if min(args.limit, args.byte_limit) < 1:
        parser.error("limits must be positive")
    info = json.loads((args.run / "run_info.json").read_text(encoding="utf-8"))
    os.environ.update({key: str(value) for key, value in info["env"].items()})
    from mica_r1 import decode, engine, serialize

    config = {"model_sha256": sha(args.run / "best.mica"),
              "prompts_sha256": sha(args.prompts),
              "decoder_sha256": sha(R1 / "mica_r1/decode.py"),
              "byte_limit": args.byte_limit, "limit": args.limit,
              "method": "exact integer engine + reference greedy decoder"}
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))["rows"][:args.limit]
    if args.out.exists():
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        if payload.get("config") != config:
            raise ValueError("existing output uses different settings")
    else:
        payload = {"config": config, "rows": []}
    model = serialize.load(args.run / "best.mica")
    for row in prompts[len(payload["rows"]):]:
        session = engine.new_session(model)
        prompt = row["prompt"]
        error = decode.feed_prompt(model, session, prompt.encode("utf-8"))
        if error:
            raise ValueError(f"invalid prompt {row['id']}: {error}")
        started = time.perf_counter()
        raw, status = decode.generate(model, session, args.byte_limit,
                                      text_mode=True)
        continuation = raw.decode("utf-8")
        payload["rows"].append({"id": row["id"], "prompt": prompt,
                                "continuation": continuation,
                                "full_text": prompt + continuation,
                                "status": status,
                                "seconds": time.perf_counter() - started})
        save(args.out, payload)
        print(f"[mica-raw] {len(payload['rows'])}/{len(prompts)} "
              f"{(prompt + continuation)!r}", flush=True)
    payload["summary"] = {"outputs": len(payload["rows"]),
                          "distinct_continuations": len({
                              row["continuation"] for row in payload["rows"]}),
                          "mean_repeated_fourgram_fraction": sum(
                              repeated_fourgram_fraction(row["continuation"])
                              for row in payload["rows"]) / len(payload["rows"])}
    save(args.out, payload)


if __name__ == "__main__":
    main()
