#!/usr/bin/env python3
"""Bit-identical replay of Flame-W 0.3.1 playground sentences.

Uses the website sentence path: B-740 automaton, 0.3.1 memory readout,
answer mode off, decoder w-sent-bos.5f. Weights are in this repo.

    python flame/replay/replay_flamew_031.py --check

`--check` reruns flame/replay/prompts.jsonl and exits 0 only when the
UTF-8 bytes match flame/replay/out.jsonl. Same prompt, same continuation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROMPTS = HERE / "prompts.jsonl"
OUT = HERE / "out.jsonl"

AUTOMATON = ROOT / "flame/runs/claude_flamew_b740_20261001/train/best.mica"
MEMORY = ROOT / "flame/runs/claude_flamew_031_20261004/memory.npz"
AUTOMATON_SHA256 = "1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d"
MEMORY_SHA256 = "5fda5a6c744749932bfba7f18352ce532405e60fca7d64fd99b705c28b928be7"
DECODER = "w-sent-bos.5f"
MODEL = "flame-w-0.3.1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompts() -> list[dict]:
    rows = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no prompts in {PROMPTS}")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate prompt id")
    for r in rows:
        if not isinstance(r.get("id"), str) or not isinstance(r.get("prompt"), str) or not r["prompt"].strip():
            raise SystemExit(f"bad prompt row: {r!r}")
    return rows


def render(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)


def generate(prompts: list[dict]) -> list[dict]:
    auto = sha256(AUTOMATON)
    mem = sha256(MEMORY)
    if auto != AUTOMATON_SHA256 or mem != MEMORY_SHA256:
        raise SystemExit(f"checkpoint mismatch\nautomaton {auto}\nmemory {mem}")
    # Pin the historical route directly so a newer playground release cannot
    # silently change the 0.3.1 replay.
    sys.path.insert(0, str(ROOT / "r1/runs/claude_flame_word_20260928"))
    import word_eval as E  # noqa: E402
    sys.path.insert(0, str(ROOT / "r1/runs/claude_flamew_night_20261001"))
    import decode_bos as B  # noqa: E402
    sys.path.insert(0, str(ROOT / "flame/runs/claude_flamew_031_20261004"))
    from memory import Memory, MemoryMicaWord  # noqa: E402
    model = MemoryMicaWord(E.load_model(f"mica:{AUTOMATON.parent}"), Memory(MEMORY))
    vocab = E.W.Vocab.load(ROOT / "r1/data/word/vocab.json")
    dec = B.BosDecoder(model, vocab, **B.BOS5F)
    if dec.name != DECODER:
        raise SystemExit(f"decoder is {dec.name}, expected {DECODER}")
    out = []
    for item in prompts:
        continuation = dec.complete(item["prompt"])["continuation"]
        out.append({
            "answer_mode": False,
            "automaton_sha256": auto,
            "continuation": continuation,
            "decoder": dec.name,
            "id": item["id"],
            "memory_sha256": mem,
            "model": MODEL,
            "prompt": item["prompt"],
            "text": item["prompt"] + continuation,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="fail unless out.jsonl matches a fresh run")
    args = ap.parse_args()
    text = render(generate(load_prompts()))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if args.check:
        got = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if got != text:
            sys.stderr.write("replay mismatch: continuations differ from flame/replay/out.jsonl\n")
            return 1
        print(f"match {digest}  {text.count(chr(10))} lines  {OUT.relative_to(ROOT)}")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {digest}  {text.count(chr(10))} lines  {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
