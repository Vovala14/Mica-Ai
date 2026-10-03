#!/usr/bin/env python3
"""Flame-W B-740 + memory: continue a prompt, the same way the website does.

    python flame/runs/claude_flamew_memory_20261003/generate.py "I was thinking about"
    python flame/runs/claude_flamew_memory_20261003/generate.py --mode suggest "Can you send me"
    add --no-memory for the automaton alone
"""
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "r1/runs/claude_flame_word_20260928"))
sys.path.insert(0, str(ROOT / "r1/runs/claude_flamew_night_20261001"))
sys.path.insert(0, str(HERE))
import word_eval as E  # noqa: E402
import word_decode as D  # noqa: E402
import decode_bos as B  # noqa: E402
from memory import Memory, MemoryMicaWord  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt"); ap.add_argument("--mode", choices=["sentence", "suggest"], default="sentence")
    ap.add_argument("--no-memory", action="store_true")
    a = ap.parse_args()
    model = E.load_model(f"mica:{ROOT / 'flame/runs/claude_flamew_b740_20261001/train'}")
    if not a.no_memory:
        model = MemoryMicaWord(model, Memory(HERE / "memory.npz"))
    vocab = E.W.Vocab.load(ROOT / "r1/data/word/vocab.json")
    dec = B.BosDecoder(model, vocab, **B.BOS5F) if a.mode == "sentence" else D.WordDecoder(model, vocab, **D.MODES["suggest"])
    print(a.prompt + dec.complete(a.prompt)["continuation"])


if __name__ == "__main__":
    main()
