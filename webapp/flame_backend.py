"""MICA Flame-W (word model): checkpoint B-740 with the memory readout (v0.3, 2026-10-03).

Sentence mode uses decoder w-sent-bos.5f; next-words mode uses w6-mmi.5."""
from __future__ import annotations

import sys
import time

from .common import ROOT, clean_prompt, result

RUN = ROOT / "flame/runs/claude_flamew_b740_20261001/train"
MEMORY = ROOT / "flame/runs/claude_flamew_memory_20261003/memory.npz"
VOCAB = ROOT / "r1/data/word/vocab.json"
SHA = "1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d"
MODEL = "flame-w-memory-20261003"

sys.path.insert(0, str(ROOT / "r1/runs/claude_flame_word_20260928"))
import word_decode as D  # noqa: E402
import word_eval as E  # noqa: E402
sys.path.insert(0, str(ROOT / "r1/runs/claude_flamew_night_20261001"))
import decode_bos as B  # noqa: E402
sys.path.insert(0, str(ROOT / "flame/runs/claude_flamew_memory_20261003"))
from memory import Memory, MemoryMicaWord  # noqa: E402

_model = None
_vocab = None
_decoders: dict = {}


def decoder(mode: str):
    global _model, _vocab
    if mode not in ("sentence", "suggest"):
        raise ValueError("Unknown mode.")
    if mode not in _decoders:
        if _model is None:
            _model = MemoryMicaWord(E.load_model(f"mica:{RUN}"),   # sets the word geometry, then imports the engine
                                    Memory(MEMORY))
            _vocab = E.W.Vocab.load(VOCAB)
        if mode == "sentence":
            _decoders[mode] = B.BosDecoder(_model, _vocab, **B.BOS5F)
        else:
            _decoders[mode] = D.WordDecoder(_model, _vocab, **D.MODES[mode])
    return _decoders[mode]


def run(body: dict) -> dict:
    prompt = clean_prompt(body)
    mode = body.get("mode", "sentence")
    t0 = time.perf_counter()
    dec = decoder(mode)
    r = dec.complete(prompt)
    return result(MODEL, mode, prompt, r["continuation"], t0,
                  decoder=dec.name, model_sha256=SHA,
                  bits_per_token=r.get("bits_per_token"))
