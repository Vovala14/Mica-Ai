"""MICA Flame-W (word model): official full40 checkpoint, native word decoders."""
from __future__ import annotations

import sys
import time

from .common import ROOT, clean_prompt, result

RUN = ROOT / "flame/runs/codex_flame_word_full40_20260928/train"
VOCAB = ROOT / "r1/data/word/vocab.json"
SHA = "ef969c0e96173fc11ecc3b5cf0d1d28d04e21fe846a0752297d5549e90f7e6b7"
MODEL = "flame-w-full40"

sys.path.insert(0, str(ROOT / "r1/runs/claude_flame_word_20260928"))
import word_decode as D  # noqa: E402
import word_eval as E  # noqa: E402

_model = None
_vocab = None
_decoders: dict = {}


def decoder(mode: str):
    global _model, _vocab
    if mode not in ("sentence", "suggest"):
        raise ValueError("Unknown mode.")
    if mode not in _decoders:
        if _model is None:
            _model = E.load_model(f"mica:{RUN}")   # sets the word geometry, then imports the engine
            _vocab = E.W.Vocab.load(VOCAB)
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
