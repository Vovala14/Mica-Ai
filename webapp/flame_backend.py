"""MICA Flame-W 0.3.3: unchanged B-740 cellular automaton, integer memory blend.

Long-context sentences use w-sent-bos.25-long; short sentences keep w-sent-bos.5f."""
from __future__ import annotations

import sys
import time

from .common import ROOT, clean_prompt, result

RUN = ROOT / "flame/runs/claude_flamew_b740_20261001/train"
MEMORY = ROOT / "flame/runs/codex_flamew_033_20261009/memory.npz"
MEMORY_FITTED = ROOT / "flame/runs/codex_flamew_033_20261009/memory_fitted.npz"
MEMORY_LONG = ROOT / "flame/runs/codex_flamew_033_20261009/memory_long.npz"
VOCAB = ROOT / "r1/data/word/vocab.json"
SHA = "1f4d550386930c8485534368033532301f2db9a49a14a69fd35b702d9e1a1f6d"
MEMORY_SHA = "5fda5a6c744749932bfba7f18352ce532405e60fca7d64fd99b705c28b928be7"
MEMORY_FITTED_SHA = "5564ba558e7c1a6e594e324575333a626df16cc8605c9a23534d880acebab4ac"
MEMORY_LONG_SHA = "f2abb7913dd626fb75cfeb96dd97c17c7d7ecbaed88f54c95fdb986734469f3b"
MODEL = "flame-w-0.3.3"

sys.path.insert(0, str(ROOT / "r1/runs/claude_flame_word_20260928"))
import word_decode as D  # noqa: E402
import word_eval as E  # noqa: E402
sys.path.insert(0, str(ROOT / "r1/runs/claude_flamew_night_20261001"))
import decode_bos as B  # noqa: E402
sys.path.insert(0, str(ROOT / "flame/runs/codex_flamew_033_20261009"))
from memory import Memory  # noqa: E402
from blend_memory import QuarterBlendMicaWord  # noqa: E402

_model = None
_vocab = None
_decoders: dict = {}


def decoder(mode: str, prompt: str = ""):
    global _model, _vocab
    if mode not in ("sentence", "suggest"):
        raise ValueError("Unknown mode.")
    if _model is None:
        _model = QuarterBlendMicaWord(E.load_model(f"mica:{RUN}"),  # original integer CA
                                      Memory(MEMORY), Memory(MEMORY_FITTED), Memory(MEMORY_LONG))
        _vocab = E.W.Vocab.load(VOCAB)
    long_prompt = len(E.prefix_ids(_vocab, prompt.encode("utf-8"))) > 64
    key = (mode, long_prompt if mode == "sentence" else False)
    if key not in _decoders:
        if mode == "sentence":
            settings = ({**B.BOS5F, "lam": 0.25, "name": "w-sent-bos.25-long"}
                        if long_prompt else B.BOS5F)
            _decoders[key] = B.BosDecoder(_model, _vocab, **settings)
        else:
            _decoders[key] = D.WordDecoder(_model, _vocab, **D.MODES[mode])
    return _decoders[key]


def run(body: dict) -> dict:
    prompt = clean_prompt(body)
    mode = body.get("mode", "sentence")
    t0 = time.perf_counter()
    dec = decoder(mode, prompt)
    r = dec.complete(prompt)
    return result(MODEL, mode, prompt, r["continuation"], t0,
                  decoder=dec.name, model_sha256=SHA,
                  memory_sha256=(MEMORY_LONG_SHA if len(E.prefix_ids(_vocab, prompt.encode("utf-8"))) > 64
                                 else MEMORY_SHA),
                  memory_fitted_sha256=MEMORY_FITTED_SHA,
                  bits_per_token=r.get("bits_per_token"))
