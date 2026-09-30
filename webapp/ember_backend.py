"""MICA Ember (byte/letter model): official v0.2A checkpoint, R1 decoder."""
from __future__ import annotations

import json
import math
import os
import sys
import time

from .common import ROOT, clean_prompt, result

RUN = ROOT / "ember/runs/codex_ember_balanced_v02a_20260928/train"
SHA = "3b2a94e203fc6b0cd20365e4d5c46085035722f578161a2213b8cf148a126e72"
MODEL = "ember-v02a"

_info = json.loads((RUN / "run_info.json").read_text(encoding="utf-8"))
os.environ.update({k: str(v) for k, v in _info["env"].items()})
sys.path.insert(0, str(ROOT / "r1"))
import numpy as np  # noqa: E402
from mica_r1 import decode, engine, serialize, spec  # noqa: E402  (spec binds MICA_* at import)

_model = None


def model():
    global _model
    if _model is None:
        _model = serialize.load(RUN / "best.mica")
    return _model


def _probs(m, s, st, temperature: float) -> np.ndarray:
    z = decode.probe_scores(m, s).astype(np.float64) / spec.LOGIT_DIVISOR / temperature
    z = np.where(decode.eligible_mask(st, True), z, -np.inf)
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def _show(sym: int) -> str:
    if sym == spec.EOS:
        return "⏎ end"
    if sym == 32:
        return "␣"
    return chr(sym) if sym < 128 else f"0x{sym:02x}"


def sample(m, s, limit: int, temperature: float, rng) -> tuple[bytes, str]:
    """decode.generate with sampling instead of argmax; same UTF-8 display rules."""
    out, pending, st = bytearray(), bytearray(), decode.Utf8State()
    while len(out) + len(pending) < limit:
        sym = int(rng.choice(len(p := _probs(m, s, st, temperature)), p=p))
        if sym == spec.EOS:
            s.ended = True
            return bytes(out), decode.EOS_STOP
        pending.append(sym)
        st = decode.advance(st, sym)
        if st.at_boundary():
            out.extend(pending)
            pending.clear()
        engine.ingest(m, s, sym)
    return bytes(out), decode.INCOMPLETE_LIMIT if pending else decode.LIMIT_STOP


def run(body: dict) -> dict:
    prompt = clean_prompt(body)
    try:
        temperature = float(body.get("temperature", 0))
        limit = int(body.get("limit", 120))
    except (TypeError, ValueError):
        raise ValueError("Bad settings.")
    if not (0 <= temperature <= 2) or not (1 <= limit <= 240):
        raise ValueError("Temperature must be 0-2 and length 1-240 bytes.")
    t0 = time.perf_counter()
    m = model()
    s = engine.new_session(m)
    if decode.feed_prompt(m, s, prompt.encode("utf-8")) is not None:
        raise ValueError("The prompt is not valid text.")
    p = _probs(m, s, decode.Utf8State(), 1.0)
    top = [{"byte": _show(int(i)), "p": round(float(p[i]), 4)} for i in np.argsort(-p)[:6]]
    if temperature == 0:
        out, status = decode.generate(m, s, limit)
    else:
        out, status = sample(m, s, limit, temperature, np.random.default_rng())
    mode = "greedy" if temperature == 0 else f"sample-t{temperature:g}"
    return result(MODEL, mode, prompt, out.decode("utf-8"), t0,
                  status=str(status), model_sha256=SHA, next_byte=top,
                  temperature=temperature, limit=limit)
