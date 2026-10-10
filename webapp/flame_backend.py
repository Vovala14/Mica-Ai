"""MICA Flame-W 0.3.4: B-740 plus experimental cellular belief-memory QA.

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
MODEL = "flame-w-0.3.4"
BELIEF_RUN = ROOT / "flame/runs/codex_flamew_034_20261010"
BELIEF_RULE_SHA = "9a0ad7b1cc0c607176092d333f29f1ce4b993f460fde1a57fab338aca1400a9c"
GROUNDING_RULE_SHA = "0a2edf02857ca8a021aadb4a0e1f9d7b4cc918957e2ff32cfaabdb6b4e981c0c"

sys.path.insert(0, str(ROOT / "r1/runs/claude_flame_word_20260928"))
import word_decode as D  # noqa: E402
import word_eval as E  # noqa: E402
sys.path.insert(0, str(ROOT / "r1/runs/claude_flamew_night_20261001"))
import decode_bos as B  # noqa: E402
sys.path.insert(0, str(ROOT / "flame/runs/codex_flamew_033_20261009"))
from memory import Memory  # noqa: E402
from blend_memory import QuarterBlendMicaWord  # noqa: E402
sys.path.insert(0, str(BELIEF_RUN))
from belief_memory import Adapter as BeliefAdapter  # noqa: E402

_model = None
_vocab = None
_decoders: dict = {}
_belief = None
_answer_model = None


def answer(body: dict, prompt: str, t0: float) -> dict:
    """Four-choice QA: definite learned CA-memory answer or exact 0.3.3 fallback."""
    global _belief, _answer_model, _vocab
    question = body.get("question", "")
    choices = body.get("choices")
    if not isinstance(question, str) or len(question) > 400:
        raise ValueError("Questions are limited to 400 characters.")
    if (not isinstance(choices, list) or len(choices) != 4
            or any(not isinstance(c, str) or not c.strip() or len(c) > 200 for c in choices)
            or len(set(choices)) != 4):
        raise ValueError("Provide four distinct nonempty choices, up to 200 characters each.")
    if _belief is None:
        _belief = BeliefAdapter()
    pred = _belief.predict(prompt, question, choices)
    index = pred["answer"]
    route = "cellular-belief-memory"
    if index is None:
        route = "word-model-fallback"
        if _answer_model is None:
            mode_path = ROOT / "flame/runs/claude_flamew_031_20261004/answer_mode.npz"
            _answer_model = QuarterBlendMicaWord(E.load_model(f"mica:{RUN}"),
                Memory(MEMORY, mode_path), Memory(MEMORY_FITTED, mode_path), Memory(MEMORY_LONG, mode_path))
        if _vocab is None:
            _vocab = E.W.Vocab.load(VOCAB)
        ids = lambda s: [E.W.SEP if t == "<sep>" else _vocab.index.get(t, E.W.oov_id(t)) for t in E.W.tokenize(s)]
        state = _answer_model.start()
        for token in ids(prompt + (" " + question if question else "")):
            state = _answer_model.feed(state, token)
        scores = []
        for choice in choices:
            branch = _answer_model.fork(state)
            tokens = ids(choice)
            if not tokens:
                raise ValueError("Each choice must contain a word or punctuation.")
            total = 0.0
            for token in tokens:
                total += float(_answer_model.logp(branch, [token])[0])
                branch = _answer_model.feed(branch, token)
            scores.append(total / len(tokens))
        index = max(range(4), key=lambda i: scores[i])
    # Signed prompt includes all four choices so later feedback retains the task.
    logged = prompt + ("\nQuestion: " + question if question else "") + "\n" + "\n".join(
        f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
    return result(MODEL, "answer", logged, choices[index], t0,
        answer_index=int(index), choices=choices, route=route, decoder="cellular-belief-memory-or-mean-word",
        model_sha256=SHA, belief_rule_sha256=BELIEF_RULE_SHA, grounding_rule_sha256=GROUNDING_RULE_SHA)


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
    if mode == "answer":
        return answer(body, prompt, t0)
    dec = decoder(mode, prompt)
    r = dec.complete(prompt)
    return result(MODEL, mode, prompt, r["continuation"], t0,
                  decoder=dec.name, model_sha256=SHA,
                  memory_sha256=(MEMORY_LONG_SHA if len(E.prefix_ids(_vocab, prompt.encode("utf-8"))) > 64
                                 else MEMORY_SHA),
                  memory_fitted_sha256=MEMORY_FITTED_SHA,
                  bits_per_token=r.get("bits_per_token"))
