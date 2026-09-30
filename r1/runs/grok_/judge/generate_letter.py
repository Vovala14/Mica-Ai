#!/usr/bin/env python3
"""Letter-control sentences on the same frozen prompts.

Integer byte engine plus a sentence counterpart of w-sent-mmi.3:
whole words, pair ban, MMI lambda 0.3, no rerolls. Prints counts only.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
R1 = ROOT / "r1"
JUDGE = Path(__file__).resolve().parent
RUN = R1 / "runs" / "codex_page_block_refit_20260927_v2" / "control"
MODEL = RUN / "best.mica"
VOCAB = R1 / "data" / "word" / "vocab.json"
PROMPTS = JUDGE / "prompts" / "mica-complete-v0.1-dev.jsonl"
EXPECT_MODEL = "d6cedad89ba210efe4ee50c2b87d6989f9fff241b73c75718b401d14b6e127cc"

LAM = 0.3
TOP = 4
PHRASE_BEAM = 4
BYTE_BEAM = 8
MAX_WORDS = 20
MIN_WORDS = 3
ENDS = (ord("."), ord("!"), ord("?"))
COMMA = ord(",")
SPACE = ord(" ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_words():
    payload = json.loads(VOCAB.read_text(encoding="utf-8"))
    tokens = payload["tokens"]
    surface = payload.get("surface") or tokens
    return tokens, surface


def lp_after(scorer, log_probs, context: bytes, continuation: bytes) -> float:
    state = scorer.start(context)
    total = 0.0
    for byte in continuation:
        probs = log_probs(scorer.scores(state))
        total += float(probs[byte])
        state = scorer.advance(state, byte)
    return total


def search(scorer, vocabulary, suggest, prompt: str) -> dict:
    t0 = time.perf_counter()
    prefix, history0, _joiner = suggest._prompt_context(prompt)
    history = tuple(history0) + ((prefix.casefold(),) if prefix else ())
    state = scorer.start(prompt.encode("utf-8"))
    logp = 0.0
    if prefix:
        probs = suggest._log_probs(scorer.scores(state))
        logp += float(probs[SPACE])
        state = scorer.advance(state, SPACE)
        emitted = b" "
    else:
        emitted = b""
    beam = [{
        "state": state,
        "emitted": emitted,
        "history": history,
        "lp": logp,
        "words": 0,
        "done": False,
    }]
    finished = []
    for step in range(MAX_WORDS):
        expanded = []
        for hyp in beam:
            if hyp["done"]:
                continue
            if step + 1 < MIN_WORDS:
                boundaries = (SPACE,)
            elif step + 1 < MAX_WORDS:
                boundaries = (SPACE, COMMA, *ENDS)
            else:
                boundaries = ENDS
            choices = suggest._word_choices(
                scorer, hyp["state"], vocabulary.root, hyp["history"],
                beam_width=BYTE_BEAM, max_word_bytes=32, max_loop=1,
                limit=64, continue_after_space=step + 1 < MAX_WORDS,
                boundaries=boundaries,
            )
            pairs = set(zip(hyp["history"], hyp["history"][1:]))
            prev = hyp["history"][-1] if hyp["history"] else None
            picks = []
            for choice in choices:
                word = choice.word.casefold()
                if prev is not None and (word == prev or (prev, word) in pairs):
                    continue
                if choice.boundary not in (SPACE, COMMA) and choice.boundary not in ENDS:
                    continue
                picks.append(choice)
                if len(picks) >= TOP:
                    break
            for choice in picks:
                raw = hyp["emitted"] + choice.suffix + bytes((choice.boundary,))
                words = hyp["words"] + 1
                ended = choice.boundary in ENDS
                if choice.boundary == SPACE:
                    nxt_state = choice.state
                elif choice.boundary == COMMA and not ended:
                    nxt_state = scorer.advance(choice.state, choice.boundary)
                else:
                    nxt_state = choice.state
                nxt = {
                    "state": nxt_state,
                    "emitted": raw,
                    "history": hyp["history"] + (choice.word.casefold(),),
                    "lp": hyp["lp"] + choice.logp,
                    "words": words,
                    "done": ended or words >= MAX_WORDS,
                }
                if words >= MIN_WORDS and ended:
                    finished.append(nxt)
                if not nxt["done"]:
                    expanded.append(nxt)
        expanded.sort(key=lambda hyp: (-hyp["lp"], hyp["emitted"]))
        beam = expanded[:PHRASE_BEAM]
        if not beam:
            break
    cands = finished or [hyp for hyp in beam if hyp["words"]]
    if not cands:
        return {
            "continuation": "",
            "ids": [],
            "candidates": 0,
            "seconds": time.perf_counter() - t0,
        }
    prompt_bytes = prompt.encode("utf-8")
    last = prefix.encode("utf-8") if prefix else prompt_bytes
    for hyp in cands:
        continuation = hyp["emitted"]
        hyp["lp"] = lp_after(scorer, suggest._log_probs, prompt_bytes, continuation)
        hyp["lpg"] = lp_after(scorer, suggest._log_probs, last, continuation)
        hyp["n"] = max(1, len(continuation))
    best = min(cands, key=lambda hyp: -(hyp["lp"] - LAM * hyp["lpg"]) / hyp["n"])
    raw = best["emitted"]
    while raw.endswith(b","):
        raw = raw[:-1]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
        raw = b""
    return {
        "continuation": text,
        "ids": list(raw),
        "candidates": len(cands),
        "seconds": time.perf_counter() - t0,
    }


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nshard = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if nshard < 1 or not 0 <= shard < nshard:
        raise SystemExit("bad shard")
    model_sha = sha256(MODEL)
    if model_sha != EXPECT_MODEL:
        raise SystemExit(f"model sha {model_sha} != {EXPECT_MODEL}")
    info = json.loads((RUN / "run_info.json").read_text(encoding="utf-8"))
    for key in list(os.environ):
        if key.startswith("MICA_"):
            os.environ.pop(key)
    for key, value in info["env"].items():
        os.environ[key] = str(value)
    os.environ.pop("MICA_SYMBOLS", None)
    sys.path.insert(0, str(R1))
    from mica_r1 import serialize, spec
    from mica_r1 import suggest

    if spec.N_SYMBOLS != 258 or spec.BOS != 256 or spec.EOS != 257:
        raise SystemExit(f"unexpected letter geometry {spec.N_SYMBOLS}")
    model = serialize.load(str(MODEL))
    scorer = suggest.MicaScorer(model)
    tokens, surface = load_words()
    words = []
    for token, shown in zip(tokens, surface):
        form = shown if suggest.WORD_RE.fullmatch(shown or "") else token
        if suggest.WORD_RE.fullmatch(form or ""):
            words.append(form)
    vocabulary = suggest.Vocabulary(words)
    rows = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for index, row in enumerate(rows[:limit]) if index % nshard == shard]
    out_name = "letter.jsonl" if nshard == 1 else f"letter.shard{shard}.jsonl"
    out = JUDGE / "gen" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    fresh = [row for row in rows if row["id"] not in done]
    print(f"letter shard {shard}/{nshard} vocab {vocabulary.size} new {len(fresh)}", flush=True)
    t0 = time.perf_counter()
    with out.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(fresh, start=1):
            result = search(scorer, vocabulary, suggest, item["prompt"])
            row = {
                "system": "letter",
                "decoding": "letter-sent-mmi.3",
                "model_sha256": model_sha,
                "engine": "integer-byte-mica",
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "continuation": result["continuation"],
                "ids": result["ids"],
                "candidates": result["candidates"],
                "seconds": result["seconds"],
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            gc.collect()
            if index % 5 == 0 or index == len(fresh):
                print(
                    f"letter {index}/{len(fresh)} elapsed {time.perf_counter() - t0:.0f}s",
                    flush=True,
                )
    print(f"letter wrote {out} sha {model_sha[:12]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
