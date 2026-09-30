#!/usr/bin/env python3
"""W2 letter arm letter-sent-mmi.3-end. Counts only. Does not print text."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import generate_letter as base

JUDGE = Path(__file__).resolve().parent
OUT_DIR = JUDGE / "w2" / "gen"
DECODER = "letter-sent-mmi.3-end"


def search_end(scorer, vocabulary, suggest, prompt: str) -> dict:
    t0 = time.perf_counter()
    prefix, history0, _joiner = suggest._prompt_context(prompt)
    history = tuple(history0) + ((prefix.casefold(),) if prefix else ())
    state = scorer.start(prompt.encode("utf-8"))
    logp = 0.0
    if prefix:
        probs = suggest._log_probs(scorer.scores(state))
        logp += float(probs[base.SPACE])
        state = scorer.advance(state, base.SPACE)
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
    for step in range(base.MAX_WORDS):
        expanded = []
        for hyp in beam:
            if hyp["done"]:
                continue
            if step + 1 < base.MIN_WORDS:
                boundaries = (base.SPACE,)
            elif step + 1 < base.MAX_WORDS:
                boundaries = (base.SPACE, base.COMMA, *base.ENDS)
            else:
                boundaries = base.ENDS
            choices = suggest._word_choices(
                scorer, hyp["state"], vocabulary.root, hyp["history"],
                beam_width=base.BYTE_BEAM, max_word_bytes=32, max_loop=1,
                limit=64, continue_after_space=step + 1 < base.MAX_WORDS,
                boundaries=boundaries,
            )
            pairs = set(zip(hyp["history"], hyp["history"][1:]))
            prev = hyp["history"][-1] if hyp["history"] else None
            picks = []
            for choice in choices:
                word = choice.word.casefold()
                if prev is not None and (word == prev or (prev, word) in pairs):
                    continue
                if choice.boundary not in (base.SPACE, base.COMMA) and choice.boundary not in base.ENDS:
                    continue
                picks.append(choice)
                if len(picks) >= base.TOP:
                    break
            if hyp["words"] >= base.MIN_WORDS and any(choice.boundary in base.ENDS for choice in picks):
                picks = [choice for choice in picks if choice.boundary in base.ENDS]
            for choice in picks:
                raw = hyp["emitted"] + choice.suffix + bytes((choice.boundary,))
                words = hyp["words"] + 1
                ended = choice.boundary in base.ENDS
                if choice.boundary == base.SPACE:
                    nxt_state = choice.state
                elif choice.boundary == base.COMMA and not ended:
                    nxt_state = scorer.advance(choice.state, choice.boundary)
                else:
                    nxt_state = choice.state
                nxt = {
                    "state": nxt_state,
                    "emitted": raw,
                    "history": hyp["history"] + (choice.word.casefold(),),
                    "lp": hyp["lp"] + choice.logp,
                    "words": words,
                    "done": ended or words >= base.MAX_WORDS,
                }
                if words >= base.MIN_WORDS and ended:
                    finished.append(nxt)
                if not nxt["done"]:
                    expanded.append(nxt)
        expanded.sort(key=lambda hyp: (-hyp["lp"], hyp["emitted"]))
        beam = expanded[:base.PHRASE_BEAM]
        if not beam:
            break
    cands = finished or [hyp for hyp in beam if hyp["words"]]
    if not cands:
        return {"continuation": "", "ids": [], "candidates": 0, "seconds": time.perf_counter() - t0}
    prompt_bytes = prompt.encode("utf-8")
    last = prefix.encode("utf-8") if prefix else prompt_bytes
    for hyp in cands:
        continuation = hyp["emitted"]
        hyp["lp"] = base.lp_after(scorer, suggest._log_probs, prompt_bytes, continuation)
        hyp["lpg"] = base.lp_after(scorer, suggest._log_probs, last, continuation)
        hyp["n"] = max(1, len(continuation))
    best = min(cands, key=lambda hyp: -(hyp["lp"] - base.LAM * hyp["lpg"]) / hyp["n"])
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
    model_sha = base.sha256(base.MODEL)
    if model_sha != base.EXPECT_MODEL:
        raise SystemExit(f"model sha {model_sha} != {base.EXPECT_MODEL}")
    info = json.loads((base.RUN / "run_info.json").read_text(encoding="utf-8"))
    for key in list(os.environ):
        if key.startswith("MICA_"):
            os.environ.pop(key)
    for key, value in info["env"].items():
        os.environ[key] = str(value)
    os.environ.pop("MICA_SYMBOLS", None)
    sys.path.insert(0, str(base.R1))
    from mica_r1 import serialize, spec
    from mica_r1 import suggest

    if spec.N_SYMBOLS != 258 or spec.BOS != 256 or spec.EOS != 257:
        raise SystemExit(f"unexpected letter geometry {spec.N_SYMBOLS}")
    model = serialize.load(str(base.MODEL))
    scorer = suggest.MicaScorer(model)
    tokens, surface = base.load_words()
    words = []
    for token, shown in zip(tokens, surface):
        form = shown if suggest.WORD_RE.fullmatch(shown or "") else token
        if suggest.WORD_RE.fullmatch(form or ""):
            words.append(form)
    vocabulary = suggest.Vocabulary(words)
    rows = [json.loads(line) for line in base.PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for index, row in enumerate(rows[:limit]) if index % nshard == shard]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"letter.shard{shard}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    fresh = [row for row in rows if row["id"] not in done]
    print(f"w2 shard {shard}/{nshard} vocab {vocabulary.size} new {len(fresh)}", flush=True)
    t0 = time.perf_counter()
    with out.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(fresh, start=1):
            result = search_end(scorer, vocabulary, suggest, item["prompt"])
            row = {
                "system": "letter-end",
                "decoding": DECODER,
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
                print(f"w2 {shard} {index}/{len(fresh)} elapsed {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"w2 wrote shard {shard} n {len(done) + len(fresh)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
