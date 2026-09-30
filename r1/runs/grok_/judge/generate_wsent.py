#!/usr/bin/env python3
"""Deterministic Flame-W sentences with the named w-sent-mmi.3 decoder.

Imports Claude's word_decode.py read-only. Writes raw token ids. Prints counts
only, never continuation text.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # mica/
R1 = ROOT / "r1"
JUDGE = Path(__file__).resolve().parent
CLAUDE = R1 / "runs" / "claude_flame_word_20260928"
PKG = R1 / "runs" / "codex_flame_word_full40_20260928"
MODEL = PKG / "train" / "best.mica"
VOCAB = R1 / "data" / "word" / "vocab.json"
ZIP = Path(r"C:\Users\vlavrik\PycharmProjects\mica_ember_v0.1\mica-complete-v0.1-dev.zip")
MEMBER = "mica-complete-v0.1-dev/mica-complete-v0.1-dev.jsonl"
EXPECT_PROMPT = "773593ead8d9408dbb9c1fd3ae5782f330a0095255f547c2592cf29485ebabfd"
EXPECT_MODEL = "ef969c0e96173fc11ecc3b5cf0d1d28d04e21fe846a0752297d5549e90f7e6b7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompts() -> list[dict]:
    out = JUDGE / "prompts" / "mica-complete-v0.1-dev.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = zipfile.ZipFile(ZIP).read(MEMBER)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECT_PROMPT:
        raise SystemExit(f"prompt sha {digest} != {EXPECT_PROMPT}")
    if not out.exists() or hashlib.sha256(out.read_bytes()).hexdigest() != digest:
        out.write_bytes(raw)
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise SystemExit(f"expected 100 prompts, got {len(rows)}")
    return rows


def complete_with_ids(dec, prompt: str) -> dict:
    """word_decode.WordDecoder.complete, plus the chosen token ids."""
    import math
    import numpy as np

    t0 = time.perf_counter()
    ctx = dec_prefix(dec, prompt)
    st = dec.m.start()
    for token in ctx:
        st = dec.m.feed(st, token)
    hist0 = [token for token in ctx if dec.is_word[token]]
    beam = [{"ids": [], "lp": 0.0, "st": st, "words": 0, "done": False}]
    finished = []
    for _ in range(dec.max_words * 2):
        new = []
        for hyp in beam:
            if hyp["done"]:
                continue
            hist = hist0 + [token for token in hyp["ids"] if dec.is_word[token]]
            prev = hist[-1] if hist else None
            banned = {b for a, b in zip(hist, hist[1:]) if a == prev}
            if prev is not None:
                banned.add(prev)
            logp = dec.m.logp(hyp["st"], dec.allowed)
            order = np.argsort(-logp, kind="stable")
            picks = 0
            for key in order:
                token = int(dec.allowed[key])
                if token in banned:
                    continue
                if not dec.is_word[token] and (
                    hyp["words"] == 0 or (hyp["ids"] and not dec.is_word[hyp["ids"][-1]])
                ):
                    continue
                nxt = {
                    "ids": hyp["ids"] + [token],
                    "lp": hyp["lp"] + float(logp[key]),
                    "words": hyp["words"] + int(dec.is_word[token]),
                    "done": token in dec.ends,
                }
                nxt["done"] = nxt["done"] or nxt["words"] >= dec.max_words
                if not nxt["done"]:
                    nxt["st"] = dec.m.feed(dec.m.fork(hyp["st"]), token)
                new.append(nxt)
                if nxt["words"] >= dec.min_words and (not dec.must_end or token in dec.ends):
                    finished.append(nxt)
                picks += 1
                if picks >= dec.top:
                    break
        beam = sorted(new, key=lambda hyp: -hyp["lp"])[: dec.beam]
        if not any(not hyp["done"] for hyp in beam):
            break
    cands = finished or [hyp for hyp in beam if hyp["ids"]]
    if not cands:
        return {
            "continuation": "",
            "ids": [],
            "bits_per_token": None,
            "candidates": 0,
            "seconds": time.perf_counter() - t0,
        }
    if dec.lam:
        generic = ctx[-1:] if ctx else []
        for hyp in cands:
            hyp["lpg"] = dec._lp_after(generic, hyp["ids"])
        rank = lambda hyp: -(hyp["lp"] - dec.lam * hyp["lpg"]) / len(hyp["ids"])
    else:
        rank = lambda hyp: -hyp["lp"] / len(hyp["ids"])
    best = min(cands, key=rank)
    ids = list(best["ids"])
    while ids and ids[-1] == dec.comma:
        ids = ids[:-1]
    cap = not prompt.strip() or prompt.rstrip()[-1:] in ".!?\n"
    bits = None if not ids else -best["lp"] / len(best["ids"]) / math.log(2)
    return {
        "continuation": dec.text(ids, cap),
        "ids": ids,
        "bits_per_token": bits,
        "candidates": len(cands),
        "seconds": time.perf_counter() - t0,
    }


def dec_prefix(dec, prompt: str):
    from word_eval import prefix_ids
    return prefix_ids(dec.v, prompt.encode("utf-8"))


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    model_sha = sha256(MODEL)
    if model_sha != EXPECT_MODEL:
        raise SystemExit(f"model sha {model_sha} != {EXPECT_MODEL}")
    sys.path.insert(0, str(CLAUDE))
    import word_decode as D
    import word_eval as E

    rows = prompts()[:limit]
    vocab = E.W.Vocab.load(VOCAB)
    model = E.load_model(f"mica:{PKG}")
    flags = dict(D.MODES["sentence"])
    dec = D.WordDecoder(model, vocab, **flags)
    if dec.name != "w-sent-mmi.3":
        raise SystemExit(f"unexpected decoder {dec.name}")
    out = JUDGE / "gen" / "flamew.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    fresh = [row for row in rows if row["id"] not in done]
    t0 = time.perf_counter()
    with out.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(fresh, start=1):
            result = complete_with_ids(dec, item["prompt"])
            row = {
                "system": "flamew",
                "decoding": dec.name,
                "model_sha256": model_sha,
                "engine": "integer-word-mica",
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
            if index % 5 == 0 or index == len(fresh):
                print(
                    f"flamew {index}/{len(fresh)} elapsed {time.perf_counter() - t0:.0f}s",
                    flush=True,
                )
    print(
        f"flamew wrote {out} prompts {len(rows)} new {len(fresh)} "
        f"sha {model_sha[:12]} decoder {dec.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
