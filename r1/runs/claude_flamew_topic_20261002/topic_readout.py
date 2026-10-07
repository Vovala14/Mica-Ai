#!/usr/bin/env python3
"""Readout-only topic bias for frozen Flame-W B-740.

The automaton file is not modified and phases 8–15 are not rewired. Each
word gets an int8 weight on the simulated topic register (the third probe:
neither a tape code nor a rule immediate). The table lives in a sidecar
npz. Optional: add it on top of the 0.3.1 memory readout.

    # sha256 of the frozen checkpoints (does not score text)
    python topic_readout.py hash

    # fit on a training jsonl of word-id records. Never pass Tiny ToM.
    python topic_readout.py fit --train TRAIN.jsonl --codes CODES.npz --out bias.npz

    # bits on a held-out jsonl you built. This is not the published val500.
    python topic_readout.py bits --tokens HELD.jsonl --bias bias.npz --out bits.json

    # Tiny Theory of Mind, pinned Hugging Face revision
    python topic_readout.py tom --revision ba4c644ea7ace67e096305fe2c9f87c4cf4879ad \
        --bias bias.npz --out tom.json

val500 is not in this repo. Do not read a bits number from this script as
the published 5.208 comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORD_EVAL = ROOT / "r1/runs/claude_flame_word_20260928"
B740 = ROOT / "flame/runs/claude_flamew_b740_20261001/train"
MEM = ROOT / "flame/runs/claude_flamew_031_20261004"
VOCAB = ROOT / "r1/data/word/vocab.json"
TOM_REPO = "AxiomicLabs/Tiny_Theory_of_Mind"
TOM_FILE = "tiny_theory_of_mind_2000.jsonl"
TOM_REVISION = "ba4c644ea7ace67e096305fe2c9f87c4cf4879ad"

# Published 0.3.1 bars, for the report only. This script does not claim them.
PUBLISHED = {"val500_bits": 5.208, "tom_word": 0.3125, "tom_char": 0.3145,
             "use_w8_chat": 0.0183, "use_w8_everyday": 0.0483}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_hash(_a) -> None:
    files = {
        "b740_mica": B740 / "best.mica",
        "memory_031": MEM / "memory.npz",
        "answer_mode": MEM / "answer_mode.npz",
        "vocab": VOCAB,
    }
    out = {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                  "bytes": path.stat().st_size} for name, path in files.items()}
    out["published_not_measured_here"] = PUBLISHED
    out["note"] = ("best.mica is not rewritten by the topic readout. "
                   "val500 is not in the repo; do not compare to 5.208 here.")
    print(json.dumps(out, indent=1))


def _load_stack(bias: str | None, memory: bool, answer: bool):
    sys.path.insert(0, str(WORD_EVAL))
    sys.path.insert(0, str(MEM))
    import word_eval as E
    from memory import Memory, MemoryMicaWord
    model = E.load_model(f"mica:{B740}")
    if bias:
        from mica_r1 import engine
        engine.load_topic_readout(bias, model.model)
    if memory:
        model = MemoryMicaWord(model, Memory(MEM / "memory.npz",
                                             MEM / "answer_mode.npz" if answer else None))
    return model


def _refuse_tom(path: Path) -> None:
    name = path.name.lower()
    if "theory_of_mind" in name or "tiny_tom" in name or name.startswith("tom"):
        raise SystemExit(f"refusing to fit on {path}: Tiny ToM is an eval set, not training text")


def cmd_fit(a) -> None:
    train = Path(a.train)
    _refuse_tom(train)
    if not train.is_file():
        raise SystemExit(f"missing training records: {train}. "
                         "The no-TinyStories word corpus is not in this checkout, "
                         "so a real fit cannot be run until you pass one.")
    sys.path.insert(0, str(WORD_EVAL))
    import word_eval as E
    from mica_r1 import engine, fit
    from word_eval import W
    codes = np.load(a.codes, allow_pickle=False)["codes"]
    if codes.dtype != np.int8 or codes.shape != (W.N_SYMBOLS, a.channels):
        raise SystemExit(f"codes must be int8 {(W.N_SYMBOLS, a.channels)}")
    model = E.load_model(f"mica:{B740}")
    if a.memory:
        sys.path.insert(0, str(MEM))
        from memory import Memory, MemoryMicaWord
        model = MemoryMicaWord(model, Memory(MEM / "memory.npz"))
    records = [W.unpack(line) for line in open(train, encoding="ascii") if line.strip()]
    if a.limit:
        records = records[:a.limit]
    registers = engine.topic_registers(records, codes, a.shift)
    # Frozen scores, one row per target (record symbols, then EOS).
    rows_base, rows_tgt = [], []
    t0 = time.time()
    for rec in records:
        st = model.start()
        for sym in list(rec) + [W.EOS]:
            rows_base.append(model.scores(st).astype(np.int64))
            rows_tgt.append(int(sym))
            if sym != W.EOS:
                st = model.feed(st, int(sym))
    print(f"[topic] froze {len(rows_tgt)} scores in {time.time() - t0:.0f}s", flush=True)
    base = np.stack(rows_base)
    elig = model.elig if hasattr(model, "elig") else model.base.elig
    out = fit.fit_topic_bias(registers, base, np.asarray(rows_tgt, np.int64),
                             steps=a.steps, lr=a.lr, l2=a.l2, divisor=1024,
                             eligible=elig, seed=0)
    host = model.model if hasattr(model, "model") else model.base.model
    engine.bind_topic_readout(host, codes, out["W"], shift=a.shift)
    engine.save_topic_readout(a.out, host)
    # the automaton file is a separate object; confirm we did not write it
    mica = B740 / "best.mica"
    print(json.dumps({"out": a.out, "heldout_bits_start": out["heldout_bits_start"],
                      "heldout_bits_best": out["heldout_bits_best"],
                      "b740_sha256": sha256(mica),
                      "train_sha256": sha256(train),
                      "positions": int(registers.shape[0]),
                      "note": "held-out here is a split of --train, not val500"},
                     indent=1))


def cmd_bits(a) -> None:
    tokens = Path(a.tokens)
    if not tokens.is_file():
        raise SystemExit(f"missing {tokens}. This is not the published val500 set.")
    model = _load_stack(a.bias, a.memory, answer=False)
    sys.path.insert(0, str(WORD_EVAL))
    import word_eval as E
    records = E.token_records(str(tokens))
    total = 0.0
    n = 0
    for rec in records:
        b = E.record_bits(model, rec)
        total += float(b.sum())
        n += len(b)
    bits = total / n if n else float("nan")
    payload = {"bits_per_token": bits, "tokens": n,
               "tokens_sha256": sha256(tokens),
               "set": "caller-supplied held-out, not published val500",
               "published_val500_not_measured": PUBLISHED["val500_bits"],
               "bias": a.bias or None, "memory": bool(a.memory)}
    Path(a.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload, indent=1))


def _tom_rows(path: Path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def cmd_tom(a) -> None:
    data = Path(a.data) if a.data else None
    if data is None:
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(TOM_REPO, TOM_FILE, repo_type="dataset", revision=a.revision)
        data = Path(got)
    rows = _tom_rows(data)
    if a.limit:
        rows = rows[:a.limit]
    digest = sha256(data)
    sys.path.insert(0, str(WORD_EVAL))
    import word_eval as E
    from word_eval import W
    model = _load_stack(a.bias, memory=True, answer=not a.no_answer)
    vocab = W.Vocab.load(VOCAB)

    def ids_of(text: str):
        return [W.SEP if t == "<sep>" else vocab.index.get(t, W.oov_id(t))
                for t in W.tokenize(text)]

    ok_c = ok_w = 0
    out = []
    t0 = time.time()
    for k, r in enumerate(rows):
        st = model.start()
        for t in ids_of(r["ctx"]):
            st = model.feed(st, t)
        ends = []
        for e in r["endings"]:
            ei = ids_of(e)
            s2, lp = model.fork(st), 0.0
            for t in ei:
                lp += float(model.logp(s2, [t])[0])
                s2 = model.feed(s2, t)
            ends.append({"sum_lnp": lp, "tokens": len(ei), "chars": len(e)})
        by_c = max(range(4), key=lambda i: ends[i]["sum_lnp"] / max(ends[i]["chars"], 1))
        by_w = max(range(4), key=lambda i: ends[i]["sum_lnp"] / ends[i]["tokens"]
                   if ends[i]["tokens"] else -math.inf)
        lab = int(r["label"])
        ok_c += by_c == lab
        ok_w += by_w == lab
        out.append({"ind": r["ind"], "label": lab, "pred_char": by_c, "pred_word": by_w,
                    "topic": r.get("metadata", {}).get("topic")})
        if (k + 1) % 25 == 0:
            print(f"{k + 1}/{len(rows)}  acc_norm {ok_c / (k + 1):.4f}  "
                  f"acc_word {ok_w / (k + 1):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    n = len(rows)
    payload = {"model": "B-740 + 0.3.1 memory + topic readout" if a.bias
               else "B-740 + 0.3.1 memory",
               "answer_mode": not a.no_answer,
               "n": n,
               "acc_norm": ok_c / n, "acc_word": ok_w / n,
               "hf_repo": TOM_REPO, "hf_file": TOM_FILE, "hf_revision": a.revision,
               "data_sha256": digest, "data_path": str(data),
               "bias": a.bias,
               "published_0.3.1": {"acc_word": PUBLISHED["tom_word"],
                                   "acc_norm": PUBLISHED["tom_char"]},
               "note": "Trained weights are required before this number is a topic result. "
                       "A missing --bias scores 0.3.1 only. Tiny ToM was not a fit set.",
               "rows": out}
    Path(a.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"acc_norm {ok_c / n:.4f}  acc_word {ok_w / n:.4f}  n={n}  sha256 {digest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hash").set_defaults(fn=cmd_hash)
    fit = sub.add_parser("fit")
    fit.add_argument("--train", required=True)
    fit.add_argument("--codes", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--channels", type=int, default=8)
    fit.add_argument("--shift", type=int, default=3)
    fit.add_argument("--steps", type=int, default=400)
    fit.add_argument("--lr", type=float, default=0.05)
    fit.add_argument("--l2", type=float, default=1e-4)
    fit.add_argument("--limit", type=int, default=0)
    fit.add_argument("--memory", action="store_true")
    fit.set_defaults(fn=cmd_fit)
    bits = sub.add_parser("bits")
    bits.add_argument("--tokens", required=True)
    bits.add_argument("--bias", default="")
    bits.add_argument("--out", required=True)
    bits.add_argument("--memory", action="store_true", default=True)
    bits.add_argument("--no-memory", action="store_false", dest="memory")
    bits.set_defaults(fn=cmd_bits)
    tom = sub.add_parser("tom")
    tom.add_argument("--revision", default=TOM_REVISION)
    tom.add_argument("--data", default="")
    tom.add_argument("--bias", default="")
    tom.add_argument("--out", default="tom_topic.json")
    tom.add_argument("--limit", type=int, default=0)
    tom.add_argument("--no-answer", action="store_true")
    tom.set_defaults(fn=cmd_tom)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
