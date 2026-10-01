#!/usr/bin/env python3
"""Topic codes for the Flame-W topic register (spec.TOPIC_CHANNELS).

    python topic_codes.py TRAIN.jsonl VOCAB.json OUT.npz [--channels 8]

Each content word gets T small integers that say what it is about; function
words get zeros. The register then holds a fading sum of the recent content
words' codes, so nearby words on the same subject ("museum", "paintings",
"exhibit") push it the same way.

How the codes are made, from the TRAINING records only (numpy, CPU, a few
minutes):

  1. Content words: the --content most frequent alphabetic vocabulary words
     after skipping the --skip most frequent ones (the, a, I, you, ... carry
     no topic). Context words: the --contexts most frequent of those.
  2. Count how often each content word appears within --window words of each
     context word in the same record, on up to --records records.
  3. Positive PMI with context smoothing 0.75, then the top T singular
     directions (U * sqrt(S)) of the content x context matrix.
  4. Scale: run the register's own integer decay over sample records and
     choose one scale so the 99th percentile of |register| is --target,
     leaving room below saturation (127). Round to int8.

Writes OUT.npz with codes (N_SYMBOLS, T) int8 and a summary .json next to it
listing, for each channel, the words at both ends (a sanity check you can
read: each channel should separate subjects, e.g. food vs travel).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np


def read_records(path: Path, limit: int, seed: int = 0) -> list[np.ndarray]:
    """A random sample of `limit` records, read in two passes so only the
    sample is ever held in memory."""
    with open(path, encoding="ascii") as fh:
        n = sum(1 for l in fh if l.strip())
    keep = (set(np.random.default_rng(seed).choice(n, limit, replace=False).tolist())
            if n > limit else None)
    out, i = [], 0
    with open(path, encoding="ascii") as fh:
        for line in fh:
            if not line.strip():
                continue
            if keep is None or i in keep:
                out.append(np.frombuffer(bytes.fromhex(line.strip()), "<u2").astype(np.int64))
            i += 1
    return out


def cooccurrence(records, row_of, col_of, n_rows, n_cols, window):
    """Counts of (content word, context word) pairs 1..window apart in a
    record, both orders, as a dense float64 matrix."""
    ids = np.concatenate(records)
    rec = np.repeat(np.arange(len(records)), [len(r) for r in records])
    C = np.zeros(n_rows * n_cols, np.float64)
    for d in range(1, window + 1):
        same = rec[:-d] == rec[d:]
        for a, b in ((ids[:-d][same], ids[d:][same]), (ids[d:][same], ids[:-d][same])):
            r, c = row_of[a], col_of[b]
            ok = (r >= 0) & (c >= 0)
            C += np.bincount(r[ok] * n_cols + c[ok], minlength=n_rows * n_cols)
    return C.reshape(n_rows, n_cols)


def register_values(records, codes_f, shift):
    """The register's integer recurrence, in float with the same decay, over
    every position of `records` (for choosing the scale)."""
    keep = 1.0 - 1.0 / (1 << shift)
    out = []
    for r in records:
        t = np.zeros(codes_f.shape[1])
        for w in r:
            t = t * keep + codes_f[w]
            out.append(np.abs(t).max())
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train", type=Path)
    ap.add_argument("vocab", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--shift", type=int, default=3, help="the register's decay (MICA_TOPIC_SHIFT)")
    ap.add_argument("--skip", type=int, default=150, help="most frequent words treated as function words")
    ap.add_argument("--content", type=int, default=6000)
    ap.add_argument("--contexts", type=int, default=2000)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--records", type=int, default=400_000)
    ap.add_argument("--target", type=float, default=90.0)
    a = ap.parse_args()
    t0 = time.time()
    v = json.loads(a.vocab.read_text(encoding="utf-8"))
    n_sym, first = int(v["n_symbols"]), int(v["first_word"])
    tokens, counts = v["tokens"], np.asarray(v["counts"], np.float64)
    order = np.argsort(-counts, kind="stable")
    alpha = [i for i in order if re.fullmatch(r"[a-z][a-z']*", tokens[i])]
    content = alpha[a.skip:a.skip + a.content]
    contexts = content[:a.contexts]
    row_of = np.full(n_sym, -1, np.int64)
    col_of = np.full(n_sym, -1, np.int64)
    row_of[first + np.asarray(content)] = np.arange(len(content))
    col_of[first + np.asarray(contexts)] = np.arange(len(contexts))

    records = read_records(a.train, a.records)
    print(f"[topic] {len(records):,} records, {sum(len(r) for r in records):,} words; "
          f"{len(content)} content words, {len(contexts)} contexts", flush=True)
    C = cooccurrence(records, row_of, col_of, len(content), len(contexts), a.window)
    total = C.sum()
    if total == 0:
        raise SystemExit("no co-occurrences found; wrong file or vocabulary?")
    pw = C.sum(1, keepdims=True) / total
    pc = C.sum(0, keepdims=True) ** 0.75
    pc /= pc.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C / total) / (pw * pc))
    ppmi = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    vec = U[:, :a.channels] * np.sqrt(S[:a.channels])           # (content, T)
    vec -= vec.mean(0)                                          # centred: no constant push
    codes_f = np.zeros((n_sym, a.channels))
    codes_f[first + np.asarray(content)] = vec
    vals = register_values(records[:5000], codes_f, a.shift)
    scale = a.target / max(1e-9, np.percentile(vals, 99))
    codes = np.clip(np.rint(codes_f * scale), -127, 127).astype(np.int8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, codes=codes)
    ends = {}
    for t in range(a.channels):
        o = np.argsort(vec[:, t])
        ends[str(t)] = {"low": [tokens[content[i]] for i in o[:12]],
                        "high": [tokens[content[i]] for i in o[-12:][::-1]]}
    info = {"symbols": n_sym, "channels": a.channels, "shift": a.shift,
            "content_words": len(content), "skip": a.skip, "window": a.window,
            "records": len(records), "scale": scale,
            "code_abs_mean_content": float(np.abs(codes[first + np.asarray(content)]).mean()),
            "register_p99_after_scaling": a.target, "channel_ends": ends,
            "seconds": round(time.time() - t0)}
    a.out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"[topic] wrote {a.out} in {info['seconds']} s; mean |code| of content words "
          f"{info['code_abs_mean_content']:.1f}", flush=True)
    for t in range(min(3, a.channels)):
        print(f"[topic]   channel {t}: {' '.join(ends[str(t)]['low'][:6])}  <->  "
              f"{' '.join(ends[str(t)]['high'][:6])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
