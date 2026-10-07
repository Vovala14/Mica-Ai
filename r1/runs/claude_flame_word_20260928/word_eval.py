#!/usr/bin/env python3
"""Word-level metrics for Flame-W (word-symbol MICA) and word references.

    python word_eval.py nextword --model SPEC --vocab vocab.json --set BYTES.jsonl --out F.json
    python word_eval.py complete --model SPEC --vocab vocab.json --set BYTES.jsonl --out F.json
    python word_eval.py bits     --model SPEC --vocab vocab.json --tokens TOK.jsonl --out F.json
    python word_eval.py ctxuse   --model SPEC --vocab vocab.json --tokens TOK.jsonl --out F.json

SPEC is "kenlm:MODEL.binary" (a KenLM model over "w<id>" words) or
"mica:PACKAGE_DIR" (package.json names the code directory, the model file and
the geometry environment, including MICA_SYMBOLS).

M1 nextword  The letter models' next-word metric at the same positions
             (baselines/common.nextword_positions on the set's byte records:
             2,000 word starts after whitespace, seed 0). The word model reads
             the record up to the position, tokenized as in training, and ranks
             vocabulary words ([a-z][a-z']*); lower-case exact match.
M2 complete  Same positions with words of at least 4 letters; the first two
             letters are typed, and the ranking is restricted to vocabulary
             words that start with them. top-1, top-3 and the share of letters
             saved, as wordcomplete_bytes.py.
M3 ctxuse    use_w(k): the first SPLICE tokens of each record are replaced by
             another record's; mean over targets at p >= SPLICE + k of
             bits(spliced) - bits(original). Donors and records as
             context_use.py (records of >= 2 * SPLICE tokens, seed 20260928).
M4 bits      bits per token, tokens plus one EOS per record, from BOS.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
# build_word_corpus.py lives in r1/data: next to this folder's r1/ in the cloud
# layout, or two levels up when this file sits in r1/runs/<run>/ on the PC
for _d in (HERE.parent / "r1" / "data", HERE.parents[1] / "data", HERE.parents[2] / "data"):
    if (_d / "build_word_corpus.py").exists():
        sys.path.insert(0, str(_d))
        break
import build_word_corpus as W  # noqa: E402

LN2 = math.log(2)
SPLICE = 6
WORDLIKE = re.compile(r"[a-z][a-z']*")


# ---------------------------------------------------------------- models
class KenlmWord:
    """KenLM trained on lines of "w<id>" words (one record per line)."""

    def __init__(self, path: str):
        import kenlm
        self.kenlm = kenlm
        self.m = kenlm.Model(path)
        self.names = [f"w{i}" for i in range(W.N_SYMBOLS)]
        self.info = {"kind": "kenlm-word", "file": Path(path).name,
                     "file_bytes": Path(path).stat().st_size, "order": self.m.order}

    def start(self):
        s = self.kenlm.State()
        self.m.BeginSentenceWrite(s)
        return s

    def feed(self, st, i: int):
        out = self.kenlm.State()
        self.m.BaseScore(st, self.names[i], out)
        return out

    def fork(self, st):
        return st                      # KenLM states are values

    def logp(self, st, ids) -> np.ndarray:
        """Natural log-probabilities of the given ids (EOS = end of record)."""
        out = self.kenlm.State()
        f = self.m.BaseScore
        return np.array([f(st, "</s>" if i == W.EOS else self.names[i], out) for i in ids]
                        ) * math.log(10)


class MicaWord:
    """The exact integer engine at N_SYMBOLS word symbols."""

    def __init__(self, pkg: str):
        cfg = json.loads((Path(pkg) / "package.json").read_text())
        for k, v in cfg["geometry_env"].items():
            os.environ[k] = str(v)
        sys.path.insert(0, str(Path(pkg) / cfg.get("code_dir", ".")))
        from mica_r1 import engine, serialize, spec
        assert spec.N_SYMBOLS == W.N_SYMBOLS and spec.BOS == W.BOS and spec.EOS == W.EOS
        self.eng, self.spec = engine, spec
        self.model = serialize.load(str(Path(pkg) / cfg["model"]))
        self.div = float(spec.LOGIT_DIVISOR)
        self.elig = np.zeros(W.N_SYMBOLS, bool)
        self.elig[:W.BOS] = True
        self.elig[W.EOS] = True
        m = self.model
        self.shared = bool((m.pr_cell == m.pr_cell[:1]).all() and
                           (m.pr_chan == m.pr_chan[:1]).all())
        self.co = m.pr_co.astype(np.int32)
        self.bias = m.pr_bias.astype(np.int32)
        self.info = {"kind": "mica-word", "model": cfg["model"],
                     "model_sha256": cfg.get("model_sha256"), "shared_probe_wiring": self.shared}

    def start(self):
        return self.eng.new_session(self.model)

    def feed(self, s, i: int):
        self.eng.ingest(self.model, s, int(i))
        return s

    def fork(self, s):
        t = self.eng.Session()
        t.restore(s.snapshot())
        return t

    def scores(self, s) -> np.ndarray:
        m = self.model
        if not self.shared:
            return self.eng.probe_scores(m, s)
        cells = m.pr_cell[0].astype(np.int32)
        if self.spec.ROLLING_READOUT:
            cells = (cells + s.position) % self.spec.N_CELLS
        f = s.F[cells, m.pr_chan[0].astype(np.int32)].astype(np.int32)
        base = self.bias + self.co @ f
        # The fast path does not call probe_scores. Add the same sidecar the
        # engine adds there. None when no topic readout is bound.
        bonus = self.eng.topic_bonus(m, s)
        return base if bonus is None else base + bonus

    def logp(self, s, ids) -> np.ndarray:
        z = self.scores(s).astype(np.float64) / self.div
        z = np.where(self.elig, z, -np.inf)
        z -= z.max()
        lse = math.log(np.exp(z[self.elig]).sum())
        return z[np.asarray(ids)] - lse


def load_model(spec: str):
    kind, _, path = spec.partition(":")
    return KenlmWord(path) if kind == "kenlm" else MicaWord(path)


# ---------------------------------------------------------------- data
def byte_records(path: str) -> list[bytes]:
    return [bytes.fromhex(l.strip()) for l in open(path) if l.strip()]


def token_records(path: str) -> list[list[int]]:
    return [W.unpack(l) for l in open(path) if l.strip()]


_WORD_START = re.compile(rb"(?<=[ \t\n])[A-Za-z][A-Za-z']*")


def letter_positions(path: str, n: int = 2000, seed: int = 0):
    """Exactly the letter models' positions (baselines/common.nextword_positions):
    word starts after whitespace, not cut by the record's end; a fixed sample
    of n, sorted. Returns [(record, byte offset, lower-case word)], records."""
    recs = byte_records(path)
    cand = [(i, m.start(), m.group().decode().lower())
            for i, r in enumerate(recs) for m in _WORD_START.finditer(r) if m.end() < len(r)]
    pick = sorted(random.Random(seed).sample(range(len(cand)), n))
    return [cand[k] for k in pick], recs


def word_ids(vocab: W.Vocab) -> np.ndarray:
    return np.array([W.FIRST_WORD + k for k, t in enumerate(vocab.tokens)
                     if WORDLIKE.fullmatch(t)], np.int64)


def prefix_ids(vocab: W.Vocab, prefix: bytes) -> list[int]:
    """The record up to a position, as the model saw it in training (no cut:
    a prefix longer than MAX_TOKENS keeps every token)."""
    toks = W.tokenize(W.record_text(prefix))
    return [W.SEP if t == "<sep>" else vocab.index.get(t, W.oov_id(t)) for t in toks]


def run_prefixes(model, vocab, path, query):
    """Feed each record's token prefix and call query(state, i, off, w) at every
    letter position; prefixes are re-tokenized from the bytes (a word model
    sees what training showed it)."""
    pos, recs = letter_positions(path)
    out = []
    by_rec = {}
    for i, off, w in pos:
        by_rec.setdefault(i, []).append((off, w))
    t0 = time.time()
    for n, i in enumerate(sorted(by_rec)):
        for off, w in sorted(by_rec[i]):
            ids = prefix_ids(vocab, recs[i][:off])
            st = model.start()
            for t in ids:
                st = model.feed(st, t)
            out.append(query(st, i, off, w))
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(by_rec)} records ({time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------- metrics
def nextword(model, vocab, path, k=10):
    cand = word_ids(vocab)
    toks = vocab.tokens

    def q(st, i, off, w):
        lp = model.logp(st, cand)
        top = cand[np.argsort(-lp, kind="stable")[:k]]
        return w, [toks[j - W.FIRST_WORD] for j in top]
    res = run_prefixes(model, vocab, path, q)
    n = len(res)
    sc = {f"top{j}": sum(w in p[:j] for w, p in res) / n for j in (1, 3, 10)}
    oov = sum(w not in vocab.index for w, _ in res) / n
    return {"metric": "M1 next word", "positions": n, **sc, "target_oov_share": oov,
            "examples": [{"target": w, "pred": p[:3]} for w, p in res[:40]]}


def complete(model, vocab, path, prefix_len=2):
    cand = word_ids(vocab)
    toks = vocab.tokens
    by_prefix = {}
    for j in cand:
        t = toks[j - W.FIRST_WORD]
        by_prefix.setdefault(t[:prefix_len], []).append(j)
    by_prefix = {p: np.array(v) for p, v in by_prefix.items()}

    def q(st, i, off, w):
        if len(w) < prefix_len + 2:
            return None
        ids = by_prefix.get(w[:prefix_len], np.zeros(0, np.int64))
        if not len(ids):
            return w, []
        lp = model.logp(st, ids)
        return w, [toks[j - W.FIRST_WORD] for j in ids[np.argsort(-lp, kind="stable")[:3]]]
    res = [r for r in run_prefixes(model, vocab, path, q) if r is not None]
    n = len(res)
    top1 = sum(bool(p) and p[0] == w for w, p in res)
    saved = sum(len(w) - prefix_len for w, p in res if p and p[0] == w)
    letters = sum(len(w) for w, _ in res)
    return {"metric": "M2 completion", "prefix_letters": prefix_len, "positions": n,
            "top1": top1 / n, "top3": sum(w in p[:3] for w, p in res) / n,
            "letters_saved_share": saved / letters}


def record_bits(model, ids) -> np.ndarray:
    """bits of every target (tokens, then EOS) of one record, from BOS."""
    st = model.start()
    out = []
    for t in list(ids) + [W.EOS]:
        out.append(-float(model.logp(st, [t])[0]) / LN2)
        if t != W.EOS:
            st = model.feed(st, t)
    return np.array(out)


def bits(model, path):
    recs = token_records(path)
    tot, n, per = 0.0, 0, []
    t0 = time.time()
    for k, r in enumerate(recs):
        b = record_bits(model, r)
        tot += b.sum()
        n += len(b)
        per.append(round(float(b.sum()), 6))
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(recs)} records {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return {"metric": "M4 bits per token", "records": len(recs), "targets": n,
            "bits_per_token": tot / n, "record_bits": per}


def ctxuse(model, path, ks=(0, 1, 2, 4, 8), seed=20260928):
    recs = [r for r in token_records(path) if len(r) >= 2 * SPLICE]
    rng = random.Random(seed)
    donors = list(range(len(recs)))
    rng.shuffle(donors)
    diffs = {k: [] for k in ks}
    t0 = time.time()
    for i, r in enumerate(recs):
        d = recs[donors[i]] if donors[i] != i else recs[(i + 1) % len(recs)]
        s = d[:SPLICE] + r[SPLICE:]
        br, bs = record_bits(model, r), record_bits(model, s)
        for k in ks:
            p0 = SPLICE + k
            if p0 < len(br):
                diffs[k].extend((bs[p0:] - br[p0:]).tolist())
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    return {"metric": "M3 context use", "splice_tokens": SPLICE, "records": len(recs),
            **{f"use_w({k})": float(np.mean(v)) if v else None for k, v in diffs.items()},
            **{f"positions({k})": len(v) for k, v in diffs.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("metric", choices=["nextword", "complete", "bits", "ctxuse"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--set", help="byte records (nextword, complete)")
    ap.add_argument("--tokens", help="token records (bits, ctxuse)")
    ap.add_argument("--system", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vocab = W.Vocab.load(a.vocab)
    model = load_model(a.model)
    t0 = time.time()
    if a.metric == "nextword":
        res = nextword(model, vocab, a.set)
    elif a.metric == "complete":
        res = complete(model, vocab, a.set)
    elif a.metric == "bits":
        res = bits(model, a.tokens)
    else:
        res = ctxuse(model, a.tokens)
    res = {"system": a.system, **model.info, "set": Path(a.set or a.tokens).name, **res,
           "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("examples", "record_bits")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
