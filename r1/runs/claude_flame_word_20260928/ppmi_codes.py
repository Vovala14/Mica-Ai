#!/usr/bin/env python3
"""Optional Flame-W tape-code start: 16-dimensional word codes from word
co-occurrence on the mix A TRAINING tokens only (PPMI, then truncated SVD),
scaled to the integer range tape codes use.

    python ppmi_codes.py WORD_DIR OUT.npz

Counts pairs of tokens 1..WINDOW apart inside a record (both orders), forms
positive PMI with context-distribution smoothing (alpha 0.75), and keeps the
top 16 singular directions, U * sqrt(S). Each channel is scaled so its 99.9th
percentile magnitude is 64 (clipped to 127), then rounded. Ids that never
occur (BOS, EOS) get random +-24..64 codes as in tape_init_ext. The pairdiff
routing compares channels 0-4 with 5-9, so the first ten channels are the ten
strongest directions, and channels 0-4 are shifted so that every routing bit
splits the training tokens (by frequency) about 50/50.
"""
from __future__ import annotations

import gzip
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

V = 16384
K = 16
WINDOW = 2


def shard_arrays(path: Path):
    data = gzip.open(path, "rb").read()
    ids, lens = [], []
    i = 0
    while i < len(data):
        n = struct.unpack_from("<H", data, i)[0]
        ids.append(np.frombuffer(data, "<u2", n // 2, i + 2))
        lens.append(n // 2)
        i += 2 + n
    ids = np.concatenate(ids).astype(np.int64)
    rec = np.repeat(np.arange(len(lens)), lens)
    return ids, rec


def main() -> int:
    word_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    t0 = time.time()
    C = sp.csr_matrix((V, V), dtype=np.float64)
    freq = np.zeros(V, np.float64)
    n_tok = 0
    for p in sorted((word_dir / "v02a").glob("train.part*.u16.gz")):
        ids, rec = shard_arrays(p)
        n_tok += len(ids)
        freq += np.bincount(ids, minlength=V)
        for d in range(1, WINDOW + 1):
            ok = rec[:-d] == rec[d:]
            a, b = ids[:-d][ok], ids[d:][ok]
            key, cnt = np.unique(a * V + b, return_counts=True)
            M = sp.csr_matrix((cnt.astype(np.float64), (key // V, key % V)), shape=(V, V))
            C = C + M + M.T
        print(f"  {p.name}: {len(ids):,} tokens, nnz {C.nnz:,} ({time.time() - t0:.0f}s)", flush=True)
    C = C.tocsr()
    total = C.sum()
    row = np.asarray(C.sum(1)).ravel()
    col = np.asarray(C.sum(0)).ravel() ** 0.75
    col = col / col.sum()
    C = C.tocoo()
    pmi = np.log(C.data / total) - np.log(row[C.row] / total) - np.log(col[C.col])
    keep = pmi > 0
    P = sp.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=(V, V))
    print(f"  PPMI nnz {P.nnz:,} ({time.time() - t0:.0f}s)", flush=True)
    U, S, _ = svds(P, k=K, random_state=0)
    order = np.argsort(-S)
    E = U[:, order] * np.sqrt(S[order])
    seen = freq > 0
    scale = 64.0 / np.percentile(np.abs(E[seen]), 99.9, axis=0)
    codes = np.clip(np.round(E * scale), -127, 127)
    g = np.random.default_rng(1)
    unseen = ~seen
    mag = 24 + 40 * g.random((unseen.sum(), K))
    codes[unseen] = np.round(np.where(g.random((unseen.sum(), K)) < 0.5, -1, 1) * mag)
    # pairdiff routing compares channel c with c+5 (c = 0..4): shift channels
    # 0-4 so each routing bit splits the training tokens about 50/50
    w = freq / freq.sum()
    codes = codes.astype(np.int32)
    for c in range(5):
        d = codes[:, c] - codes[:, c + 5]
        o = np.argsort(d, kind="stable")
        m = d[o][np.searchsorted(np.cumsum(w[o]), 0.5)]
        codes[:, c] = np.clip(codes[:, c] - m, -127, 127)
    codes = codes.astype(np.int8)
    c32 = codes.astype(np.int32)
    bal = [float((w * (c32[:, c] >= c32[:, c + 5])).sum()) for c in range(5)]
    cls = sum((c32[:, c] >= c32[:, c + 5]).astype(np.int64) << c for c in range(5))
    pc = np.bincount(cls, weights=w, minlength=32)
    pc = pc[pc > 0]
    entropy = float(-(pc * np.log2(pc)).sum())
    np.savez(out, codes=codes, singular_values=S[order], freq=freq)
    meta = {"tokens": int(n_tok), "window": WINDOW, "k": K, "ppmi_nnz": int(P.nnz),
            "unseen_ids": int(unseen.sum()), "routing_bit_balance": bal,
            "routing_class_entropy_bits": entropy,
            "singular_values": [float(x) for x in S[order]], "seconds": round(time.time() - t0, 1)}
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
