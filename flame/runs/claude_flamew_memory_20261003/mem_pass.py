#!/usr/bin/env python3
"""Exact-engine pass that stores, per scored position, the 240 probe reads f (int8) of
Flame-W's shared-wiring readout, so its integer scores bias + co @ f can be recomputed
exactly anywhere. Records keep their token ids, so any memory readout over the history can
be fitted afterwards.

    python mem_pass.py tokens PKG TOK.jsonl[:a:b] OUT.npz [--minlen N]
    python mem_pass.py shards PKG N_RECORDS SEED OUT.npz [--minlen N]
    python mem_pass.py tom    PKG TOM.jsonl[:a:b] OUT.pkl
"""
import gzip, json, pickle, random, struct, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/claude/mica/wordw/eval")
import word_eval as E
from word_eval import W

SHARDS = Path("/mnt/user-data/uploads/PycharmProjects/mica/r1/data/word/v02a")


class Reader:
    def __init__(self, m):
        self.m = m
        mm = m.model
        assert m.shared
        self.cells = mm.pr_cell[0].astype(np.int64)
        self.chan = mm.pr_chan[0].astype(np.int64)
        self.N = m.spec.N_CELLS
        self.roll = bool(m.spec.ROLLING_READOUT)

    def f(self, s):
        c = (self.cells + s.position) % self.N if self.roll else self.cells
        return s.F[c, self.chan].astype(np.int8)


def run_record(m, R, ids):
    s = m.start()
    out = np.empty((len(ids) + 1, len(R.cells)), np.int8)
    for k, t in enumerate(list(ids) + [W.EOS]):
        out[k] = R.f(s)
        if t != W.EOS:
            s = m.feed(s, t)
    return out


def shard_records(n, seed, minlen):
    recs = []
    for p in sorted(SHARDS.glob("train.part*.u16.gz")):
        d = gzip.open(p).read()
        i = 0
        while i < len(d):
            k = struct.unpack_from("<H", d, i)[0]
            if k // 2 >= minlen:
                recs.append(np.frombuffer(d, "<u2", k // 2, i + 2).astype(np.int64))
            i += 2 + k
    random.Random(seed).shuffle(recs)
    return recs[:n]


def main():
    mode, pkg = sys.argv[1], sys.argv[2]
    minlen = int(sys.argv[sys.argv.index("--minlen") + 1]) if "--minlen" in sys.argv else 0
    m = E.load_model(f"mica:{pkg}")
    R = Reader(m)
    t0 = time.time()
    if mode in ("tokens", "shards"):
        if mode == "tokens":
            path, *rng = sys.argv[3].split(":")
            recs = E.token_records(path)
            if rng:
                recs = recs[int(rng[0]):int(rng[1])]
            outp = sys.argv[4]
        else:
            recs = shard_records(int(sys.argv[3]), int(sys.argv[4]), minlen)
            outp = sys.argv[5]
        recs = [list(map(int, r)) for r in recs if len(r) >= minlen]
        Fs, lens, toks = [], [], []
        for k, r in enumerate(recs):
            Fs.append(run_record(m, R, r))
            lens.append(len(r))
            toks.extend(r)
            if (k + 1) % 500 == 0:
                print(f"{k+1}/{len(recs)} recs {sum(lens)+len(lens)} pos {time.time()-t0:.0f}s", flush=True)
        np.savez(outp, F=np.concatenate(Fs), lens=np.array(lens), toks=np.array(toks, np.int32))
    else:
        path, *rng = sys.argv[3].split(":")
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if rng:
            rows = rows[int(rng[0]):int(rng[1])]
        vocab = W.Vocab.load("/home/claude/mica/wordw/data/word/vocab.json")
        ids_of = lambda s: [W.SEP if t == "<sep>" else vocab.index.get(t, W.oov_id(t)) for t in W.tokenize(s)]
        res = []
        for k, r in enumerate(rows):
            ctx = ids_of(r["ctx"])
            s = m.start()
            for t in ctx:
                s = m.feed(s, t)
            ends = []
            for e in r["endings"]:
                ei = ids_of(e)
                s2 = m.fork(s)
                Fe = np.empty((len(ei), len(R.cells)), np.int8)
                for j, t in enumerate(ei):
                    Fe[j] = R.f(s2)
                    s2 = m.feed(s2, t)
                ends.append((ei, Fe))
            res.append({"ind": r["ind"], "label": int(r["label"]), "topic": r["metadata"].get("topic"),
                        "ctx": ctx, "ends": ends})
            if (k + 1) % 100 == 0:
                print(f"{k+1} rows {time.time()-t0:.0f}s", flush=True)
        pickle.dump(res, open(sys.argv[4], "wb"))
    print("done", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
