#!/usr/bin/env python3
"""What does MICA's rule machinery add over an ordinary hashed context table?

MICA's structured rule book works as a hashed context lookup: in phase p the
byte at the cell picks a page, and the argmax of the page's candidates' scores
over the older bytes picks one of C buckets, whose six learned immediates the
readout reads. This script trains the SAME readout on the SAME data with the
SAME fitting procedure, and changes only where the bucket index comes from:

  tape_only      no rule features at all (the byte-tape readout alone)
  mica_rules     the bucket MICA's rule book assigns (taken from a trained
                 .mica file: its routing, scoring terms and balanced biases);
                 the immediates are refitted from a random start, like MICA's
  hash_same_buckets  an ordinary hash of exactly the bytes the phase can see
                 (the cell's byte and the max_back[p] before it), into the
                 same number of buckets per phase as MICA
  hash_same_bytes    the same hash with as many buckets as fit in MICA's file
                 size, storing only the six int8 values per bucket

Every variant: the byte codes of the given .mica file, its 240-probe layout
(48 tape probes, 192 rule-feature probes), int8 probe coefficients, int8
immediates, the file's rounding; an initial joint fit, then refit rounds on
fresh records at a smaller step size, each with early stopping on held-out
positions of its own sample -- the procedure MICA's training uses. Scores:
pooled bits per target on the val1000 record set (checks/common.py), which no
variant trains on.

    python r1/checks/hash_control.py --run r1/runs/sweep/p16_5bit_c256_self1 \
        --fit-records 40000 --out r1/runs/checks/hash_control.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (utf8_stdout, R1, record_set, run_geometry, worker_env, beat,
                    hard_exit)

EMPTY = 257                      # "no byte here" (before the record's BOS)


def seq_of(recs):
    """Per record: [BOS, bytes...] as int64 arrays, and the flat target list
    (each byte, then EOS) in the order the features are built."""
    import numpy as np
    from mica_r1 import spec
    seqs, tgts = [], []
    for r in recs:
        a = np.frombuffer(r, np.uint8).astype(np.int64)
        seqs.append(np.concatenate([[spec.BOS], a]))
        tgts.append(np.concatenate([a, [spec.EOS]]))
    return seqs, np.concatenate(tgts)


def windows(seqs, back: int):
    """(N, back+1) symbols at the cell and the `back` cells before it, for
    every cell of every record (cell j holds seqs[j]); EMPTY before BOS."""
    import numpy as np
    out = []
    for s in seqs:
        pad = np.concatenate([np.full(back, EMPTY), s])
        idx = np.arange(len(s))[:, None] + back - np.arange(back + 1)[None, :]
        out.append(pad[idx])
    return np.concatenate(out)


def cell_to_target_rows(seqs):
    """Row of cell j (holding seqs[j]) in the flat cell list, for the target
    predicted with the head at cell j (lag 1) and at cell j-1 (lag 2; -1 when
    there is none)."""
    import numpy as np
    lag1, lag2, base = [], [], 0
    for s in seqs:
        n = len(s)                                   # cells = targets
        j = np.arange(n)
        lag1.append(base + j)
        lag2.append(np.where(j >= 1, base + j - 1, -1))
        base += n
    return np.concatenate(lag1), np.concatenate(lag2)


def mica_buckets(m, seqs, dev, chunk: int = 20000):
    """(N_cells, N_PHASE) global bucket rows page*C + winner, computed from
    the rule book directly: scoring reads only the byte tape, so the winner
    at a cell is a function of the bytes (exactly what fit.provenance sees)."""
    import numpy as np
    import torch
    from mica_r1 import spec
    K = spec.TAPE_CHANNELS
    C = spec.N_CANDIDATES
    off = np.array(spec.OFFSETS)
    back = int(-off.min())
    W = windows(seqs, back)                                   # (N, back+1)
    codes = np.zeros((259, K), np.int64)
    codes[:m.inj_delta.shape[0]] = m.inj_delta[:, :K].astype(np.int64)
    # EMPTY cells (index 257 in `codes` would be EOS: use a separate row)
    codes_t = torch.as_tensor(np.concatenate([codes, np.zeros((1, K), np.int64)]),
                              device=dev)
    EMPTY_ROW = codes_t.shape[0] - 1
    Wt = torch.as_tensor(np.where(W == EMPTY, EMPTY_ROW, W), device=dev)
    rc = torch.tensor(spec.ROUTING_CHANNELS, device=dev)
    rw = 1 << torch.arange(len(spec.ROUTING_CHANNELS), device=dev)
    sc_nb = torch.as_tensor(m.sc_nb.astype(np.int64), device=dev)
    sc_ch = torch.as_tensor(m.sc_ch.astype(np.int64), device=dev)
    sc_co = torch.as_tensor(m.sc_co.astype(np.int64), device=dev)
    sc_bias = torch.as_tensor(m.sc_bias.astype(np.int64), device=dev)
    dist = torch.as_tensor(-off, device=dev)                   # selector -> back
    tie = torch.arange(C, device=dev)
    out = torch.empty(len(W), spec.N_PHASE, dtype=torch.int32, device=dev)
    for i in range(0, len(W), chunk):
        w = Wt[i:i + chunk]                                    # (n, back+1)
        own = codes_t[w[:, 0]]                                 # (n, K)
        bits = ((own[:, rc] >= own[:, rc + spec.ROUTING_PAIR_OFFSET]).long()
                * rw).sum(-1)
        for p in range(spec.N_PHASE):
            page = spec.PAGE_STRIDE * p + bits                 # (n,)
            nb = sc_nb[page]                                   # (n,C,T)
            sym = torch.gather(w, 1, dist[nb].reshape(len(w), -1)
                               ).reshape(nb.shape)             # (n,C,T)
            vals = codes_t[sym, sc_ch[page]]                   # (n,C,T)
            sc = sc_bias[page] + (sc_co[page] * vals).sum(-1)  # (n,C)
            win = (sc * (C + 1) - tie).argmax(-1)
            out[i:i + chunk, p] = (page * C + win).int()
        beat("hash-control mica buckets")
    return out


def hash_buckets(seqs, max_back, n_buckets: int, dev, salt: int = 12345):
    """(N_cells, N_PHASE) rows p*n_buckets + h_p(cell's byte, max_back[p]
    bytes before it): a 64-bit multiply-xorshift hash, independent per phase."""
    import numpy as np
    import torch
    from mica_r1 import spec
    rng = np.random.default_rng(salt)
    back = max(max_back)
    W = windows(seqs, back).astype(np.uint64) + np.uint64(1)
    out = np.empty((len(W), spec.N_PHASE), np.int64)
    with np.errstate(over="ignore"):
        for p in range(spec.N_PHASE):
            mult = rng.integers(1, 2**63, size=back + 1, dtype=np.uint64) | np.uint64(1)
            h = np.full(len(W), np.uint64(rng.integers(1, 2**63)), np.uint64)
            for j in range(max_back[p] + 1):
                h = (h ^ (W[:, j] * mult[j])) * np.uint64(0x9E3779B97F4A7C15)
                h ^= h >> np.uint64(29)
            h *= np.uint64(0xBF58476D1CE4E5B9)
            h ^= h >> np.uint64(32)
            out[:, p] = p * n_buckets + (h % np.uint64(n_buckets)).astype(np.int64)
    return torch.as_tensor(out, dtype=torch.int32, device=dev)


class Readout:
    """The tape machine's readout with the file's integer rounding:
    logit = (bias + sum_tape w*code + sum_groups w*imm[bucket]) / divisor."""

    def __init__(self, m, n_rows, dev, seed=0):
        import numpy as np
        import torch
        from mica_r1 import spec
        self.dev = dev
        self.D = spec.LOGIT_DIVISOR
        K, VW = spec.TAPE_CHANNELS, spec.VSET_WIDTH
        cells = (spec.N_CELLS - m.pr_cell[0].astype(np.int64)) % spec.N_CELLS
        chans = m.pr_chan[0].astype(np.int64)
        self.tape = [(int(l), int(c)) for l, c in zip(cells, chans) if c < K]
        work = sorted({(int(l), int((c - K) // VW))
                       for l, c in zip(cells, chans) if c >= K})
        self.groups = work                                  # (lag, phase group)
        self.ulags = sorted({l for l, _ in self.tape})
        codes = np.zeros((spec.N_SYMBOLS + 2, K), np.float32)
        codes[:m.inj_delta.shape[0]] = m.inj_delta[:, :K]
        self.codes = torch.as_tensor(codes, device=dev)     # row +1: EMPTY
        g = torch.Generator().manual_seed(seed)
        V = spec.N_SYMBOLS
        self.w_tape = torch.zeros(V, len(self.tape), device=dev, requires_grad=True)
        self.w_work = torch.zeros(V, len(work) * VW, device=dev, requires_grad=True)
        self.bias = torch.zeros(V, device=dev, requires_grad=True)
        sign = torch.where(torch.rand(n_rows, VW, generator=g) < 0.5, -1.0, 1.0)
        self.imm = (sign * (4 + 12 * torch.rand(n_rows, VW, generator=g))
                    ).to(dev).requires_grad_(True)
        elig = torch.full((V,), float("-inf"), device=dev)
        elig[:256] = 0.0
        elig[spec.EOS] = 0.0
        self.elig = elig

    @staticmethod
    def rnd(x, lo, hi):
        return x + (x.detach().round().clamp(lo, hi) - x.detach())

    def tape_feats(self, ctx):
        """ctx: (n, max lag) symbols at lags 1.. (EMPTY row = N_SYMBOLS+1)."""
        import torch
        cols = [self.codes[ctx[:, l - 1], c] for l, c in self.tape]
        return torch.stack(cols, 1)                          # (n, 48)

    def logits(self, tf, prov, use_rules: bool):
        import torch
        sc = self.rnd(self.bias, -32767, 32767) + \
            tf @ self.rnd(self.w_tape, -127, 127).T
        if use_rules:
            pv = prov.long()                                  # (n, G)
            iv = torch.where(pv[..., None] >= 0,
                             self.rnd(self.imm, -127, 127)[pv.clamp(min=0)],
                             torch.zeros((), device=self.dev))  # (n,G,VW)
            sc = sc + iv.reshape(len(pv), -1) @ \
                self.rnd(self.w_work, -127, 127).T
        return sc / self.D + self.elig


def build(seqs, tgt, bucket_fn, ro, dev):
    """Features for every target: tape symbols at each lag, and per rule
    group the bucket row of the cell the group's lag points at (or -1)."""
    import numpy as np
    import torch
    from mica_r1 import spec
    maxlag = max(ro.ulags)
    ctx = []
    for s in seqs:
        pad = np.concatenate([np.full(maxlag, spec.N_SYMBOLS + 1), s])
        n = len(s)
        idx = np.arange(n)[:, None] + maxlag - np.arange(maxlag)[None, :]
        ctx.append(pad[idx])                                  # lags 1..maxlag
    ctx = torch.as_tensor(np.concatenate(ctx), device=dev)
    prov = None
    if bucket_fn is not None:
        cellb = bucket_fn(seqs)                               # (N_cells, P)
        lag1, lag2 = cell_to_target_rows(seqs)
        cols = []
        for lag, grp in ro.groups:
            rows = torch.as_tensor(lag1 if lag == 1 else lag2, device=dev)
            col = torch.where(rows >= 0, cellb[rows.clamp(min=0), grp],
                              torch.full_like(rows, -1, dtype=torch.int32))
            cols.append(col)
        prov = torch.stack(cols, 1).int()
    return ctx, torch.as_tensor(tgt, device=dev), prov


def fit(ro, data, steps, lr_scale, use_rules, log, batch=16384, check=250):
    """fit.fit_readout_and_rules' procedure: Adam on every coefficient, the
    biases and (with rules) the immediates, file rounding in the forward
    pass, early stopping on 10% held-out positions of this sample."""
    import torch
    import torch.nn.functional as Fn
    ctx, tgt, prov = data
    n = len(tgt)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g).to(ro.dev)
    n_hold = min(n // 10, 200_000)
    hold, tr = perm[:n_hold], perm[n_hold:]
    params = [ro.w_tape, ro.w_work, ro.bias, ro.imm]
    opt = torch.optim.Adam([
        {"params": [ro.w_tape, ro.w_work], "lr": 1.0 * lr_scale},
        {"params": [ro.bias], "lr": 0.01 * ro.D * lr_scale},
        {"params": [ro.imm], "lr": 1.0 * lr_scale}])

    def feats(rows):
        return ro.tape_feats(ctx[rows]), (prov[rows] if prov is not None else None)

    def held():
        with torch.no_grad():
            tot = 0.0
            for i in range(0, n_hold, 32768):
                r = hold[i:i + 32768]
                tf, pv = feats(r)
                tot += float(Fn.cross_entropy(ro.logits(tf, pv, use_rules),
                                              tgt[r], reduction="sum"))
            return tot / n_hold / math.log(2)
    best = held()
    keep = [p.detach().clone() for p in params]
    for it in range(steps):
        r = tr[torch.randint(0, len(tr), (batch,), device=ro.dev)]
        tf, pv = feats(r)
        loss = Fn.cross_entropy(ro.logits(tf, pv, use_rules), tgt[r])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % check == 0 or it == steps - 1:
            h = held()
            if h < best:
                best = h
                keep = [p.detach().clone() for p in params]
            beat(f"hash-control fit {it + 1}")
    with torch.no_grad():
        for p, k in zip(params, keep):
            p.copy_(k)
    log(f"  fit: held-out {best:.4f} after {steps} steps")
    return best


def evaluate(ro, data, use_rules):
    import torch
    import torch.nn.functional as Fn
    ctx, tgt, prov = data
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(tgt), 32768):
            sl = slice(i, i + 32768)
            tf = ro.tape_feats(ctx[sl])
            pv = prov[sl] if prov is not None else None
            tot += float(Fn.cross_entropy(ro.logits(tf, pv, use_rules), tgt[sl],
                                          reduction="sum"))
    return tot / len(tgt) / math.log(2)


def worker(a) -> int:
    import numpy as np
    import torch
    from mica_r1 import spec, serialize, engine
    from make_records import load_records
    assert engine.ROUTING_MODE == "pairdiff"
    torch.set_num_threads(max(1, a.threads))
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                       if a.device == "auto" else a.device)
    t0 = time.time()

    def log(msg):
        beat(f"hash-control {msg}")
        print(f"[hash] {msg}  ({time.time() - t0:.0f}s)", flush=True)

    m = serialize.load(a.mica)
    info = json.loads((Path(a.run) / "run_info.json").read_text())
    mb = tuple(int(x) for x in info["args"]["rule_max_back"].split(","))
    mb = tuple(mb[p % len(mb)] for p in range(spec.N_PHASE))
    file_bytes = Path(a.mica).stat().st_size
    K, VW, C = spec.TAPE_CHANNELS, spec.VSET_WIDTH, spec.N_CANDIDATES
    per_phase = spec.N_PAGES * C // spec.N_PHASE
    readout_bytes = spec.PROBE_SECTION + spec.N_SYMBOLS * K + spec.HEADER_LEN
    same_bytes = (file_bytes - readout_bytes) // (VW * spec.N_PHASE)
    variants = {
        "tape_only": (None, 1),
        "mica_rules": ("mica", spec.N_PAGES * C),
        "hash_same_buckets": ("hash", per_phase * spec.N_PHASE),
        "hash_same_bytes": ("hash_big", same_bytes * spec.N_PHASE),
    }
    wanted = a.variants.split(",")
    train = load_records(a.train)
    rng = np.random.default_rng(a.seed)
    samples = []
    for r_ in range(1 + a.rounds):
        pick = rng.choice(len(train), a.fit_records, replace=False)
        samples.append([train[i][:256] for i in pick if len(train[i])])
    del train
    val = record_set(a.eval_set, a.data_root)
    log(f"{len(samples)} fit samples of {a.fit_records:,} records, eval on "
        f"{a.eval_set} ({len(val)} records); buckets per phase: MICA "
        f"{per_phase:,}, same-bytes hash {same_bytes:,}")

    def bucket_fn(kind):
        if kind == "mica":
            return lambda seqs: mica_buckets(m, seqs, dev)
        if kind == "hash":
            return lambda seqs: hash_buckets(seqs, mb, per_phase, dev)
        if kind == "hash_big":
            return lambda seqs: hash_buckets(seqs, mb, same_bytes, dev)
        return None

    out = {"mica": a.mica, "file_bytes": file_bytes, "fit_records": a.fit_records,
           "rounds": a.rounds, "steps": [2 * a.fit_steps] + [a.fit_steps] * a.rounds,
           "eval_set": a.eval_set, "max_back": mb, "results": {}}
    if Path(a.out).exists():
        out["results"] = json.loads(Path(a.out).read_text()).get("results", {})
    # MICA's pipeline fits the tape readout first and starts the rule fit
    # from it (work coefficients 0, immediates random). Every variant starts
    # from the same tape readout, fitted once here on the first sample.
    ro0 = Readout(m, 1, dev)
    ss, st = seq_of(samples[0])
    d0 = build(ss, st, None, ro0, dev)
    log("shared start: tape readout fit")
    fit(ro0, d0, 2 * a.fit_steps, 1.0, False, log)
    start = [t.detach().clone() for t in (ro0.w_tape, ro0.bias)]
    del ro0, d0
    for name in wanted:
        kind, n_rows = variants[name]
        log(f"variant {name}: {n_rows:,} bucket rows")
        ro = Readout(m, max(1, n_rows), dev)
        with torch.no_grad():
            ro.w_tape.copy_(start[0])
            ro.bias.copy_(start[1])
        use_rules = kind is not None
        bf = bucket_fn(kind)
        vs, vt = seq_of(val)
        vdata = build(vs, vt, bf, ro, dev)
        hist = []
        for r_, sample in enumerate(samples):
            ss, st = seq_of(sample)
            data = build(ss, st, bf, ro, dev)
            steps = 2 * a.fit_steps if r_ == 0 else a.fit_steps
            lr = 1.0 if r_ == 0 else a.round_lr
            h = fit(ro, data, steps, lr, use_rules, log)
            v = evaluate(ro, vdata, use_rules)
            hist.append({"round": r_, "fit_heldout": h, "eval_bits": v})
            log(f"variant {name} round {r_}: {a.eval_set} {v:.4f} bits/target")
            del data
        imm_bytes = n_rows * VW if use_rules else 0
        out["results"][name] = {
            "bucket_rows": n_rows if use_rules else 0,
            "eval_bits": hist[-1]["eval_bits"], "history": hist,
            "table_bytes": imm_bytes + readout_bytes}
        Path(a.out).write_text(json.dumps(out, indent=1))
        del ro, vdata
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    log("done")
    hard_exit(0)


def main() -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", default=str(R1 / "data/bulk/train.jsonl"))
    ap.add_argument("--fit-records", type=int, default=40000)
    ap.add_argument("--fit-steps", type=int, default=1500)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--round-lr", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants",
                    default="tape_only,mica_rules,hash_same_buckets,hash_same_bytes")
    ap.add_argument("--eval-set", default="val1000")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mica")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    cmd = [sys.executable, __file__, "--worker", "--run", a.run,
           "--mica", str(Path(a.run) / "best.mica")] + \
        [x for k, v in vars(a).items() if k not in ("run", "worker", "mica")
         and v is not None for x in (f"--{k.replace('_', '-')}", str(v))]
    p = subprocess.run(cmd, env=worker_env(run_geometry(a.run)))
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
