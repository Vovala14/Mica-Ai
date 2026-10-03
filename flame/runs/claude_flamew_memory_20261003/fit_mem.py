#!/usr/bin/env python3
"""Fit Flame-W memory readouts on top of the frozen automaton (B_lr010).

score(w) = B's integer score  (bias + co @ f, from mem_pass.py, exact)
         + recall:      sum over occurrences of w in the last 64 tokens of R[class(w), lagbucket]
         + association: Bm[w] . m,   m = sum_j lam[lagbucket(j)] * A[token at lag j]   (k dims)

    python fit_mem.py --k 16 --l2 1e-6 --epochs 4 --tag k16
"""
import argparse, json, math, pickle, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/claude/mica/wordw/eval")
import word_eval as E
from word_eval import W

torch.set_num_threads(2)
V = W.N_SYMBOLS
WIN = 64
LAGB = np.array([0, 0, 1, 2, 3, 3, 3] + [4] * 6 + [5] * 12 + [6] * 41)   # index = lag 1..64
NB = 7
NC = 5


def classes():
    cnt = np.load(Path(__file__).with_name("unigram.npy"))
    vocab = W.Vocab.load("/home/claude/mica/wordw/data/word/vocab.json")
    rank = np.empty(V, np.int64); rank[np.argsort(-cnt)] = np.arange(V)
    cls = np.full(V, 4, np.int64); cls[rank < 2000] = 3; cls[rank < 100] = 2
    for k, t in enumerate(vocab.tokens):
        if not any(ch.isalnum() for ch in t):
            cls[W.FIRST_WORD + k] = 1
    cls[1:W.FIRST_WORD] = 0
    return cls


SKIP = {0, W.BOS, W.EOS}


def histories(seqs):
    """seqs: list of (history_prefix, targets). Returns H [n,64] (-1 = none), targets [n]."""
    H, T = [], []
    for hist, tg in seqs:
        h = list(hist)
        for t in tg:
            row = [x if x not in SKIP else -1 for x in h[::-1][:WIN]]
            H.append(row + [-1] * (WIN - len(row)))
            T.append(t)
            h.append(t)
    return np.array(H, np.int64), np.array(T, np.int64)


def load_tokens_npz(paths):
    Fs, seqs = [], []
    for p in paths:
        d = np.load(p)
        Fs.append(d["F"])
        o = 0
        for L in d["lens"]:
            r = d["toks"][o:o + L].tolist(); o += L
            seqs.append(([], r + [W.EOS]))
    H, T = histories(seqs)
    return np.concatenate(Fs), H, T


class Mem(torch.nn.Module):
    def __init__(self, k, cls, use_assoc=True, mask=None):
        super().__init__()
        self.register_buffer("mask", torch.ones(V, 1) if mask is None else torch.tensor(mask, dtype=torch.float32)[:, None])
        self.k = k
        self.cls = torch.tensor(cls)
        self.lagb = torch.tensor(np.concatenate([LAGB[1:65]]))            # j=0..63 -> bucket
        self.R = torch.nn.Parameter(torch.zeros(NC, NB))
        self.use_assoc = use_assoc and k > 0
        if self.use_assoc:
            g = torch.Generator().manual_seed(0)
            self.A = torch.nn.Parameter(torch.randn(V, k, generator=g) * 0.01)
            self.Bm = torch.nn.Parameter(torch.zeros(V, k))
            self.lam = torch.nn.Parameter(torch.ones(NB))

    def bonus(self, H):
        n = H.shape[0]
        valid = H >= 0
        Hc = H.clamp(min=0)
        out = torch.zeros(n, V)
        lb = self.lagb[None, :].expand_as(H)
        rv = self.R[self.cls[Hc], lb] * valid
        out.scatter_add_(1, Hc, rv)
        if self.use_assoc:
            w = self.lam[lb] * valid                                       # (n,64)
            A, Bm = self.A * self.mask, self.Bm * self.mask
            m = torch.einsum("nj,njk->nk", w, A[Hc])
            out = out + m @ Bm.T
        return out


def base_logits(F, co, bias, elig):
    z = (bias[None, :] + F.float() @ co.T) / 1024.0
    return z.masked_fill(~elig[None, :], -1e30)


def nll(model, F, H, T, co, bias, elig, bs=1024):
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(T), bs):
            z = base_logits(F[i:i+bs], co, bias, elig) + model.bonus(H[i:i+bs])
            tot += float(torch.nn.functional.cross_entropy(z, T[i:i+bs], reduction="sum"))
    return tot / len(T) / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--train", default="train1.npz,train2.npz"); ap.add_argument("--tag", default="run")
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--topn", type=int, default=0)
    ap.add_argument("--hold", type=int, default=40000)
    a = ap.parse_args()
    torch.manual_seed(0)
    m0 = E.load_model("mica:/home/claude/mica/wordw/pkg_B")
    co = torch.tensor(m0.co, dtype=torch.float32); bias = torch.tensor(m0.bias, dtype=torch.float32)
    elig = torch.tensor(m0.elig)
    cls = classes()
    tr = [p for p in a.train.split(",") if Path(p).exists() and p != "train2.npz"]
    Ftr, Htr, Ttr = (torch.tensor(x) for x in load_tokens_npz(tr))
    Fho, Hho, Tho = (torch.tensor(x[:a.hold]) for x in load_tokens_npz(["train2.npz"]))
    Fns, Hns, Tns = (torch.tensor(x) for x in load_tokens_npz(["nsval.npz"]))
    Fdv, Hdv, Tdv = (torch.tensor(x) for x in load_tokens_npz(
        ["dev_chat.npz", "dev_every.npz", "dev_chatlong.npz", "dev_everylong.npz"]))
    mask = None
    if a.topn:
        cnt = np.load(Path(__file__).with_name("unigram.npy"))
        mask = np.zeros(V); mask[np.argsort(-cnt)[:a.topn]] = 1; mask[:W.FIRST_WORD] = 1
    model = Mem(a.k, cls, mask=mask)
    print(f"[{a.tag}] train {len(Ttr):,} positions from {tr}; dev {len(Tdv):,}", flush=True)
    base_dev = nll(Mem(0, cls), Fdv, Hdv, Tdv, co, bias, elig)
    base_ho = nll(Mem(0, cls), Fho, Hho, Tho, co, bias, elig)
    base_ns = nll(Mem(0, cls), Fns, Hns, Tns, co, bias, elig)
    print(f"[{a.tag}] B alone: dev {base_dev:.4f}, held-out long {base_ho:.4f}, nostories val {base_ns:.4f}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, best_state, t0 = float("inf"), None, time.time()
    n = len(Ttr)
    for ep in range(a.epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(ep))
        run = 0.0
        for bi, i in enumerate(range(0, n, a.bs)):
            idx = perm[i:i + a.bs]
            z = base_logits(Ftr[idx], co, bias, elig) + model.bonus(Htr[idx])
            loss = torch.nn.functional.cross_entropy(z, Ttr[idx])
            reg = 0.0
            if model.use_assoc:
                reg = a.l2 * (model.A.pow(2).sum() + model.Bm.pow(2).sum())
            opt.zero_grad(); (loss + reg).backward(); opt.step()
            run += float(loss.detach())
            if (bi + 1) % 100 == 0:
                print(f"  ep {ep} batch {bi+1} train {run/100/math.log(2):.4f} ({time.time()-t0:.0f}s)", flush=True)
                run = 0.0
        ho = nll(model, Fho, Hho, Tho, co, bias, elig)
        ns = nll(model, Fns, Hns, Tns, co, bias, elig)
        d = nll(model, Fdv, Hdv, Tdv, co, bias, elig)
        sel = (ho - base_ho + ns - base_ns) / 2
        print(f"[{a.tag}] epoch {ep}: long {ho:.4f} (B {base_ho:.4f}), nsval {ns:.4f} (B {base_ns:.4f}), "
              f"dev {d:.4f} (B {base_dev:.4f}); select {sel:+.4f} {time.time()-t0:.0f}s", flush=True)
        if sel < best:
            best = sel
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"state": best_state, "k": a.k, "dev_bits": best, "dev_base": 0.0, "args": vars(a)},
               f"mem_{a.tag}.pt")
    print(f"[{a.tag}] best dev bits {best:.4f} vs B {base_dev:.4f}", flush=True)


if __name__ == "__main__":
    main()
