#!/usr/bin/env python3
"""Scaled refit of the Flame-W memory readout (v0.3.1): same readout as v0.3 (recall R + 16-channel
association A/Bm/lam), warm-started from v0.3, trained on ~7x more positions with windows built per
batch (low memory). Selection exactly as v0.3: held-out long records (train2, first 40k positions)
and no-TinyStories validation records 500-1499; records identical to held-out ones are dropped.

    python fit_mem2.py --hours 1.25 --tag v031

Record of the 0.3.1 run: paths point at the cloud working folders. The feature files come from
flame/runs/claude_flamew_memory_20261003/mem_pass.py, and fit_mem.py / final_mem.py are in that folder too.
"""
import argparse, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/claude/mica/mem")
import fit_mem as FM
import word_eval as E
from word_eval import W

torch.set_num_threads(2)
V, WIN = W.N_SYMBOLS, 64
SKIP = np.array([0, W.BOS, W.EOS])
J = np.arange(WIN)


class Data:
    def __init__(self, paths, exclude=None, max_pos=None):
        Fs, Ts, starts, ks, G = [], [], [], [], []
        off = 0; dropped = 0; total = 0
        for p in paths:
            d = np.load(p); DF, DT, DL = d["F"], d["toks"], d["lens"]
            o = 0; fo = 0
            for L in DL:
                L = int(L); r = DT[o:o + L]
                keep = exclude is None or r.tobytes() not in exclude
                if keep:
                    Fs.append(DF[fo:fo + L + 1]); G.append(r.astype(np.int32))
                    Ts.append(np.append(r, W.EOS).astype(np.int32))
                    starts.append(np.full(L + 1, off, np.int64)); ks.append(np.arange(L + 1, dtype=np.int32))
                    off += L; total += L + 1
                else:
                    dropped += 1
                o += L; fo += L + 1
                if max_pos and total >= max_pos:
                    break
            if max_pos and total >= max_pos:
                break
        self.F = np.concatenate(Fs); self.T = np.concatenate(Ts); self.start = np.concatenate(starts)
        self.k = np.concatenate(ks); self.G = np.concatenate(G) if G else np.zeros(0, np.int32)
        Gm = self.G.astype(np.int64).copy(); Gm[np.isin(Gm, SKIP)] = -1; self.Gm = np.append(Gm, -1)
        self.dropped = dropped

    def __len__(self):
        return len(self.T)

    def batch(self, idx):
        k = self.k[idx].astype(np.int64)[:, None]
        pos = self.start[idx][:, None] + k - 1 - J[None, :]
        ok = (k - 1 - J[None, :]) >= 0
        H = np.where(ok, self.Gm[np.where(ok, pos, -1)], -1)
        return (torch.from_numpy(self.F[idx]), torch.from_numpy(H), torch.from_numpy(self.T[idx].astype(np.int64)))


def nll(model, D, co, bias, elig, bs=1500):
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(D), bs):
            F, H, T = D.batch(np.arange(i, min(i + bs, len(D))))
            z = FM.base_logits(F, co, bias, elig) + model.bonus(H)
            tot += float(torch.nn.functional.cross_entropy(z, T, reduction="sum"))
    return tot / len(D) / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.25); ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--l2", type=float, default=1e-5); ap.add_argument("--tag", default="v031")
    ap.add_argument("--eval_every", type=int, default=1500)
    a = ap.parse_args()
    torch.manual_seed(0)
    m0 = E.load_model("mica:/home/claude/mica/wordw/pkg_B")
    co = torch.tensor(m0.co, dtype=torch.float32); bias = torch.tensor(m0.bias, dtype=torch.float32)
    elig = torch.tensor(m0.elig)
    MEM = "/home/claude/mica/mem/"
    HO = Data([MEM + "train2.npz"], max_pos=40000); NS = Data([MEM + "nsval.npz"])
    excl = {HO.G[s:s + L].tobytes() for s, L in []}  # filled below
    # held-out records as byte strings
    excl = set()
    for D in (HO, NS):
        b = np.flatnonzero(D.k == 0)
        for i, s in enumerate(b):
            e = b[i + 1] if i + 1 < len(b) else len(D)
            L = e - s - 1; st = D.start[s]
            excl.add(D.G[st:st + L].astype(np.int32).tobytes())
    new = sorted(str(p) for p in Path(".").glob("long_c*.npz")) + sorted(str(p) for p in Path(".").glob("ns_c*.npz"))
    TR = Data([MEM + "train1.npz", MEM + "train_ns_a.npz", MEM + "train_ns_b.npz"] + new, exclude=excl)
    print(f"[{a.tag}] train {len(TR):,} positions from {3 + len(new)} files (dropped {TR.dropped} held-out duplicates); "
          f"held-out long {len(HO):,}, nsval {len(NS):,}", flush=True)
    model = FM.Mem(16, FM.classes(), mask=None)
    st = torch.load(MEM + "mem_mix16.pt", weights_only=False)["state"]
    model.load_state_dict(st)
    b_ho, b_ns = nll(FM.Mem(0, FM.classes()), HO, co, bias, elig), nll(FM.Mem(0, FM.classes()), NS, co, bias, elig)
    v_ho, v_ns = nll(model, HO, co, bias, elig), nll(model, NS, co, bias, elig)
    best = ((v_ho - b_ho) + (v_ns - b_ns)) / 2
    print(f"[{a.tag}] B alone: long {b_ho:.4f}, nsval {b_ns:.4f} | v0.3 start: long {v_ho:.4f}, nsval {v_ns:.4f}, select {best:+.4f}", flush=True)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    t0 = time.time(); g = np.random.default_rng(5); step = 0; ep = 0
    while time.time() - t0 < a.hours * 3600:
        perm = g.permutation(len(TR))
        for i in range(0, len(TR), 1024):
            F, H, T = TR.batch(perm[i:i + 1024])
            z = FM.base_logits(F, co, bias, elig) + model.bonus(H)
            loss = torch.nn.functional.cross_entropy(z, T)
            reg = a.l2 * (model.A.pow(2).sum() + model.Bm.pow(2).sum())
            opt.zero_grad(); (loss + reg).backward(); opt.step(); step += 1
            if step % a.eval_every == 0 or time.time() - t0 > a.hours * 3600:
                ho, ns = nll(model, HO, co, bias, elig), nll(model, NS, co, bias, elig)
                sel = ((ho - b_ho) + (ns - b_ns)) / 2
                print(f"[{a.tag}] epoch {ep} step {step}: long {ho:.4f}, nsval {ns:.4f}, select {sel:+.4f} (v0.3 {best if step == a.eval_every else ''}) "
                      f"{time.time() - t0:.0f}s", flush=True)
                if sel < best:
                    best = sel; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    torch.save({"state": best_state, "k": 16, "dev_bits": best, "dev_base": 0.0, "args": vars(a)}, f"mem_{a.tag}.pt")
                if time.time() - t0 > a.hours * 3600:
                    break
        ep += 1
    torch.save({"state": best_state, "k": 16, "dev_bits": best, "dev_base": 0.0, "args": vars(a)}, f"mem_{a.tag}.pt")
    print(f"[{a.tag}] done: best select {best:+.4f}", flush=True)


if __name__ == "__main__":
    main()
