#!/usr/bin/env python3
"""Storage-matched neural baselines. Section 15, "Language evaluation".

  "Use a smoothed byte trigram, a small recurrent predictor, and a tiny byte
   Transformer on the same split. Compare total artifact bytes and process
   memory, not neural parameter counts alone."

Section 16 adds one more: "Compare with a reservoir-style baseline as well."

Every model here is sized so its stored parameters fit the same 86,820-byte cap
as the R1 model file, assuming int8 storage. Both the int8 and float32 sizes are
reported, because a float-trained model that has not actually been quantised is
not an 86 KB artifact and it would be dishonest to present it as one.

The reservoir is the interesting one under a storage cap: its recurrent matrix
is regenerated from a 4-byte seed, so essentially the whole budget buys readout.
"""

from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from make_records import load_records
from gate import bootstrap_ci

VOCAB = 256          # raw bytes only, to match gate.py exactly (no EOS term)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class ByteGRU(nn.Module):
    def __init__(self, hidden: int, embed: int):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, embed)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.out = nn.Linear(hidden, VOCAB)

    def forward(self, x, state=None):
        y, state = self.rnn(self.emb(x), state)
        return self.out(y), state


class ByteTransformer(nn.Module):
    """One causal self-attention block. Deliberately minimal."""

    def __init__(self, d_model: int, n_head: int, ctx: int):
        super().__init__()
        self.ctx = ctx
        self.emb = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(ctx, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Linear(2 * d_model, d_model))
        self.out = nn.Linear(d_model, VOCAB)

    def forward(self, x, state=None):
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))[None]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h), attn_mask=mask,
                         need_weights=False)
        h = h + a
        h = h + self.ff(self.ln2(h))
        return self.out(h), None


class Reservoir(nn.Module):
    """Echo state network: random fixed recurrence, trained linear readout.

    Storage is a 4-byte seed plus the readout. Section 16 asks for this
    comparison specifically, and under a byte cap it is a strong baseline
    because almost the entire budget goes into the part that is learned.
    """

    def __init__(self, hidden: int, seed: int = 0, leak: float = 0.3,
                 radius: float = 0.9):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(hidden, hidden, generator=g) / math.sqrt(hidden)
        # scale to the requested spectral radius so the reservoir is stable
        with torch.no_grad():
            ev = torch.linalg.eigvals(W).abs().max().real
            W = W * (radius / max(float(ev), 1e-6))
        self.register_buffer("W", W)
        self.register_buffer("Win", (torch.rand(VOCAB, hidden, generator=g) * 2 - 1) * 0.5)
        self.leak = leak
        self.hidden = hidden
        self.out = nn.Linear(hidden, VOCAB)          # the only stored parameters

    def forward(self, x, state=None):
        B, T = x.shape
        h = torch.zeros(B, self.hidden, device=x.device) if state is None else state
        outs = []
        inp = self.Win[x]                                  # (B,T,H)
        for t in range(T):
            pre = inp[:, t] + h @ self.W
            h = (1 - self.leak) * h + self.leak * torch.tanh(pre)
            outs.append(h)
        return self.out(torch.stack(outs, 1)), h.detach()

    def stored_parameters(self) -> int:
        return sum(p.numel() for p in self.out.parameters()) + 1   # +1 for the seed


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------
def count(m: nn.Module) -> int:
    if hasattr(m, "stored_parameters"):
        return m.stored_parameters()
    return sum(p.numel() for p in m.parameters())


def fit_size(build, lo: int, hi: int, budget: int) -> int:
    """Largest size whose stored parameters fit the byte budget at int8."""
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if count(build(mid)) <= budget:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def stream(records: list[bytes]) -> torch.Tensor:
    return torch.frombuffer(bytearray(b"".join(records)), dtype=torch.uint8).long()


def batches(arr, batch, seq, steps, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, arr.numel() - seq - 1, (batch,), generator=g)
        x = torch.stack([arr[j:j + seq] for j in i])
        y = torch.stack([arr[j + 1:j + seq + 1] for j in i])
        yield x, y


@torch.no_grad()
def bits_per_byte_per_doc(model, docs, seq=256):
    # a fixed-context model must be scored in chunks no longer than its context
    seq = min(seq, getattr(model, "ctx", seq))
    model.eval()
    out = []
    for d in docs:
        a = torch.frombuffer(bytearray(d), dtype=torch.uint8).long()
        if a.numel() < 2:
            continue
        x, y = a[:-1].unsqueeze(0), a[1:].unsqueeze(0)
        total = 0.0
        state = None
        for s in range(0, x.shape[1], seq):
            xs, ys = x[:, s:s + seq], y[:, s:s + seq]
            logits, state = model(xs, state)
            if isinstance(state, torch.Tensor):
                state = state.detach()
            total += F.cross_entropy(logits.reshape(-1, VOCAB), ys.reshape(-1),
                                     reduction="sum").item()
        out.append(total / y.numel() / math.log(2))
    model.train()
    return out


def train_model(model, arr, steps, batch, seq, lr, tag, log_every=200):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    t0 = time.time()
    for step, (x, y) in enumerate(batches(arr, batch, seq, steps, 0), 1):
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % log_every == 0 or step == steps:
            print(f"    [{tag}] step {step}/{steps} train {loss.item()/math.log(2):.4f} "
                  f"bits/byte ({time.time()-t0:.0f}s)", flush=True)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="r1/data/records/train.jsonl")
    ap.add_argument("--test", default="r1/data/records/test.jsonl")
    ap.add_argument("--budget", type=int, default=86_820)
    ap.add_argument("--train-mb", type=float, default=6.0)
    ap.add_argument("--test-docs", type=int, default=120)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="r1/runs/baselines.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    train_recs = load_records(args.train)
    arr = stream(train_recs)[:int(args.train_mb * 1e6)]
    test_docs = load_records(args.test)[:args.test_docs]
    print(f"[baselines] train {arr.numel()/1e6:.2f} MB, test {len(test_docs)} docs")
    print(f"[baselines] storage cap {args.budget:,} bytes at int8\n")

    specs = []

    h = fit_size(lambda n: ByteGRU(n, 32), 4, 512, args.budget)
    specs.append(("byte GRU", ByteGRU(h, 32), f"hidden {h}, embed 32", 3e-3))

    ctx = max(args.seq, 256)
    d = fit_size(lambda n: ByteTransformer(max(n - n % 4, 4), 4, ctx),
                 8, 256, args.budget)
    d = max(d - d % 4, 4)
    specs.append(("byte Transformer", ByteTransformer(d, 4, ctx),
                  f"d_model {d}, 1 layer, 4 heads, ctx {ctx}", 3e-3))

    r = fit_size(lambda n: Reservoir(n), 16, 400, args.budget)
    specs.append(("reservoir (echo state)", Reservoir(r), f"hidden {r}, seed-generated", 5e-3))

    rows = []
    for name, model, desc, lr in specs:
        stored = count(model)
        total = sum(p.numel() for p in model.parameters())
        print(f"[{name}] {desc} | stored {stored:,} params "
              f"= {stored:,} B int8 / {total*4:,} B fp32")
        steps = args.steps if "reservoir" not in name else args.steps // 3
        seq = args.seq if "reservoir" not in name else 64
        train_model(model, arr, steps, args.batch, seq, lr, name)
        per_doc = bits_per_byte_per_doc(model, test_docs)
        pt, lo, hi = bootstrap_ci(per_doc, [len(d) for d in test_docs])
        rows.append({"model": name, "config": desc,
                     "stored_params": stored, "stored_bytes_int8": stored,
                     "all_params": total, "bytes_fp32": total * 4,
                     "fits_cap_int8": stored <= args.budget,
                     "train_steps": steps,
                     "bits_per_byte": round(pt, 4),
                     "ci95": [round(lo, 4), round(hi, 4)]})
        print(f"  -> {pt:.4f} bits/byte [{lo:.4f}, {hi:.4f}]\n", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"storage_cap_bytes": args.budget, "train_bytes": int(arr.numel()),
         "test_docs": len(test_docs), "baselines": rows}, indent=2))
    print(f"[baselines] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
