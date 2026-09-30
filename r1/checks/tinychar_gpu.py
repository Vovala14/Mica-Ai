#!/usr/bin/env python3
"""Baseline: tiny-character-transformer (github.com/maxpolaczuk/tiny-character-
transformer, downloaded by r1/data/fetch_external.py), its TinyGPT model code
used unchanged, trained on MICA's training text on the GPU and scored with
MICA's protocol on val1000 (bits per target over the bytes and one end of
record per record; each record starts from BOS).

Differences from the repo, all forced by the data:
  vocabulary 258 symbols -- bytes 0-255, BOS 256, EOS 257 -- instead of Tiny
             Shakespeare's 65 characters (the module's vocab_size is set
             before the model is built); BOS is never a target, so its logit
             is masked out, as MICA's softmax is over 256 bytes + EOS
  block      257: BOS plus a whole 256-byte record
  data       each record is one sequence, [BOS] + bytes -> bytes + [EOS]
Kept from the repo's 1M configuration: d_model 128, 4 heads, d_ff 512, 5
layers; AdamW lr 5e-4, 200 warm-up steps, cosine decay to lr/10, batch 64.

    python r1/checks/tinychar_gpu.py --corpus everyday --minutes 45 \
        --out r1/runs/claude_tinychar_20260927/everyday.json

A reference for what a 4 MB model of another kind reaches on the same text
and records; it changes nothing in MICA.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import io
import json
import math
import random
import struct
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import R1, record_set, beat, utf8_stdout, hard_exit  # noqa: E402

BOS, EOS, V = 256, 257, 258
ZIP = R1 / "data/external/tinychar/tiny-character-transformer.zip"
SRC = R1 / "data/external/tinychar/src"


def read_bin_gz(path):
    out = []
    with gzip.open(path, "rb") as f:
        while True:
            h = f.read(2)
            if not h:
                break
            n = struct.unpack("<H", h)[0]
            out.append(f.read(n))
    return out


def repo_module():
    SRC.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            base = name.split("/")[-1]
            if base in ("model.py", "input.txt"):
                (SRC / base).write_bytes(z.read(name))
    spec = importlib.util.spec_from_file_location("tinychar_model", SRC / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.vocab_size = V
    return mod


def main() -> int:
    utf8_stdout()
    import numpy as np
    import torch
    import torch.nn.functional as F
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["bulk", "everyday"], default="bulk")
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337)
    rng = random.Random(1337)
    mod = repo_module()
    cfg = mod.Config(name="1M-bytes", block_size=257, d_model=128, n_head=4,
                     d_ff=512, n_layer=5)
    model = mod.TinyGPT(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    exp = R1 / "data/export"
    if a.corpus == "bulk":
        train = []
        for k in range(4):
            train += read_bin_gz(exp / f"bulk_train256.part{k}.bin.gz")
        evals = {"val1000": [r[:256] for r in record_set("val1000", R1)]}
    else:
        train = read_bin_gz(exp / "everyday_train.bin.gz")
        ev = read_bin_gz(exp / "everyday_val.bin.gz")
        clean = [bytes.fromhex(l.strip()) for l in
                 open(R1 / "data/eval_clean/val1000.jsonl") if l.strip()]
        # the clean set first: it is the one the MICA results are reported on
        evals = {"everyday_clean_val1000": clean, "everyday_val256": ev[:256]}
    print(f"[tinychar] {dev}, params {n_params:,}, {len(train):,} records ({a.corpus})",
          flush=True)

    def tensors(recs):
        L = max(len(r) for r in recs) + 1
        x = np.zeros((len(recs), L), np.int64)
        y = np.full((len(recs), L), -100, np.int64)
        for i, r in enumerate(recs):
            b = np.frombuffer(r, np.uint8).astype(np.int64)
            x[i, 0] = BOS
            x[i, 1:len(b) + 1] = b
            y[i, :len(b)] = b
            y[i, len(b)] = EOS
        return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

    def logits(x):
        lg, _ = model(x)
        lg = lg.clone()
        lg[..., BOS] = float("-inf")
        return lg

    @torch.no_grad()
    def bits(recs):
        model.eval()
        nats, n = 0.0, 0
        for i in range(0, len(recs), 64):
            x, y = tensors(recs[i:i + 64])
            nats += F.cross_entropy(logits(x).view(-1, V).float(), y.view(-1),
                                    ignore_index=-100, reduction="sum").item()
            n += int((y != -100).sum())
        model.train()
        return nats / n / math.log(2)

    LR, MIN_LR, WARMUP, BATCH = 5e-4, 5e-5, 200, 64
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    budget = a.minutes * 60
    t0 = time.time()
    total, step, hist = None, 0, []
    mon = next(iter(evals.values()))[:200]
    while True:
        step += 1
        if total is None and step == 51:
            per = (time.time() - t0) / 50
            total = int(budget / per)
            print(f"[tinychar] {per * 1000:.0f} ms/step -> {total} steps", flush=True)
        if total is not None and step > total:
            break
        T = total or 10**9
        lr = LR * step / WARMUP if step < WARMUP else MIN_LR + 0.5 * (LR - MIN_LR) * (
            1 + math.cos(math.pi * min(1.0, (step - WARMUP) / max(1, T - WARMUP))))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = tensors([train[rng.randrange(len(train))] for _ in range(BATCH)])
        loss = F.cross_entropy(logits(x).view(-1, V), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            b = bits(mon)
            hist.append({"step": step, "monitor_bits": b,
                         "minutes": round((time.time() - t0) / 60, 1)})
            print(f"[tinychar] step {step}/{T} monitor {b:.4f} bits "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            beat(f"tinychar {step}")
    res = {"repo": "maxpolaczuk/tiny-character-transformer", "config": "1M, 258-symbol vocab",
           "params": n_params, "fp32_bytes": 4 * n_params, "corpus": a.corpus,
           "steps": step - 1, "records_seen": (step - 1) * BATCH,
           "minutes": round((time.time() - t0) / 60, 1), "history": hist,
           "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}
    for name, recs in evals.items():
        res[name] = bits(recs)
        print(f"[tinychar] {name}: {res[name]:.4f} bits per target", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out.with_suffix(".pt"))
    out.write_text(json.dumps(res, indent=1))
    hard_exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
