#!/usr/bin/env python3
"""Train a local word-context generator; checkpoints are research artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))

import torch
import torch.nn.functional as F

from research_baselines.word_gru import WordGRU
from research_baselines.word_ngram import _tokens, TERMINAL

BOS, EOS, UNK = "<bos>", "<eos>", "<unk>"


def load_sources(paths: list[Path], limit: int):
    rows = []
    seen = set()
    for path in paths:
        accepted = 0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                sentence = json.loads(line)["text"]
                tokens = _tokens(sentence)
                words = sum(token not in ".!?,;:" for token in tokens)
                if not 5 <= words <= 35 or not tokens or tokens[-1] not in TERMINAL:
                    continue
                key = hashlib.sha256(" ".join(tokens).encode()).digest()
                if key in seen:
                    continue
                seen.add(key)
                rows.append((key, tokens))
                accepted += 1
                if accepted >= limit:
                    break
        print(f"[word-gru] {path}: {accepted} sentences", flush=True)
    return rows


def make_batch(sequences: list[list[int]], rng: random.Random,
               batch_size: int, device: torch.device):
    picked = rng.choices(sequences, k=batch_size)
    length = max(len(seq) for seq in picked) + 1
    inputs = torch.zeros((batch_size, length), dtype=torch.long)
    targets = torch.full((batch_size, length), -100, dtype=torch.long)
    for i, seq in enumerate(picked):
        inputs[i, :len(seq) + 1] = torch.tensor([0, *seq])
        targets[i, :len(seq) + 1] = torch.tensor([*seq, 1])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def validate(model: WordGRU, sequences: list[list[int]],
             device: torch.device, count: int = 512) -> float:
    model.eval()
    rng = random.Random(73)
    total, targets = 0.0, 0
    for _ in range(max(1, count // 32)):
        x, y = make_batch(sequences, rng, 32, device)
        logits, _ = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 y.reshape(-1), ignore_index=-100,
                                 reduction="sum").item()
        targets += int((y != -100).sum().item())
    model.train()
    return total / targets


def save_checkpoint(path: Path, model: WordGRU, optimizer,
                    step: int, rng: random.Random) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "rng": rng.getstate()}, tmp)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-per-source", type=int, default=200_000)
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if min(args.max_per_source, args.vocab_size, args.dim, args.layers,
           args.steps, args.batch, args.log_every) < 1:
        parser.error("all limits must be positive")
    if args.vocab_size < 100:
        parser.error("vocabulary is too small for this pilot")

    torch.set_num_threads(2)
    torch.manual_seed(27)
    # This Windows ROCm build fails while compiling MIOpen GRU kernels.
    # PyTorch's native recurrent path remains trainable on the same GPU.
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_sources(args.source, args.max_per_source)
    if not rows:
        raise ValueError("no usable sentences")
    train_rows = [tokens for key, tokens in rows if int.from_bytes(key[:4], "big") % 50]
    val_rows = [tokens for key, tokens in rows if not int.from_bytes(key[:4], "big") % 50]
    counts = Counter(token for row in train_rows for token in row)
    vocab = [BOS, EOS, UNK] + [token for token, _ in counts.most_common(args.vocab_size - 3)]
    word_id = {token: i for i, token in enumerate(vocab)}
    encode = lambda rows: [[word_id.get(token, 2) for token in row] for row in rows]
    train_sequences, val_sequences = encode(train_rows), encode(val_rows)
    config = {"sources": [str(path) for path in args.source],
              "source_sentences": len(rows), "train_sentences": len(train_sequences),
              "validation_sentences": len(val_sequences), "vocab_size": len(vocab),
              "dim": args.dim, "layers": args.layers, "seed": 27,
              "holdout_rule": "SHA256 normalized-token prefix mod 50 == 0"}
    vocab_path = args.out_dir / "vocab.json"
    if vocab_path.exists():
        prior = json.loads(vocab_path.read_text(encoding="utf-8"))
        if prior != {"config": config, "vocab": vocab}:
            raise ValueError("existing vocabulary differs from current corpus/settings")
    else:
        vocab_path.write_text(json.dumps({"config": config, "vocab": vocab},
                                         ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[word-gru] train={len(train_sequences)} val={len(val_sequences)} "
          f"vocab={len(vocab)} device={device}", flush=True)

    model = WordGRU(len(vocab), args.dim, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    rng = random.Random(27)
    checkpoint = args.out_dir / "last.pt"
    start_step = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = saved["step"]
        rng.setstate(saved["rng"])
        print(f"[word-gru] resumed step {start_step}", flush=True)
    started = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        x, y = make_batch(train_sequences, rng, args.batch, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(vocab)),
                               y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            val_nats = validate(model, val_sequences, device)
            save_checkpoint(checkpoint, model, optimizer, step, rng)
            print(f"[word-gru] step={step}/{args.steps} "
                  f"train_nats={loss.item():.3f} val_nats={val_nats:.3f} "
                  f"seconds={time.perf_counter()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
