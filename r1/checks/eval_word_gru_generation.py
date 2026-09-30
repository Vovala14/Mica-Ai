#!/usr/bin/env python3
"""Archived GRU comparison; never use as MICA's release decoder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
from eval_word_ngram_generation import exact_training_matches, normal, save, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--word-run", type=Path, required=True)
    parser.add_argument("--train-text", type=Path, action="append", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=14)
    parser.add_argument("--mica-weight", type=float, default=0.35)
    args = parser.parse_args()
    if min(args.seeds, args.max_words) < 1 or args.mica_weight < 0:
        parser.error("invalid generation settings")

    info = json.loads((args.run / "run_info.json").read_text(encoding="utf-8"))
    os.environ.update({key: str(value) for key, value in info["env"].items()})
    import torch
    from mica_r1 import serialize
    from mica_r1.suggest import MicaScorer
    from research_baselines.word_gru import WordGRU, generate

    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = False
    payload_vocab = json.loads((args.word_run / "vocab.json").read_text(encoding="utf-8"))
    vocab = payload_vocab["vocab"]
    config = payload_vocab["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WordGRU(len(vocab), config["dim"], config["layers"]).to(device)
    checkpoint = torch.load(args.word_run / "last.pt", map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scorer = MicaScorer(serialize.load(args.run / "best.mica"))
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))["rows"]
    settings = {"mica_sha256": sha(args.run / "best.mica"),
                "word_checkpoint_sha256": sha(args.word_run / "last.pt"),
                "word_step": checkpoint["step"],
                "vocab_sha256": sha(args.word_run / "vocab.json"),
                "generator_sha256": sha(R1 / "research_baselines/word_gru.py"),
                "prompts_sha256": sha(args.prompts),
                "train_sha256": [sha(path) for path in args.train_text],
                "seeds": args.seeds, "max_words": args.max_words,
                "mica_weight": args.mica_weight}
    if args.out.exists():
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        if payload.get("config") != settings:
            raise ValueError("existing output has different configuration")
    else:
        payload = {"config": settings, "rows": []}
    jobs = [(item["id"], item["prompt"], seed)
            for item in prompts for seed in range(args.seeds)]
    for row, job in zip(payload["rows"], jobs):
        if (row["id"], row["prompt"], row["seed"]) != job:
            raise ValueError("existing output has different prompt order")
    print(f"[gru-gen] step={checkpoint['step']} resume={len(payload['rows'])}/"
          f"{len(jobs)} device={device}", flush=True)
    for id_, prompt, seed in jobs[len(payload["rows"]):]:
        start = time.perf_counter()
        sample = generate(model, vocab, scorer, prompt, seed=seed,
                          max_words=args.max_words,
                          mica_weight=args.mica_weight)
        payload["rows"].append({"id": id_, "prompt": prompt,
                                "seed": seed, "text": sample.text,
                                "continuation": sample.continuation,
                                "complete": sample.complete,
                                "generated_words": sample.generated_words,
                                "seconds": time.perf_counter() - start})
        save(args.out, payload)
        print(f"[gru-gen] {len(payload['rows'])}/{len(jobs)} "
              f"{sample.text!r}", flush=True)
    found = set()
    texts = {row["text"] for row in payload["rows"]}
    for path in args.train_text:
        found.update(exact_training_matches(path, texts))
    for row in payload["rows"]:
        row["exact_train_match"] = normal(row["text"]) in found
    payload["summary"] = {
        "outputs": len(payload["rows"]),
        "distinct": len(texts),
        "model_completed": sum(row["complete"] for row in payload["rows"]),
        "exact_train_matches": sum(row["exact_train_match"]
                                   for row in payload["rows"]),
        "generation_seconds": sum(row["seconds"] for row in payload["rows"]),
    }
    save(args.out, payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
