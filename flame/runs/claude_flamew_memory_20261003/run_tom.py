#!/usr/bin/env python3
"""Tiny Theory-of-Mind (AxiomicLabs/Tiny_Theory_of_Mind, 2,000 rows) for Flame-W.

Context fed in full; each of the 4 endings scored teacher-forced from a copy of that state;
choice = highest mean ln p per ending word token (length-normalized).

    python flame/runs/claude_flamew_memory_20261003/run_tom.py                  # Flame-W + memory
    python flame/runs/claude_flamew_memory_20261003/run_tom.py --no-memory      # the automaton alone
    options: --data tiny_theory_of_mind_2000.jsonl  --start 0 --stop 2000  --out tom.json
"""
import argparse, json, math, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "r1/runs/claude_flame_word_20260928"))
sys.path.insert(0, str(HERE))
import word_eval as E  # noqa: E402
from word_eval import W  # noqa: E402
from memory import Memory, MemoryMicaWord  # noqa: E402

MODEL = ROOT / "flame/runs/claude_flamew_b740_20261001/train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int, default=2000)
    ap.add_argument("--no-memory", action="store_true"); ap.add_argument("--out", default="tom_result.json")
    a = ap.parse_args()
    if not a.data:
        from huggingface_hub import hf_hub_download
        a.data = hf_hub_download("AxiomicLabs/Tiny_Theory_of_Mind", "tiny_theory_of_mind_2000.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()][a.start:a.stop]
    model = E.load_model(f"mica:{MODEL}")
    if not a.no_memory:
        model = MemoryMicaWord(model, Memory(HERE / "memory.npz"))
    vocab = W.Vocab.load(ROOT / "r1/data/word/vocab.json")
    ids_of = lambda s: [W.SEP if t == "<sep>" else vocab.index.get(t, W.oov_id(t)) for t in W.tokenize(s)]
    ok, out, t0 = 0, [], time.time()
    for k, r in enumerate(rows):
        st = model.start()
        for t in ids_of(r["ctx"]):
            st = model.feed(st, t)
        means = []
        for e in r["endings"]:
            ei = ids_of(e)
            s2, lp = model.fork(st), 0.0
            for t in ei:
                lp += float(model.logp(s2, [t])[0]); s2 = model.feed(s2, t)
            means.append(lp / len(ei) if ei else -math.inf)
        pred = max(range(4), key=lambda i: means[i])
        ok += pred == int(r["label"])
        out.append({"ind": r["ind"], "label": int(r["label"]), "pred": pred, "topic": r["metadata"].get("topic")})
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(rows)} accuracy {ok / (k + 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
    acc = ok / len(rows)
    json.dump({"accuracy": acc, "n": len(rows), "memory": not a.no_memory, "rows": out}, open(a.out, "w"))
    print(f"ACCURACY {acc:.4f} (n={len(rows)}, chance 0.25)")


if __name__ == "__main__":
    main()
