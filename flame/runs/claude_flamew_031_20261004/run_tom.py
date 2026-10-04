#!/usr/bin/env python3
"""Tiny Theory of Mind (AxiomicLabs/Tiny_Theory_of_Mind, 2,000 items) for MICA Flame-W 0.3.1.

The context is fed in full. Each of the 4 endings is scored teacher-forced from a copy of that
state. The choice is the ending with the highest length-normalized log-likelihood. Both
normalizations are reported:
  acc_norm  = per-character (sum of ln p / characters of the ending; the usual lm-eval acc_norm)
  acc_word  = per-word-token (mean ln p over the ending's word tokens)

    python flame/runs/claude_flamew_031_20261004/run_tom.py                     # 0.3.1: automaton + memory + answer mode
    python flame/runs/claude_flamew_031_20261004/run_tom.py --config memory     # automaton + memory (plain language model)
    python flame/runs/claude_flamew_031_20261004/run_tom.py --config automaton  # the automaton alone
    options: --data tiny_theory_of_mind_2000.jsonl  --start 0 --stop 2000  --out tom_result.json
"""
import argparse, json, math, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "r1/runs/claude_flame_word_20260928"))
sys.path.insert(0, str(HERE))
import word_eval as E  # noqa: E402  (sets the word geometry, then imports the engine)
from word_eval import W  # noqa: E402
from memory import Memory, MemoryMicaWord  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["full", "memory", "automaton"], default="full")
    ap.add_argument("--data"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int, default=2000)
    ap.add_argument("--out", default="tom_result.json")
    a = ap.parse_args()
    if not a.data:
        from huggingface_hub import hf_hub_download
        a.data = hf_hub_download("AxiomicLabs/Tiny_Theory_of_Mind", "tiny_theory_of_mind_2000.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()][a.start:a.stop]
    model = E.load_model(f"mica:{ROOT / 'flame/runs/claude_flamew_b740_20261001/train'}")
    if a.config != "automaton":
        model = MemoryMicaWord(model, Memory(HERE / "memory.npz",
                                             HERE / "answer_mode.npz" if a.config == "full" else None))
    vocab = W.Vocab.load(ROOT / "r1/data/word/vocab.json")
    ids_of = lambda s: [W.SEP if t == "<sep>" else vocab.index.get(t, W.oov_id(t)) for t in W.tokenize(s)]
    ok_c = ok_w = 0
    out, t0 = [], time.time()
    for k, r in enumerate(rows):
        st = model.start()
        for t in ids_of(r["ctx"]):
            st = model.feed(st, t)
        ends = []
        for e in r["endings"]:
            ei = ids_of(e)
            s2, lp = model.fork(st), 0.0
            for t in ei:
                lp += float(model.logp(s2, [t])[0]); s2 = model.feed(s2, t)
            ends.append({"sum_lnp": lp, "tokens": len(ei), "chars": len(e)})
        by_c = max(range(4), key=lambda i: ends[i]["sum_lnp"] / max(ends[i]["chars"], 1))
        by_w = max(range(4), key=lambda i: ends[i]["sum_lnp"] / ends[i]["tokens"] if ends[i]["tokens"] else -math.inf)
        lab = int(r["label"])
        ok_c += by_c == lab; ok_w += by_w == lab
        out.append({"ind": r["ind"], "label": lab, "pred_char": by_c, "pred_word": by_w,
                    "topic": r.get("metadata", {}).get("topic"), "endings": ends})
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(rows)}  acc_norm {ok_c / (k + 1):.4f}  acc_word {ok_w / (k + 1):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    n = len(rows)
    json.dump({"model": "MICA Flame-W 0.3.1", "config": a.config, "n": n,
               "acc_norm": ok_c / n, "acc_word": ok_w / n, "rows": out}, open(a.out, "w"))
    print(f"config {a.config}: acc_norm (per character) {ok_c / n:.4f}   acc_word (per word token) {ok_w / n:.4f}   n={n}, chance 0.25")


if __name__ == "__main__":
    main()
