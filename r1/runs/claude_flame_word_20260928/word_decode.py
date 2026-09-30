#!/usr/bin/env python3
"""Suggestions and sentences from a word-symbol model (Flame-W or a word
reference), in the benchmark's row format.

    python word_decode.py suggest  --model SPEC --vocab vocab.json --prompts P.jsonl --tag T --out F.jsonl
    python word_decode.py sentence --model SPEC --vocab vocab.json --prompts P.jsonl --tag T --out F.jsonl

suggest (w6-mmi.5, the word counterpart of the letter decoder v6-whole-mmi.5):
  a beam of 3 over tokens; at each step the 4 most probable allowed tokens
  (vocabulary words, "," and sentence ends); a hypothesis ends at ". ! ?" or
  after 3 words. Allowed: no out-of-vocabulary id, no <sep>, not the previous
  word again, no word pair already in the prompt or the hypothesis. Every
  hypothesis with at least 2 words is a candidate; the winner has the best
  (log p(continuation | prompt) - 0.5 log p(continuation | last prompt word))
  per token.
sentence (w-sent-mmi.3): the same search with a beam of 4, up to 20 words,
  ending only at a sentence end, MMI 0.3.

Text: each token's usual written form from vocab.json ("i" -> "I"), no space
before punctuation, and a capital after a sentence end.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import word_eval as E  # noqa: E402
from word_eval import W  # noqa: E402

LN2 = math.log(2)
MODES = {
    "suggest": dict(name="w6-mmi.5", lam=0.5, top=4, beam=3, max_words=3, min_words=2,
                    must_end=False),
    "sentence": dict(name="w-sent-mmi.3", lam=0.3, top=4, beam=4, max_words=20, min_words=3,
                     must_end=True),
    "suggest-nd": dict(name="w6-mmi.5-nd", lam=0.5, top=4, beam=3, max_words=3, min_words=2,
                       must_end=False, no_dangling=True),
}
ENDS = (".", "!", "?")
# a suggestion should not stop on a word that needs a continuation
DANGLING = frozenset("""a an the and or but nor so of to in on at by for with from into onto about
as than that if because when while my your his her their our its this these those some any
every no is are was were be been being am will would can could should shall may might must
do does did have has had i he she we they very too just not before after under over through
until since without within between behind around near during against toward towards across along
upon via per like unless whether whose then""".split())


class WordDecoder:
    def __init__(self, model, vocab: W.Vocab, name, lam, top, beam, max_words, min_words,
                 must_end, no_dangling=False):
        self.m, self.v = model, vocab
        self.no_dangling = no_dangling
        self.name, self.lam, self.top, self.beam = name, lam, top, beam
        self.max_words, self.min_words, self.must_end = max_words, min_words, must_end
        words = E.word_ids(vocab)
        self.ends = [vocab.index[p] for p in ENDS if p in vocab.index]
        self.comma = vocab.index.get(",")
        self.allowed = np.array(sorted(set(words.tolist()) | set(self.ends) |
                                       ({self.comma} if self.comma else set())))
        self.is_word = np.zeros(W.N_SYMBOLS, bool)
        self.is_word[words] = True

    def tok(self, i: int) -> str:
        return self.v.tokens[i - W.FIRST_WORD]

    def text(self, ids, capital: bool) -> str:
        out = ""
        for i in ids:
            t = self.v.surface[i - W.FIRST_WORD]
            if capital and self.is_word[i]:
                t = t[:1].upper() + t[1:]
                capital = False
            out += t if t in ".,!?;:" else " " + t
            capital = capital or t in ENDS
        return out

    def _dangles(self, ids) -> bool:
        return bool(ids) and self.is_word[ids[-1]] and self.tok(ids[-1]) in DANGLING

    def _undangle(self, h, ctx, extra: int = 2):
        """Extend a hypothesis that stops on a word needing a continuation
        by up to `extra` greedy tokens (a word or a sentence end)."""
        if not self._dangles(h["ids"]):
            return h
        st = self.m.start()
        for t in ctx + h["ids"]:
            st = self.m.feed(st, t)
        ids, lp = list(h["ids"]), h["lp"]
        hist = [t for t in ctx + ids if self.is_word[t]]
        for _ in range(extra):
            lps = self.m.logp(st, self.allowed)
            for k in np.argsort(-lps, kind="stable"):
                i = int(self.allowed[k])
                if i == self.comma or i == hist[-1] or (hist[-1], i) in set(zip(hist, hist[1:])):
                    continue
                break
            ids.append(i)
            lp += float(lps[k])
            if not self._dangles(ids):
                break
            hist.append(i)
            st = self.m.feed(st, i)
        return {**h, "ids": ids, "lp": lp}

    def _lp_after(self, ctx: list[int], cont: list[int]) -> float:
        st = self.m.start()
        for t in ctx:
            st = self.m.feed(st, t)
        lp = 0.0
        for t in cont:
            lp += float(self.m.logp(st, [t])[0])
            st = self.m.feed(st, t)
        return lp

    def complete(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        ctx = E.prefix_ids(self.v, prompt.encode("utf-8"))
        st = self.m.start()
        for t in ctx:
            st = self.m.feed(st, t)
        hist0 = [t for t in ctx if self.is_word[t]]
        beam = [{"ids": [], "lp": 0.0, "st": st, "words": 0, "done": False}]
        finished = []
        for _ in range(self.max_words * 2):
            new = []
            for h in beam:
                if h["done"]:
                    continue
                hist = hist0 + [t for t in h["ids"] if self.is_word[t]]
                prev = hist[-1] if hist else None
                banned = {b for a, b in zip(hist, hist[1:]) if a == prev}
                if prev is not None:
                    banned.add(prev)
                lp = self.m.logp(h["st"], self.allowed)
                order = np.argsort(-lp, kind="stable")
                picks = 0
                for k in order:
                    i = int(self.allowed[k])
                    if i in banned:
                        continue
                    if not self.is_word[i] and (h["words"] == 0 or
                                                (h["ids"] and not self.is_word[h["ids"][-1]])):
                        continue          # punctuation needs a word before it
                    nh = {"ids": h["ids"] + [i], "lp": h["lp"] + float(lp[k]),
                          "words": h["words"] + int(self.is_word[i]),
                          "done": i in self.ends}
                    nh["done"] = nh["done"] or nh["words"] >= self.max_words
                    if not nh["done"]:
                        nh["st"] = self.m.feed(self.m.fork(h["st"]), i)
                    new.append(nh)
                    if nh["words"] >= self.min_words and (not self.must_end or i in self.ends):
                        finished.append(nh)
                    picks += 1
                    if picks >= self.top:
                        break
            beam = sorted(new, key=lambda h: -h["lp"])[:self.beam]
            if not any(not h["done"] for h in beam):
                break
        if self.no_dangling:
            finished = [self._undangle(h, ctx) for h in finished]
            kept = [h for h in finished if not self._dangles(h["ids"])]
            finished = kept or finished
        cands = finished or [h for h in beam if h["ids"]]
        if not cands:
            return {"continuation": "", "bits_per_token": None, "candidates": 0,
                    "seconds": time.perf_counter() - t0}
        if self.lam:
            generic = ctx[-1:] if ctx else []
            for h in cands:
                h["lpg"] = self._lp_after(generic, h["ids"])
            key = lambda h: -(h["lp"] - self.lam * h["lpg"]) / len(h["ids"])
        else:
            key = lambda h: -h["lp"] / len(h["ids"])
        best = min(cands, key=key)
        ids = best["ids"]
        while ids and ids[-1] == self.comma:
            ids = ids[:-1]
        cap = not prompt.strip() or prompt.rstrip()[-1:] in ".!?\n"
        return {"continuation": self.text(ids, cap), "bits_per_token": -best["lp"] / len(best["ids"]) / LN2,
                "candidates": len(cands), "seconds": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=list(MODES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vocab = W.Vocab.load(a.vocab)
    model = E.load_model(a.model)
    kw = dict(MODES[a.mode])
    dec = WordDecoder(model, vocab, **kw)
    items = [json.loads(l) for l in open(a.prompts, encoding="utf-8") if l.strip()]
    rows = []
    t0 = time.time()
    for it in items:
        r = dec.complete(it["prompt"])
        rows.append({"system": f"{a.tag}|{dec.name}", "decoding": dec.name, **it,
                     "continuation": r["continuation"],
                     "confidence_bits_per_token": r["bits_per_token"],
                     "seconds": r["seconds"], "candidates": r["candidates"]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{dec.name}: {len(rows)} prompts, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
