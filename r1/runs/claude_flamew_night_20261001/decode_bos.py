"""Flame-W sentence decoder "w-sent-bos.5f" (experimental, 2026-09-30).

Same beam search as word_decode.py's w-sent-mmi.3, with two changes:
  * the anti-generic (MMI) term compares against NO context (BOS) instead of
    the prompt's last word, so sentences MICA would say after any prompt
    are penalised: score = (log p(cont | prompt) - 0.5 log p(cont | BOS)) / len;
  * a sentence may not end on a word that needs a continuation ("the", "as",
    "other", ...).
Blind holdout check (20 prompts, one judge): full40 15% -> 20% at least partly
useful; the lr 0.03 continuation 15% -> 25%. Small and noisy; not yet promoted.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "claude_flame_word_20260928"))
import word_decode as D  # noqa: E402
import word_eval as E  # noqa: E402

NOEND = set(D.DANGLING) | {"other", "such", "each", "both", "either", "neither", "another", "every"}
BOS5F = dict(name="w-sent-bos.5f", lam=0.5, top=6, beam=6, max_words=20, min_words=3,
             must_end=True, no_dangling=True)


class BosDecoder(D.WordDecoder):
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
                picks = 0
                for k in D.np.argsort(-lp, kind="stable"):
                    i = int(self.allowed[k])
                    if i in banned:
                        continue
                    if not self.is_word[i] and (h["words"] == 0 or (h["ids"] and not self.is_word[h["ids"][-1]])):
                        continue
                    nh = {"ids": h["ids"] + [i], "lp": h["lp"] + float(lp[k]),
                          "words": h["words"] + int(self.is_word[i]), "done": i in self.ends}
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
        cands = finished or [h for h in beam if h["ids"]]

        def ends_ok(h):
            ws = [t for t in h["ids"] if self.is_word[t]]
            return bool(ws) and self.tok(ws[-1]) not in NOEND
        cands = [h for h in cands if ends_ok(h)] or cands
        if not cands:
            return {"continuation": "", "bits_per_token": None, "candidates": 0,
                    "seconds": time.perf_counter() - t0}
        for h in cands:
            h["lpg"] = self._lp_after([], h["ids"])
        best = min(cands, key=lambda h: -(h["lp"] - self.lam * h["lpg"]) / len(h["ids"]))
        ids = best["ids"]
        while ids and ids[-1] == self.comma:
            ids = ids[:-1]
        cap = not prompt.strip() or prompt.rstrip()[-1:] in ".!?\n"
        return {"continuation": self.text(ids, cap), "bits_per_token": -best["lp"] / len(best["ids"]) / D.LN2,
                "candidates": len(cands), "seconds": time.perf_counter() - t0}
