#!/usr/bin/env python3
"""One table: a Flame-W checkpoint's development metrics next to the letter
Flame refit (the P1 bar) and the same-token word n-gram references.

    python summarize_flamew.py RESULTS_DIR [REFERENCES_JSON]

RESULTS_DIR is what lanes/flamew_eval.sh wrote (nw_*, cp_*, bits_*, ctx_*).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def main() -> int:
    rd = Path(sys.argv[1])
    ref = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else
                          "/home/claude/mica/wordw/results/references.json").read_text())
    row = {}
    for s, name in (("chatdev", "chat_dev1000"), ("everydaydev", "everyday_dev_fresh1000")):
        nw, cp, b = (load(rd / f"{k}_{s}.json") for k in ("nw", "cp", "bits"))
        row[s] = {"nw": nw and nw["top1"], "cp": cp and cp["top1"],
                  "saved": cp and cp["letters_saved_share"], "bits": b and b["bits_per_token"]}
    ctx = {s: load(rd / f"ctx_{s}.json") for s in ("chat", "everyday")}
    bar = ref["letter_flame_refit_6cdcd921"]
    lines = ["| System | Next word chat / everyday | Mean | Completion chat / everyday | Bits/token chat / everyday |",
             "|---|---:|---:|---:|---:|"]
    m = [row[s]["nw"] for s in ("chatdev", "everydaydev")]
    mean = sum(m) / 2 if None not in m else None
    bits = [row[s]["bits"] for s in ("chatdev", "everydaydev")]
    lines.append(f"| **{rd.name}** | {pct(m[0])} / {pct(m[1])} | {pct(mean)} | "
                 f"{pct(row['chatdev']['cp'])} / {pct(row['everydaydev']['cp'])} | "
                 + " / ".join("—" if b is None else f"{b:.2f}" for b in bits) + " |")
    lb = [bar["chat_dev1000"]["next_word_top1"], bar["everyday_dev_fresh1000"]["next_word_top1"]]
    lc = [bar["chat_dev1000"].get("completion_top1"), bar["everyday_dev_fresh1000"].get("completion_top1")]
    lines.append(f"| letter Flame refit 6cdcd921 (P1 bar) | {pct(lb[0])} / {pct(lb[1])} | "
                 f"{pct(sum(lb) / 2)} | {pct(lc[0])} / {pct(lc[1])} | — |")
    for tag, r in ref["references"].items():
        c, e = r["chatdev"], r["everydaydev"]
        lines.append(f"| word n-gram {tag} ({r['file_bytes'] / 1e6:.1f} MB) | {pct(c['next_word_top1'])} / "
                     f"{pct(e['next_word_top1'])} | {pct((c['next_word_top1'] + e['next_word_top1']) / 2)} | "
                     f"{pct(c['completion_top1'])} / {pct(e['completion_top1'])} | "
                     f"{c['bits_per_token']:.2f} / {e['bits_per_token']:.2f} |")
    print("\n".join(lines))
    if mean is not None:
        gate = mean >= sum(lb) / 2
        print(f"\nP1 gate (mean next-word top-1 >= letter refit {100 * sum(lb) / 2:.2f}%): "
              f"{'PASS' if gate else 'FAIL'} ({100 * mean:.2f}%)")
    for s, c in ctx.items():
        if c:
            print(f"context use {s}_long600: " + ", ".join(
                f"use_w({k}) {c[f'use_w({k})']:+.4f}" for k in (0, 1, 2, 4, 8) if c.get(f"use_w({k})") is not None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
