#!/usr/bin/env python3
"""Turn plain text (one record per line) into MICA evaluation records.

    python r1/data/text_to_records.py my.txt --words my_words.jsonl --bytes my_bytes.jsonl

--words: Flame-W token records (the vocabulary in r1/data/word/vocab.json),
         for `word_eval.py bits --tokens ...`.
--bytes: UTF-8 byte records (hex, <= 256 bytes), for `word_eval.py nextword
         --set ...` and for the byte models.
Empty lines are skipped. Use text that none of the models trained on.
"""
import argparse
from pathlib import Path

import build_word_corpus as W
from make_records import utf8_safe_cut

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("text", type=Path)
ap.add_argument("--words", type=Path)
ap.add_argument("--bytes", type=Path)
a = ap.parse_args()
lines = [l.strip() for l in a.text.read_text(encoding="utf-8").splitlines() if l.strip()]
if a.words:
    vocab = W.Vocab.load(Path(__file__).resolve().parent / "word" / "vocab.json")
    recs = [W.pack(vocab.encode(t)) for t in (W.tokenize(l) for l in lines) if t]
    a.words.write_text("".join(r + "\n" for r in recs))
    print(f"{len(recs)} word records -> {a.words}")
if a.bytes:
    recs = [(b := l.encode("utf-8"))[:utf8_safe_cut(b, 256)].hex() for l in lines]
    a.bytes.write_text("".join(r + "\n" for r in recs))
    print(f"{len(recs)} byte records -> {a.bytes}")
