#!/usr/bin/env python3
"""Word-symbol corpora for MICA Flame-W: the same cellular automaton, with
one symbol per word instead of one per byte (owner decision, 28 Sep 2026:
Flame goes word-level, Ember stays letter-level).

    python r1/data/build_word_corpus.py                 # vocabulary + token files
    python r1/data/build_word_corpus.py --check         # coverage report only
    python r1/data/build_word_corpus.py --export-u16 4  # also compact train shards

Tokens (``tokenize``): curly quotes become straight ones; a token is a run of
letters or digits with optional inner apostrophes ("don't", "it's"), or one
other non-space character; a line break (the turn break of a dialogue pair)
is <sep>. Each token is casefolded. vocab.json keeps each word's usual
written form ("i" -> "I", "paris" -> "Paris") for showing suggestions.

Alphabet (N_SYMBOLS = 16,384; the engine takes it from MICA_SYMBOLS):
  id 0                  <sep>
  ids 1..256            <oov:0>..<oov:255>: a token outside the vocabulary,
                        by a stable hash of its text, so the context keeps a
                        little identity; never suggested
  ids 257..16381        16,125 vocabulary tokens, most frequent first (ties
                        by text), counted on the mix A TRAINING split only
  id 16382 / 16383      BOS / EOS (N_SYMBOLS - 2 / N_SYMBOLS - 1; the byte
                        engine's 256 / 257)

Records: little-endian uint16 ids, hex-encoded, one record per line (the
trainer's hex-record format, four hex digits per token), cut to the first
MAX_TOKENS tokens. Records that tokenize to nothing are dropped.

Outputs (under r1/data/word/):
  vocab.json                  alphabet, tokenizer, counts, written forms
  v02a/train.jsonl, val.jsonl mix A train / val as tokens
  v02a/train.partK.u16.gz     (--export-u16 N) the same train records as N
                              gzip shards, each record a <u16 byte length>
                              and its ids, for the cloud n-gram references
  eval/<name>.jsonl           every development, clean and probe set as
                              tokens; the byte files stay the reference for
                              next-word positions
  manifest.json               SHA-256 of inputs and outputs, counts, OOV rates

Deterministic: the same inputs give byte-identical outputs. The sealed test
(r1/data/eval_clean/test1000.jsonl) is never opened.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R1 = HERE.parent
OUT = HERE / "word"
MIX = HERE / "mix" / "v02a"

N_SYMBOLS = 16384
N_OOV = 256
SEP, OOV0 = 0, 1
FIRST_WORD = OOV0 + N_OOV                  # 257
BOS, EOS = N_SYMBOLS - 2, N_SYMBOLS - 1    # 16382, 16383
N_VOCAB = BOS - FIRST_WORD                 # 16,125
MAX_TOKENS = 64

_REFIT = R1 / "runs" / "codex_flame_refit_20260928_retry2"
EVAL_SETS = {
    "chat_dev1000": HERE / "chat" / "dev1000.jsonl",
    "everyday_dev_fresh1000": R1 / "runs" / "codex_flame_scaling_20260928" / "dev_fresh1000.jsonl",
    "chat_dev2_1000": _REFIT / "chat_dev2_1000.jsonl",
    "everyday_dev2_1000": _REFIT / "everyday_dev2_1000.jsonl",
    "chat_val1000_clean": HERE / "eval_clean" / "chat_val1000.jsonl",
    "everyday_val1000_clean": HERE / "eval_clean" / "val1000.jsonl",
    "chat_long600": HERE / "ctxprobe" / "chat_long600.jsonl",
    "everyday_long600": HERE / "ctxprobe" / "everyday_long600.jsonl",
}
SEALED = "test1000.jsonl"

_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*|\n|[^\w\s]|_")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'",
                         "“": '"', "”": '"'})
_SENTENCE_START = {None, "<sep>", ".", "!", "?", '"', ":", ";", "-", "(", "*"}


def surface_tokens(text: str) -> list[str]:
    """Tokens as written (case kept); a line break is <sep>."""
    return ["<sep>" if t == "\n" else t
            for t in _TOKEN.findall(text.replace("\r", "").translate(_QUOTES))]


def tokenize(text: str) -> list[str]:
    """Casefolded word and punctuation tokens; a line break is <sep>."""
    return [t if t == "<sep>" else t.casefold() for t in surface_tokens(text)]


def record_text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def oov_id(token: str) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return OOV0 + int.from_bytes(h, "little") % N_OOV


def pack(ids: list[int]) -> str:
    return struct.pack(f"<{len(ids)}H", *ids).hex()


def unpack(line: str) -> list[int]:
    b = bytes.fromhex(line.strip())
    return list(struct.unpack(f"<{len(b) // 2}H", b))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hex_lines(path: Path):
    """Yield each non-empty record of a hex-record file as bytes."""
    if path.name == SEALED:
        raise ValueError("the sealed test set is never opened")
    with open(path, encoding="ascii") as f:
        for line in f:
            if line.strip():
                yield bytes.fromhex(line.strip())


class Vocab:
    def __init__(self, tokens: list[str], surface: list[str] | None = None):
        assert len(tokens) <= N_VOCAB and len(set(tokens)) == len(tokens)
        assert "<sep>" not in tokens
        self.tokens = tokens
        self.surface = surface or list(tokens)
        self.index = {t: FIRST_WORD + i for i, t in enumerate(tokens)}

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        v = json.loads(Path(path).read_text(encoding="utf-8"))
        assert v["n_symbols"] == N_SYMBOLS and v["first_word"] == FIRST_WORD
        return cls(v["tokens"], v.get("surface"))

    def encode(self, toks: list[str]) -> list[int]:
        return [SEP if t == "<sep>" else self.index.get(t, oov_id(t))
                for t in toks[:MAX_TOKENS]]

    def decode(self, ids) -> list[str]:
        out = []
        for i in map(int, ids):
            if i == SEP:
                out.append("<sep>")
            elif OOV0 <= i < FIRST_WORD:
                out.append(f"<oov:{i - OOV0}>")
            elif FIRST_WORD <= i < BOS:
                out.append(self.tokens[i - FIRST_WORD])
            else:
                out.append("<bos>" if i == BOS else "<eos>")
        return out


def count_train(path: Path):
    """Token counts, and the written form of each token where it does not
    start a sentence or a quotation (a capital there is not the word's own)."""
    cnt, mid, caps = collections.Counter(), collections.Counter(), collections.Counter()
    n_rec = n_tok = 0
    for raw in hex_lines(path):
        prev = None
        toks = surface_tokens(record_text(raw))[:MAX_TOKENS]
        for s in toks:
            if s == "<sep>":
                prev = s
                continue
            low = s.casefold()
            cnt[low] += 1
            if prev not in _SENTENCE_START:
                mid[low] += 1
                if s != low:
                    caps[(low, s)] += 1
            prev = low
        n_rec += 1
        n_tok += len(toks)
    return cnt, mid, caps, n_rec, n_tok


def build_vocab(cnt, mid=None, caps=None) -> Vocab:
    ranked = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    tokens = [t for t, _ in ranked[:N_VOCAB]]
    best = {}
    for (low, s), c in (caps or {}).items():
        if low not in best or (-c, s) < (-best[low][1], best[low][0]):
            best[low] = (s, c)
    surface = []
    for t in tokens:
        s, c = best.get(t, (t, 0))
        surface.append(s if mid and 3 * c > 2 * mid[t] else t)   # capitalised in > 2/3
    return Vocab(tokens, surface)


def convert(src: Path, dst: Path, vocab: Vocab, parts: int = 0, n_records: int = 0) -> dict:
    """Write src's records as token records; with parts > 0 also write the
    same records as contiguous gzip shards next to dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shards = [gzip.GzipFile(dst.parent / f"{dst.stem}.part{k}.u16.gz", "wb", compresslevel=6,
                            mtime=0) for k in range(parts)]
    n_rec = n_tok = n_oov = n_cut = n_empty = 0
    try:
        with open(dst, "w", encoding="ascii", newline="\n") as g:
            for raw in hex_lines(src):
                toks = tokenize(record_text(raw))
                if not toks:
                    n_empty += 1
                    continue
                n_cut += len(toks) > MAX_TOKENS
                ids = vocab.encode(toks)
                g.write(pack(ids) + "\n")
                if parts:
                    k = min(parts - 1, n_rec * parts // max(1, n_records))
                    shards[k].write(struct.pack(f"<H{len(ids)}H", 2 * len(ids), *ids))
                n_rec += 1
                n_tok += len(ids)
                n_oov += sum(OOV0 <= i < FIRST_WORD for i in ids)
    finally:
        for s in shards:
            s.close()
    info = {"records": n_rec, "tokens": n_tok, "oov_rate": round(n_oov / max(1, n_tok), 5),
            "records_cut": n_cut, "records_empty": n_empty, "sha256": sha256_file(dst)}
    if parts:
        info["parts"] = {p.name: sha256_file(p) for p in
                         (dst.parent / f"{dst.stem}.part{k}.u16.gz" for k in range(parts))}
    return info


def build(mix: Path = MIX, out: Path = OUT, eval_sets: dict | None = None,
          check_only: bool = False, export_parts: int = 0, log=print) -> dict:
    t0 = time.time()
    eval_sets = EVAL_SETS if eval_sets is None else eval_sets
    cnt, mid, caps, n_rec, n_tok = count_train(mix / "train.jsonl")
    vocab = build_vocab(cnt, mid, caps)
    total = sum(cnt.values())
    covered = sum(cnt[t] for t in vocab.tokens)
    report = {"train_records": n_rec, "train_tokens": n_tok, "types": len(cnt),
              "vocab_tokens": len(vocab.tokens),
              "train_coverage": round(covered / max(1, total), 5)}
    log(f"[word] {json.dumps(report)} ({time.time() - t0:.0f}s)")
    if check_only:
        return report
    out.mkdir(parents=True, exist_ok=True)
    vj = {"n_symbols": N_SYMBOLS, "sep": SEP, "oov_ids": [OOV0, FIRST_WORD - 1],
          "first_word": FIRST_WORD, "bos": BOS, "eos": EOS, "max_tokens": MAX_TOKENS,
          "tokenizer": "curly quotes -> straight; tokens = regex " + _TOKEN.pattern +
                       " on the text; a line break is <sep>; each other token is casefolded",
          "oov_hash": "blake2b digest_size=8 of the casefolded token, little-endian, mod 256",
          "tokens": vocab.tokens, "surface": vocab.surface,
          "counts": [cnt[t] for t in vocab.tokens]}
    (out / "vocab.json").write_text(json.dumps(vj, ensure_ascii=False), encoding="utf-8")
    files = {}
    for split in ("train", "val"):
        files[f"v02a/{split}.jsonl"] = convert(
            mix / f"{split}.jsonl", out / "v02a" / f"{split}.jsonl", vocab,
            parts=export_parts if split == "train" else 0, n_records=n_rec)
        log(f"[word] {split}: " + json.dumps({k: v for k, v in files[f'v02a/{split}.jsonl'].items()
                                              if k != 'parts'}))
    missing = []
    for name, path in eval_sets.items():
        if not path.exists():
            missing.append(name)
            continue
        files[f"eval/{name}.jsonl"] = {**convert(path, out / "eval" / f"{name}.jsonl", vocab),
                                       "source_sha256": sha256_file(path)}
    manifest = {"version": 1, "n_symbols": N_SYMBOLS, "max_tokens": MAX_TOKENS,
                "source": {"mix": mix.name,
                           "mix_manifest_sha256": sha256_file(mix / "manifest.json")
                           if (mix / "manifest.json").exists() else None,
                           "train_sha256": sha256_file(mix / "train.jsonl"),
                           "val_sha256": sha256_file(mix / "val.jsonl")},
                "vocab_sha256": sha256_file(out / "vocab.json"), "report": report,
                "files": files, "missing_eval_sets": missing}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"[word] done in {time.time() - t0:.0f}s; missing eval sets: {missing or 'none'}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="coverage report only, no files")
    ap.add_argument("--export-u16", type=int, default=0, metavar="N",
                    help="also write the train records as N gzip shards")
    a = ap.parse_args()
    build(check_only=a.check, export_parts=a.export_u16, log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
