#!/usr/bin/env python3
"""Build the conversational corpus for MICA Ember v0.1 and MICA Flame v0.1:
text that looks like what people type (messages, chat, everyday speech),
plus simple coherent stories and the existing everyday corpus.

    python r1/data/build_chat_corpus.py fetch   # download the sources
    python r1/data/build_chat_corpus.py build   # write the corpus
    python r1/data/build_chat_corpus.py check   # read-only: eval overlaps of the current corpus
    python r1/data/build_chat_corpus.py mix     # byte-balanced training mixes (MIXES)
    python r1/data/build_chat_corpus.py export  # pack it for the cloud baselines
    python r1/data/build_chat_corpus.py all     # fetch, build, export

Sources (licences checked on the dataset pages on 27 Sep 2026 and approved by
the owner; attribution in r1/data/chat/ATTRIBUTION.md):

  SODA (AllenAI)                    CC BY 4.0         everyday social dialogues
  Taskmaster-1, Taskmaster-2        CC BY 4.0         human task dialogues
  Schema-Guided Dialogue            CC BY-SA 4.0      human-assistant dialogues
  Topical-Chat (Amazon)             CDLA-Sharing 1.0  human-human chats
  TinyStories V2 (GPT-4 stories)    CDLA-Sharing 1.0  simple coherent stories
  r1/data/everyday                  as before         Tatoeba, prose, COCO captions

Records are at most 256 bytes: one utterance per record, plus one adjacent
pair of turns per dialogue ("turn\\nnext turn"); stories are cut into
sentence-aligned chunks. Plain text, no speaker tags.

Splits never share a dialogue or a story: the sources' own splits where they
exist (SODA, Schema-Guided Dialogue, Topical-Chat, TinyStories), otherwise a
hash of the conversation id (Taskmaster, 96/2/2). COCO captions in the
everyday training split are cut to COCO_KEEP of them. Identical training
records are kept at most MAX_COPIES times.

Evaluation hygiene (version 2): no training, validation or test record
matches a record of a development or clean validation set -- everyday clean
val1000, chat_val1000, chat dev1000 and the EXTRA_EVAL development sets --
exactly or after the clean sets' normalisation (UTF-8 replacement, casefold,
whitespace collapse, strip; checks/make_clean_eval.py). Version 1 removed
exact copies from training only. The sealed clean test1000 is never opened.
The evaluation sets are chosen exactly as in version 1, so dev1000 and
chat_val1000 are unchanged, and a rebuild leaves those two files untouched
when their content is the same.

Every split has a source label per record (<split>.src, one byte per record,
codes in SOURCES), so `mix` can compose training sets by source. The version 1
corpus was about half TinyStories by bytes, which is where Ember v0.1's story
phrasing and everyday regression came from.

Outputs (hex records, one per line, the trainer's format):

  r1/data/chat/train.jsonl, val.jsonl, test.jsonl (+ .src labels)
  r1/data/chat/dev1000.jsonl            development set for decisions
  r1/data/eval_clean/chat_val1000.jsonl clean set for reporting only
  r1/data/chat/manifest.json            counts, parameters, SHA-256 of all files
  r1/data/chat/ATTRIBUTION.md
  r1/data/mix/<name>/train.jsonl, val.jsonl (+ .src), manifest.json

Deterministic: the same sources give byte-identical outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
R1 = HERE.parent
EXT = HERE / "external" / "chat"
OUT = HERE / "chat"
CLEAN = HERE / "eval_clean"

LIMIT = 256
SODA_TRAIN_KEEP = 42         # percent of SODA training dialogues used
TINY_TRAIN_KEEP = 20         # percent of TinyStories V2 training stories used
COCO_KEEP = 25               # percent of COCO captions kept from the everyday corpus
MAX_COPIES = 20              # identical training records kept at most this often
MIN_BYTES = 2                # shorter records are dropped
EVAL_MIN_BYTES = 8
EVAL_MIX = (("soda", 500), ("taskmaster", 150), ("sgd", 150), ("topicalchat", 200))
EVAL_PAIR_SHARE = 0.2
SEED = 20260927
VERSION = 2

# source label codes (<split>.src files); never reorder, only append
SOURCES = ("soda", "taskmaster", "sgd", "topicalchat", "tinystories", "everyday", "coco")
CODE = {s: i for i, s in enumerate(SOURCES)}
# development sets kept in run folders; excluded from every split when present
EXTRA_EVAL = (R1 / "runs/codex_flame_scaling_20260928/dev_fresh1000.jsonl",)

# byte-balanced training mixes: share of training BYTES per group. "everyday"
# is the complete r1/data/everyday training split (the mature everyday
# model's data, COCO included); the other groups come from chat/train.jsonl.
GROUPS = {"conversation": ("soda", "taskmaster", "sgd", "topicalchat"),
          "tinystories": ("tinystories",),
          "everyday": ("everyday", "coco")}
MIXES = {"v02a": {"conversation": 0.45, "everyday": 0.40, "tinystories": 0.15},
         "v02b": {"conversation": 0.60, "everyday": 0.30, "tinystories": 0.10}}
MIX_VAL_BYTES = 2_000_000

GH = "https://raw.githubusercontent.com"
HF = "https://huggingface.co/datasets"
SGD_FILES = {"train": 127, "dev": 20, "test": 34}
TM2_DOMAINS = ("flights", "food-ordering", "hotels", "movies", "music",
               "restaurant-search", "sports")


def downloads() -> list[tuple[str, str]]:
    """(relative path under external/chat, URL) for every source file."""
    out = []
    tm = f"{GH}/google-research-datasets/Taskmaster/master"
    for f in ("self-dialogs.json", "woz-dialogs.json"):
        out.append((f"taskmaster/TM-1-2019/{f}", f"{tm}/TM-1-2019/{f}"))
    for d in TM2_DOMAINS:
        out.append((f"taskmaster/TM-2-2020/{d}.json", f"{tm}/TM-2-2020/data/{d}.json"))
    sgd = f"{GH}/google-research-datasets/dstc8-schema-guided-dialogue/master"
    for split, n in SGD_FILES.items():
        for i in range(1, n + 1):
            f = f"dialogues_{i:03d}.json"
            out.append((f"sgd/{split}/{f}", f"{sgd}/{split}/{f}"))
    tc = f"{GH}/alexa/Topical-Chat/master/conversations"
    for f in ("train.json", "valid_freq.json", "valid_rare.json",
              "test_freq.json", "test_rare.json"):
        out.append((f"topicalchat/{f}", f"{tc}/{f}"))
    for f in ("train.parquet", "valid.parquet", "test.parquet"):
        out.append((f"soda/{f}", f"{HF}/allenai/soda/resolve/main/{f}"))
    for f in ("TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"):
        out.append((f"tinystories/{f}", f"{HF}/roneneldan/TinyStories/resolve/main/{f}"))
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch() -> dict:
    EXT.mkdir(parents=True, exist_ok=True)
    got = {}
    for rel, url in downloads():
        dst = EXT / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            t = time.time()
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
            tmp.replace(dst)
            print(f"[fetch] {rel} {dst.stat().st_size:,} B ({time.time() - t:.0f}s)",
                  flush=True)
        got[rel] = {"url": url, "bytes": dst.stat().st_size, "sha256": sha256_file(dst)}
    (EXT / "manifest.json").write_text(json.dumps(got, indent=1))
    return got


# ---------------------------------------------------------------- text helpers

def bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100


def clean_text(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def utf8_cut(b: bytes, limit: int) -> bytes:
    """At most `limit` bytes, never inside a UTF-8 character; prefer a space."""
    if len(b) <= limit:
        return b
    k = limit
    while k > 0 and (b[k] & 0xC0) == 0x80:
        k -= 1
    sp = b.rfind(b" ", 0, k + 1)
    return b[:sp] if sp > limit // 2 else b[:k]


_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])")


def chunks(text: str, limit: int = LIMIT) -> list[bytes]:
    """Sentence-aligned pieces of at most `limit` bytes."""
    out, cur = [], b""
    for sent in _SENT.split(text):
        s = sent.encode("utf-8")
        if len(s) > limit:
            if cur:
                out.append(cur)
                cur = b""
            while len(s) > limit:
                piece = utf8_cut(s, limit)
                out.append(piece.strip())
                s = s[len(piece):].strip()
            cur = s
            continue
        joined = (cur + b" " + s) if cur else s
        if len(joined) <= limit:
            cur = joined
        else:
            out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return [c for c in out if len(c) >= MIN_BYTES]


def dialogue_records(key: str, turns: list[str]) -> tuple[list[bytes], list[bytes]]:
    """(single utterances, one adjacent pair) of a dialogue."""
    utt = []
    for t in turns:
        t = clean_text(t)
        if not t:
            continue
        b = t.encode("utf-8")
        if len(b) > LIMIT:
            utt.extend(chunks(t))
        elif len(b) >= MIN_BYTES:
            utt.append(b)
    pairs = []
    cand = [(turns[i], turns[i + 1]) for i in range(len(turns) - 1)]
    cand = [(clean_text(a), clean_text(b)) for a, b in cand]
    cand = [(a, b) for a, b in cand if a and b]
    if cand:
        a, b = cand[bucket("pair:" + key) % len(cand)]
        p = (a + "\n" + b).encode("utf-8")
        if len(p) <= LIMIT:
            pairs.append(p)
    return utt, pairs


# ---------------------------------------------------------------- sources

def taskmaster():
    """(key, split, turns) for Taskmaster-1 (written and spoken) and -2."""
    files = [EXT / "taskmaster/TM-1-2019/self-dialogs.json",
             EXT / "taskmaster/TM-1-2019/woz-dialogs.json"] + \
        [EXT / f"taskmaster/TM-2-2020/{d}.json" for d in TM2_DOMAINS]
    for f in files:
        for conv in json.loads(f.read_text(encoding="utf-8")):
            key = "tm:" + conv["conversation_id"]
            b = bucket(key)
            split = "train" if b < 96 else ("val" if b < 98 else "test")
            yield key, split, [u.get("text", "") for u in conv["utterances"]]


def sgd():
    for split, n in SGD_FILES.items():
        s = {"train": "train", "dev": "val", "test": "test"}[split]
        for i in range(1, n + 1):
            f = EXT / f"sgd/{split}/dialogues_{i:03d}.json"
            for d in json.loads(f.read_text(encoding="utf-8")):
                yield "sgd:" + d["dialogue_id"], s, [t["utterance"] for t in d["turns"]]


def topicalchat():
    for fname, split in (("train.json", "train"), ("valid_freq.json", "val"),
                         ("valid_rare.json", "val"), ("test_freq.json", "test"),
                         ("test_rare.json", "test")):
        data = json.loads((EXT / "topicalchat" / fname).read_text(encoding="utf-8"))
        for cid in sorted(data):
            yield "tc:" + cid, split, [m["message"] for m in data[cid]["content"]]


def soda():
    import pyarrow.parquet as pq
    for fname, split in (("train.parquet", "train"), ("valid.parquet", "val"),
                         ("test.parquet", "test")):
        pf = pq.ParquetFile(EXT / "soda" / fname)
        row = 0
        for batch in pf.iter_batches(batch_size=20000, columns=["dialogue"]):
            for turns in batch.column(0).to_pylist():
                key = f"soda:{split}:{row}"
                row += 1
                if split == "train" and bucket(key) >= SODA_TRAIN_KEEP:
                    continue
                yield key, split, list(turns or [])


def tinystories():
    """(key, split, text) for TinyStories V2: training stories sampled by a
    hash of their text, validation stories split 50/50 into val and test."""
    for fname, split in (("TinyStoriesV2-GPT4-train.txt", "train"),
                         ("TinyStoriesV2-GPT4-valid.txt", "val/test")):
        buf = []
        with open(EXT / "tinystories" / fname, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip() == "<|endoftext|>":
                    text = clean_text(" ".join(buf))
                    buf = []
                    if not text:
                        continue
                    key = "ts:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                    b = bucket(key)
                    if split == "train":
                        if b < TINY_TRAIN_KEEP:
                            yield key, "train", text
                    else:
                        yield key, ("val" if b < 50 else "test"), text
                else:
                    buf.append(line)
        # a final story without a closing marker is ignored on purpose


def coco_captions() -> set[bytes]:
    """The everyday corpus's normalised COCO captions (build_everyday.py)."""
    out = set()
    for fname in ("captions_train2017.json", "captions_val2017.json"):
        ann = json.loads((HERE / "external/coco" / fname).read_text(encoding="utf-8"))
        for a in ann["annotations"]:
            c = clean_text(a["caption"])
            if not c:
                continue
            c = c[0].upper() + c[1:]
            if c[-1] not in ".!?":
                c += "."
            out.add(c.encode("utf-8"))
    return out


def read_hex(path: Path) -> list[bytes]:
    return [bytes.fromhex(l.strip()) for l in open(path, encoding="ascii") if l.strip()]


def norm_key(raw: bytes) -> bytes:
    """checks/make_clean_eval.key: the clean evaluation sets are disjoint from
    the everyday training split under this normalisation."""
    text = raw[:LIMIT].decode("utf-8", errors="replace").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def legacy_eval_sets() -> dict[str, list[bytes]]:
    """Evaluation sets that exist before this build: everyday clean val1000
    and the EXTRA_EVAL development sets that are present. The sealed clean
    test1000 is never opened: it stays untouched until a release evaluation."""
    out = {"everyday_clean_val1000": read_hex(CLEAN / "val1000.jsonl")}
    for p in EXTRA_EVAL:
        if p.exists():
            out[p.stem] = read_hex(p)
    return out


def _permute(rng: random.Random, n: int) -> list[int]:
    """The permutation rng.shuffle applies to any list of length n (shuffle's
    swaps depend on the length only), so parallel lists stay aligned."""
    order = list(range(n))
    rng.shuffle(order)
    return order


def write_hex(path: Path, recs: list[bytes]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        for r in recs:
            f.write(r.hex() + "\n")
    return sha256_file(path)


def write_hex_if_changed(path: Path, recs: list[bytes]) -> str:
    """write_hex, but leave an existing file alone (not even reopened for
    writing) when it already holds exactly these records: evaluators may be
    reading the development and clean sets during a rebuild."""
    body = "".join(r.hex() + "\n" for r in recs).encode("ascii")
    if path.exists() and path.stat().st_size == len(body) and path.read_bytes() == body:
        return hashlib.sha256(body).hexdigest()
    return write_hex(path, recs)


# ---------------------------------------------------------------- build

def build() -> dict:
    t0 = time.time()
    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}          # CODE per record
    evalpool = {src: {"single": [], "pair": []} for src, _ in EVAL_MIX}
    stats = Counter()

    def add(src, split, singles, pairs):
        for r in singles:
            splits[split].append(r)
            labels[split].append(CODE[src])
            stats[f"{src}/{split}/single"] += 1
        for r in pairs:
            splits[split].append(r)
            labels[split].append(CODE[src])
            stats[f"{src}/{split}/pair"] += 1
        if split == "val" and src in evalpool:
            evalpool[src]["single"].extend(singles)
            evalpool[src]["pair"].extend(pairs)

    for src, gen in (("taskmaster", taskmaster), ("sgd", sgd),
                     ("topicalchat", topicalchat), ("soda", soda)):
        n = 0
        for key, split, turns in gen():
            singles, pairs = dialogue_records(key, turns)
            add(src, split, singles, pairs)
            n += 1
        stats[f"{src}/dialogues"] = n
        print(f"[chat] {src}: {n:,} dialogues ({time.time() - t0:.0f}s)", flush=True)

    n = 0
    for key, split, text in tinystories():
        recs = chunks(text)
        splits[split].extend(recs)
        labels[split].extend([CODE["tinystories"]] * len(recs))
        stats[f"tinystories/{split}/chunk"] += len(recs)
        n += 1
    stats["tinystories/stories"] = n
    print(f"[chat] tinystories: {n:,} stories ({time.time() - t0:.0f}s)", flush=True)

    coco = coco_captions()
    for r in read_hex(HERE / "everyday/train.jsonl"):
        if r in coco:
            if bucket("coco:" + hashlib.sha256(r).hexdigest()) >= COCO_KEEP:
                stats["everyday/train/coco_dropped"] += 1
                continue
            stats["everyday/train/coco_kept"] += 1
            code = CODE["coco"]
        else:
            stats["everyday/train/other"] += 1
            code = CODE["everyday"]
        splits["train"].append(r)
        labels["train"].append(code)
    print(f"[chat] everyday train added ({time.time() - t0:.0f}s)", flush=True)

    # evaluation sets from the validation splits, disjoint from each other
    # (chosen exactly as in version 1)
    everyday_clean = set(read_hex(CLEAN / "val1000.jsonl"))
    train_hashes = {hashlib.sha256(r).digest()[:8] for r in splits["train"]}

    def eligible(r):
        return (len(r) >= EVAL_MIN_BYTES and r not in everyday_clean and
                hashlib.sha256(r).digest()[:8] not in train_hashes)

    chosen = {"chat_val1000": [], "dev1000": []}
    used = set()
    for src, n in EVAL_MIX:
        for kind, k in (("pair", round(n * EVAL_PAIR_SHARE)),
                        ("single", n - round(n * EVAL_PAIR_SHARE))):
            pool = sorted({r for r in evalpool[src][kind] if eligible(r)})
            rng.shuffle(pool)
            if len(pool) < 2 * k:
                raise SystemExit(f"not enough {src} {kind} records for the eval sets")
            for name, part in (("chat_val1000", pool[:k]), ("dev1000", pool[k:2 * k])):
                chosen[name].extend(part)
                used.update(part)
    for name in chosen:
        rng.shuffle(chosen[name])
        assert len(chosen[name]) == sum(n for _, n in EVAL_MIX), (name, len(chosen[name]))

    # version 2: no split keeps a record that matches an evaluation record,
    # exactly or after normalisation
    legacy = legacy_eval_sets()
    eval_keys = {norm_key(r) for recs in legacy.values() for r in recs} | \
        {norm_key(r) for r in used}
    stats["eval/keys"] = len(eval_keys)

    # training set: no evaluation record, identical records capped
    banned = everyday_clean | used                          # version 1 rule
    copies, keep = Counter(), []
    for i, r in enumerate(splits["train"]):
        if r in banned:
            stats["train/removed_eval_overlap"] += 1
            continue
        if norm_key(r) in eval_keys:
            stats["train/removed_eval_normalised"] += 1
            continue
        copies[r] += 1
        if copies[r] > MAX_COPIES:
            stats["train/removed_copies"] += 1
            continue
        keep.append(i)
    order = _permute(rng, len(keep))
    idx = [keep[j] for j in order]
    splits["train"] = [splits["train"][i] for i in idx]
    labels["train"] = [labels["train"][i] for i in idx]
    for s in ("val", "test"):
        keep = []
        for i, r in enumerate(splits[s]):
            if r in used:
                continue                                    # version 1 rule
            if norm_key(r) in eval_keys:
                stats[f"{s}/removed_eval_normalised"] += 1
                continue
            keep.append(i)
        order = _permute(rng, len(keep))
        idx = [keep[j] for j in order]
        splits[s] = [splits[s][i] for i in idx]
        labels[s] = [labels[s][i] for i in idx]

    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for s in ("train", "val", "test"):
        files[f"chat/{s}.jsonl"] = write_hex(OUT / f"{s}.jsonl", splits[s])
        (OUT / f"{s}.src").write_bytes(bytes(labels[s]))
        files[f"chat/{s}.src"] = sha256_file(OUT / f"{s}.src")
    files["chat/dev1000.jsonl"] = write_hex_if_changed(OUT / "dev1000.jsonl",
                                                       chosen["dev1000"])
    files["eval_clean/chat_val1000.jsonl"] = write_hex_if_changed(
        CLEAN / "chat_val1000.jsonl", chosen["chat_val1000"])
    (OUT / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")
    by_source = {s: {src: {"records": 0, "bytes": 0} for src in SOURCES}
                 for s in splits}
    for s in splits:
        for r, c in zip(splits[s], labels[s]):
            d = by_source[s][SOURCES[c]]
            d["records"] += 1
            d["bytes"] += len(r)
    manifest = {
        "version": VERSION,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {"limit": LIMIT, "soda_train_keep_pct": SODA_TRAIN_KEEP,
                       "tinystories_train_keep_pct": TINY_TRAIN_KEEP,
                       "coco_keep_pct": COCO_KEEP, "max_copies": MAX_COPIES,
                       "eval_mix": EVAL_MIX, "eval_pair_share": EVAL_PAIR_SHARE,
                       "seed": SEED},
        "source_codes": list(SOURCES),
        "eval_sets_excluded": {k: len(v) for k, v in legacy.items()} |
        {"chat_dev1000": len(chosen["dev1000"]),
         "chat_val1000": len(chosen["chat_val1000"])},
        "records": {s: len(v) for s, v in splits.items()},
        "bytes": {s: sum(map(len, v)) for s, v in splits.items()},
        "by_source": by_source,
        "counts": dict(sorted(stats.items())),
        "files_sha256": files,
        "sources": json.loads((EXT / "manifest.json").read_text())
        if (EXT / "manifest.json").exists() else {},
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("[chat]", json.dumps({k: manifest[k] for k in ("records", "bytes", "files_sha256")},
                               indent=1), flush=True)
    return manifest


def _eval_key_sets() -> dict[str, set[bytes]]:
    sets = legacy_eval_sets()
    for name, path in (("chat_val1000", CLEAN / "chat_val1000.jsonl"),
                       ("chat_dev1000", OUT / "dev1000.jsonl")):
        if path.exists():
            sets[name] = read_hex(path)
    return {k: {norm_key(r) for r in v} for k, v in sets.items()}


def check() -> dict:
    """Read-only. For each evaluation set, how many of its records have an
    exact or a normalised copy in the current chat train/val/test. Useful for
    corpora built before version 2 (whose training split removed exact copies
    of val1000, chat dev1000 and chat_val1000 only)."""
    t0 = time.time()
    raw = legacy_eval_sets()
    for name, path in (("chat_val1000", CLEAN / "chat_val1000.jsonl"),
                       ("chat_dev1000", OUT / "dev1000.jsonl")):
        if path.exists():
            raw[name] = read_hex(path)
    keys = {k: {norm_key(r) for r in v} for k, v in raw.items()}
    exact = {k: set(v) for k, v in raw.items()}
    report = {"sets": {k: len(v) for k, v in raw.items()}}
    for split in ("train", "val", "test"):
        path = OUT / f"{split}.jsonl"
        if not path.exists():
            continue
        hit_n = {k: set() for k in raw}
        hit_x = {k: set() for k in raw}
        rows_n, rows_x = Counter(), Counter()
        with open(path, encoding="ascii") as f:
            for line in f:
                if not line.strip():
                    continue
                r = bytes.fromhex(line.strip())
                kr = norm_key(r)
                for name in raw:
                    if kr in keys[name]:
                        hit_n[name].add(kr)
                        rows_n[name] += 1
                        if r in exact[name]:
                            hit_x[name].add(r)
                            rows_x[name] += 1
        report[split] = {name: {"eval_records_with_exact_copy": len(hit_x[name]),
                                "eval_records_with_normalised_copy": len(hit_n[name]),
                                f"{split}_records_exact": rows_x[name],
                                f"{split}_records_normalised": rows_n[name]}
                         for name in raw}
    report["seconds"] = round(time.time() - t0, 1)
    (OUT / "check.json").write_text(json.dumps(report, indent=1))
    print("[chat] check", json.dumps(report, indent=1), flush=True)
    return report


def _pool(path: Path, src_path: Path | None, groups: dict[str, tuple[str, ...]],
          eval_keys: set[bytes], stats: Counter, tag: str) -> dict[str, list]:
    """Records of `path` per group, as (record, code); without a label file
    every record belongs to the "everyday" group. Evaluation matches and
    copies beyond MAX_COPIES are dropped."""
    recs = read_hex(path)
    codes = (list(src_path.read_bytes()) if src_path is not None
             else [CODE["everyday"]] * len(recs))
    assert len(codes) == len(recs), f"{src_path}: {len(codes)} labels for {len(recs)} records"
    group_of = {CODE[s]: g for g, ss in groups.items() for s in ss}
    out = {g: [] for g in groups}
    copies = Counter()
    for r, c in zip(recs, codes):
        g = group_of.get(c)
        if g is None:
            continue
        if norm_key(r) in eval_keys:
            stats[f"{tag}/removed_eval"] += 1
            continue
        copies[r] += 1
        if copies[r] > MAX_COPIES:
            stats[f"{tag}/removed_copies"] += 1
            continue
        out[g].append((r, c))
    return out


def _take(pool: list, budget: float, rng: random.Random) -> list:
    """A random subset of `pool` whose bytes just reach `budget`."""
    order = _permute(rng, len(pool))
    got, total = [], 0
    for i in order:
        if total >= budget:
            break
        got.append(pool[i])
        total += len(pool[i][0])
    return got


def mix(name: str, shares: dict[str, float] | None = None) -> dict:
    """A training set whose BYTES come from the groups in the given shares.
    The scarcest group (relative to its share) is used completely; the others
    are sampled at random, without replacement, to their byte budgets. The
    validation set has the same shares (MIX_VAL_BYTES in all)."""
    t0 = time.time()
    shares = dict(shares or MIXES[name])
    if set(shares) - set(GROUPS) or abs(sum(shares.values()) - 1) > 1e-9 or \
            min(shares.values()) < 0:
        raise SystemExit(f"bad shares {shares}: groups {sorted(GROUPS)}, summing to 1")
    groups = {g: GROUPS[g] for g, v in shares.items() if v > 0}
    rng = random.Random(f"{SEED}:mix:{name}")
    eval_keys = set().union(*_eval_key_sets().values())
    stats = Counter()
    chat = {g: v for g, v in groups.items() if g != "everyday"}
    out_dir = HERE / "mix" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    files, report = {}, {}
    for split, total_bytes in (("train", None), ("val", MIX_VAL_BYTES)):
        pools = _pool(OUT / f"{split}.jsonl", OUT / f"{split}.src", chat,
                      eval_keys, stats, f"chat_{split}") if chat else {}
        if "everyday" in groups:
            pools |= _pool(HERE / f"everyday/{split}.jsonl", None,
                           {"everyday": GROUPS["everyday"]}, eval_keys, stats,
                           f"everyday_{split}")
        avail = {g: sum(len(r) for r, _ in pools[g]) for g in groups}
        if min(avail.values()) == 0:
            raise SystemExit(f"{split}: an empty group, available bytes {avail}")
        full = min(avail[g] / shares[g] for g in groups)
        total = full if total_bytes is None else min(full, total_bytes)
        picked = []
        for g in sorted(groups):
            got = _take(pools[g], shares[g] * total, rng)
            picked.extend(got)
            report.setdefault(split, {})[g] = {
                "available_bytes": avail[g], "records": len(got),
                "bytes": sum(len(r) for r, _ in got)}
        order = _permute(rng, len(picked))
        picked = [picked[i] for i in order]
        files[f"mix/{name}/{split}.jsonl"] = write_hex(out_dir / f"{split}.jsonl",
                                                       [r for r, _ in picked])
        (out_dir / f"{split}.src").write_bytes(bytes(c for _, c in picked))
        files[f"mix/{name}/{split}.src"] = sha256_file(out_dir / f"{split}.src")
        got_bytes = sum(len(r) for r, _ in picked)
        report[split]["total"] = {"records": len(picked), "bytes": got_bytes,
                                  "shares": {g: round(report[split][g]["bytes"] /
                                                      got_bytes, 4) for g in groups}}
    manifest = {"version": VERSION, "name": name, "shares": shares,
                "groups": {g: list(GROUPS[g]) for g in groups},
                "everyday_source": "r1/data/everyday/{train,val}.jsonl, complete",
                "chat_manifest_sha256": sha256_file(OUT / "manifest.json")
                if (OUT / "manifest.json").exists() else None,
                "seed": f"{SEED}:mix:{name}", "splits": report,
                "counts": dict(sorted(stats.items())), "files_sha256": files,
                "seconds": round(time.time() - t0, 1)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[mix] {name}", json.dumps({k: manifest[k] for k in ("shares", "splits")},
                                      indent=1), flush=True)
    return manifest


ATTRIBUTION = """# Sources of r1/data/chat

This corpus contains text from the datasets below, used under their licences.
Share-alike terms apply to redistribution of the data itself.

- SODA: Hyunwoo Kim et al., Allen Institute for AI. CC BY 4.0.
  https://huggingface.co/datasets/allenai/soda
- Taskmaster-1 and Taskmaster-2: Bill Byrne, Karthik Krishnamoorthi, Chinnadhurai
  Sankar, Arvind Neelakantan, Saravanan Ganesh, Amit Dubey, Kyu-Young Kim, Andy
  Cedilnik (Google LLC). CC BY 4.0.
  https://github.com/google-research-datasets/Taskmaster
- Schema-Guided Dialogue: Abhinav Rastogi et al. (Google). CC BY-SA 4.0.
  https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
- Topical-Chat: Karthik Gopalakrishnan et al. (Amazon). CDLA-Sharing 1.0.
  https://github.com/alexa/Topical-Chat
- TinyStories (V2, GPT-4 stories): Ronen Eldan and Yuanzhi Li. CDLA-Sharing 1.0.
  https://huggingface.co/datasets/roneneldan/TinyStories
- r1/data/everyday: Tatoeba (CC BY 2.0 FR), COCO captions (CC BY 4.0) and
  filtered prose, as documented in build_everyday.py.
"""


EXPORT = HERE / "export"
EXPORT_PARTS = 3


def export() -> dict:
    """Pack the corpus for the cloud baselines (KenLM and friends), in
    export_text.py's format: per record a 2-byte little-endian length, then
    its bytes; gzip level 6. The training split is spread round-robin over
    EXPORT_PARTS files so each stays well under 400 MB."""
    import gzip
    import struct
    EXPORT.mkdir(parents=True, exist_ok=True)
    jobs = [("chat_train", OUT / "train.jsonl", EXPORT_PARTS),
            ("chat_val", OUT / "val.jsonl", 1), ("chat_test", OUT / "test.jsonl", 1),
            ("chat_dev1000", OUT / "dev1000.jsonl", 1),
            ("chat_val1000_clean", CLEAN / "chat_val1000.jsonl", 1)]
    man = {}
    for name, src, parts in jobs:
        paths = ([EXPORT / f"{name}.part{k}.bin.gz" for k in range(parts)]
                 if parts > 1 else [EXPORT / f"{name}.bin.gz"])
        outs = [gzip.open(p, "wb", compresslevel=6) for p in paths]
        n = [0] * parts
        with open(src, encoding="ascii") as f:
            for i, line in enumerate(l for l in f if l.strip()):
                r = bytes.fromhex(line.strip())
                outs[i % parts].write(struct.pack("<H", len(r)) + r)
                n[i % parts] += 1
        for o in outs:
            o.close()
        for p, k in zip(paths, n):
            man[p.name] = {"records": k, "sha256": sha256_file(p),
                           "bytes": p.stat().st_size}
    (EXPORT / "chat_manifest.json").write_text(json.dumps(man, indent=1))
    print("[chat] export", json.dumps(man, indent=1), flush=True)
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["fetch", "build", "check", "mix", "export", "all"])
    ap.add_argument("--name", action="append", choices=sorted(MIXES),
                    help="mix: which of MIXES to write (default all)")
    a = ap.parse_args()
    if a.step in ("fetch", "all"):
        fetch()
    if a.step in ("build", "all"):
        build()
    if a.step == "check":
        check()
    if a.step == "mix":
        for name in a.name or sorted(MIXES):
            mix(name)
    if a.step in ("export", "all"):
        export()
    return 0


if __name__ == "__main__":
    sys.exit(main())
