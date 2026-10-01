#!/usr/bin/env python3
"""Flame-W v03 topical corpus: whole stretches of conversation that stay on one
subject, for the topic register (spec.TOPIC_CHANNELS).

    python r1/data/build_topical_corpus.py fetch   # download what is missing
    python r1/data/build_topical_corpus.py build   # write r1/data/word/v03_topical/

Why: the v02a records are one utterance or one pair of turns (about 20 words),
so a topic register has almost nothing to carry. Here each record is a WINDOW
of consecutive turns from one dialogue, up to 64 words (the trainer's record
length), turns separated by <sep> as in v02a pairs. A dialogue gives several
windows, so the model learns how a conversation continues on its subject.

Sources (training splits only; licences on the dataset pages, 2026-10-01):

  already used by v02a (r1/data/external/chat, build_chat_corpus.py):
    Topical-Chat (Amazon)          CDLA-Sharing 1.0   human chats about a topic
    Taskmaster-1/-2                CC BY 4.0          task dialogues
    Schema-Guided Dialogue         CC BY-SA 4.0       assistant dialogues
    SODA (AllenAI)                 CC BY 4.0          social dialogues (sampled)
  new (r1/data/external/topical):
    UltraChat 200k (HuggingFaceH4) MIT                multi-turn chats on one subject
    OpenAssistant oasst1           Apache 2.0         human-written Q&A threads
    Synthetic-Persona-Chat (Google) CC BY 4.0         casual chats: hobbies, jobs

Assistant-style turns (UltraChat, OASST) are cut to their first two sentences
and turns with code, tables or lists are dropped, so the text stays
conversational. Name placeholders in Persona-Chat become first names.

Outputs (word records, the trainer's format):
  r1/data/word/v03_topical/train.jsonl     v02a_nostories train + topical windows
  r1/data/word/v03_topical/val.jsonl       = v02a_nostories val (comparable scores)
  r1/data/word/v03_topical/topical.jsonl   the topical windows alone (topic codes)
  r1/data/word/v03_topical/manifest.json   counts per source, OOV rate, SHA-256
Deterministic: the same inputs give byte-identical outputs.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_chat_corpus as BC  # noqa: E402
import build_word_corpus as W  # noqa: E402

EXT = HERE / "external" / "topical"
BASE = W.OUT / "v02a_nostories"
OUT = W.OUT / "v03_topical"
HF = "https://huggingface.co/datasets"
NEW = {
    "ultrachat_sft0.parquet": f"{HF}/HuggingFaceH4/ultrachat_200k/resolve/main/data/"
                              "train_sft-00000-of-00003-a3ecf92756993583.parquet",
    "oasst1_train.parquet": f"{HF}/OpenAssistant/oasst1/resolve/main/data/"
                            "train-00000-of-00001-b42a775f407cee45.parquet",
    "persona_train.csv": f"{HF}/google/Synthetic-Persona-Chat/resolve/main/data/"
                         "Synthetic-Persona-Chat_train.csv",
}
# exact size and SHA-256 of each new file (the dataset repos' LFS records). A
# download that ends early is caught here and fetched again.
EXPECT = {
    "ultrachat_sft0.parquet": (243_999_189,
        "afa8fa7426081b2a0e732fb50dbb5cd402a28ad5f0dbe66c0d996d63e7220727"),
    "oasst1_train.parquet": (39_516_251,
        "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e"),
    "persona_train.csv": (15_889_931,
        "a7bb20f1c51fd18cc51b2adc942220994e302b237053110b04fce7b413c812da"),
}
# windows kept per source at most (dialogues are visited in a fixed shuffled order)
CAPS = {"topicalchat": 80_000, "taskmaster": 80_000, "sgd": 30_000, "soda": 200_000,
        "ultrachat": 120_000, "oasst": 40_000, "persona": 40_000}
MIN_TOKENS = 12
SEED = 20261002
NAMES = ("Sam", "Alex", "Jordan", "Taylor", "Chris", "Pat", "Jamie", "Morgan")


def say(msg: str) -> None:
    print(f"[topical] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ fetch
def _good(path: Path) -> bool:
    """A downloaded file is usable: right size and checksum when they are
    known (the new sources), otherwise non-empty."""
    if not path.exists():
        return False
    want = EXPECT.get(path.name) if path.parent == EXT else None
    if want is None:
        return path.stat().st_size > 0
    return path.stat().st_size == want[0] and sha256(path) == want[1]


def _download(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                size = int(r.headers.get("Content-Length") or 0)
                got = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
            if size and got != size:
                raise IOError(f"connection ended at {got:,} of {size:,} bytes")
            tmp.replace(path)
            if not _good(path):
                path.unlink()
                raise IOError("size or checksum does not match the dataset")
            return
        except Exception as exc:
            say(f"  {path.name}: attempt {attempt + 1} failed ({exc}); retrying")
            time.sleep(2 ** (attempt + 1))
    raise SystemExit(f"download failed 5 times: {url}")


def fetch() -> None:
    need = [(EXT / name, url) for name, url in NEW.items()]
    need += [(BC.EXT / rel, url) for rel, url in BC.downloads() if "tinystories" not in rel]
    missing = [(p, u) for p, u in need if not _good(p)]
    say(f"{len(need) - len(missing)} of {len(need)} source files present and complete; "
        f"downloading {len(missing)}")
    for k, (path, url) in enumerate(missing, 1):
        _download(path, url)
        say(f"  {k}/{len(missing)} {path.name} ({path.stat().st_size / 1e6:.0f} MB)")


# ------------------------------------------------------------------ dialogues
_TASK = re.compile(r"\s*(write|create|develop|compose|generate|design|draft|produce|"
                   r"provide|summari[sz]e|given|translate|rewrite|paraphrase|using|"
                   r"implement|build|make a|prepare|conduct|research)\b", re.I)
_BAD = re.compile(r"```|\|.*\||^\s*(#|\d+\.|[-*•] )", re.M)


def short_turn(text: str, sentences: int = 2) -> str:
    """First sentences of an assistant-style turn; '' for code, tables, lists."""
    text = str(text or "")
    if _BAD.search(text):
        return ""
    text = BC.clean_text(text)
    parts = BC._SENT.split(text)
    return " ".join(parts[:sentences])


def train_only(gen, name):
    for key, split, turns in gen:
        if split == "train":
            yield f"{name}:{key}", turns


def ultrachat():
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(EXT / "ultrachat_sft0.parquet")
    row = 0
    for batch in pf.iter_batches(batch_size=5000, columns=["messages"]):
        for msgs in batch.column(0).to_pylist():
            row += 1
            turns = [short_turn(m["content"]) for m in msgs]
            # the opening request is usually a writing task ("Write an essay
            # ..."), not conversation: drop it when it is long or imperative
            if msgs and (len(str(msgs[0]["content"]).split()) > 40 or
                         _TASK.match(str(msgs[0]["content"]))):
                turns[0] = ""
            yield f"uc:{row}", turns


def oasst():
    """Best-ranked thread from every English root: root, then the rank-0 reply
    at each level (the reply reviewers preferred)."""
    import pyarrow.parquet as pq
    t = pq.read_table(EXT / "oasst1_train.parquet",
                      columns=["message_id", "parent_id", "text", "lang", "rank", "deleted"]).to_pylist()
    kids: dict = {}
    for m in t:
        if m["lang"] == "en" and not m["deleted"]:
            kids.setdefault(m["parent_id"], []).append(m)
    for root in sorted(kids.get(None, []), key=lambda m: m["message_id"]):
        turns, cur = [root["text"]], root
        while kids.get(cur["message_id"]):
            cur = min(kids[cur["message_id"]], key=lambda m: (m["rank"] is None, m["rank"] or 0))
            turns.append(cur["text"])
        yield "oasst:" + root["message_id"], [short_turn(x, 3) for x in turns]


def persona():
    csv.field_size_limit(1 << 26)
    with open(EXT / "persona_train.csv", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            names = {"1": NAMES[i % 8], "2": NAMES[(i + 3) % 8]}
            turns = []
            for line in row["Best Generated Conversation"].splitlines():
                m = re.match(r"\s*User ([12]):\s*(.*)", line)
                if not m:
                    continue
                text = re.sub(r"\[user ([12])'s name\]", lambda g: names[g.group(1)],
                              m.group(2), flags=re.I)
                turns.append(re.sub(r"\[[^\]]*\]", "", text))
            yield f"persona:{i}", turns


SOURCES = {
    "topicalchat": lambda: train_only(BC.topicalchat(), "tc"),
    "taskmaster": lambda: train_only(BC.taskmaster(), "tm"),
    "sgd": lambda: train_only(BC.sgd(), "sgd"),
    "soda": lambda: ((k, t) for k, t in train_only(BC.soda(), "soda")
                     if BC.bucket(k + ":topical") < 25),
    "ultrachat": ultrachat,
    "oasst": oasst,
    "persona": persona,
}


def windows(turns: list[str], vocab) -> list[list[int]]:
    """Consecutive turns joined by <sep>, each window at most 64 tokens. A
    dropped turn (empty: code, a list, a table) ends the window, so turns
    that were not adjacent are never joined."""
    out, cur = [], []
    for t in turns:
        toks = W.tokenize(BC.clean_text(t))
        if not toks:
            if cur:
                out.append(cur)
            cur = []
            continue
        add = (["<sep>"] if cur else []) + toks
        if cur and len(cur) + len(add) > W.MAX_TOKENS:
            out.append(cur)
            cur = toks
        else:
            cur = cur + add
    if cur:
        out.append(cur)
    return [vocab.encode(w) for w in out if len(w) >= MIN_TOKENS]


# ------------------------------------------------------------------ build
def build() -> dict:
    t0 = time.time()
    vocab = W.Vocab.load(W.OUT / "vocab.json")
    if not (BASE / "train.jsonl").exists():
        raise SystemExit(f"missing {BASE}/train.jsonl (night.py builds it)")
    OUT.mkdir(parents=True, exist_ok=True)
    info = {"version": 1, "max_tokens": W.MAX_TOKENS, "min_tokens": MIN_TOKENS, "caps": CAPS,
            "sources": {}}
    topical = []
    for name, gen in SOURCES.items():
        dialogues = list(gen())
        random.Random(f"{SEED}:{name}").shuffle(dialogues)
        recs, n_dlg, n_tok, n_oov = [], 0, 0, 0
        for key, turns in dialogues:
            ws = windows(turns, vocab)
            if not ws:
                continue
            n_dlg += 1
            for w in ws:
                recs.append(W.pack(w))
                n_tok += len(w)
                n_oov += sum(W.OOV0 <= i < W.FIRST_WORD for i in w)
            if len(recs) >= CAPS[name]:
                recs = recs[:CAPS[name]]
                break
        info["sources"][name] = {"dialogues": n_dlg, "windows": len(recs), "tokens": n_tok,
                                 "oov_rate": round(n_oov / max(1, n_tok), 4)}
        say(f"{name}: {len(recs):,} windows from {n_dlg:,} dialogues, "
            f"{n_tok / max(1, len(recs)):.0f} words each, OOV {n_oov / max(1, n_tok):.1%}")
        topical += recs
    random.Random(SEED).shuffle(topical)
    (OUT / "topical.jsonl").write_text("".join(r + "\n" for r in topical), encoding="ascii")
    with open(OUT / "train.jsonl", "w", encoding="ascii", newline="\n") as out:
        with open(BASE / "train.jsonl", encoding="ascii") as base:
            n_base = 0
            for line in base:
                if line.strip():
                    out.write(line.strip() + "\n")
                    n_base += 1
        out.writelines(r + "\n" for r in topical)
    (OUT / "val.jsonl").write_bytes((BASE / "val.jsonl").read_bytes())
    info.update({"base_records": n_base, "topical_records": len(topical),
                 "files": {f: sha256(OUT / f) for f in ("train.jsonl", "val.jsonl", "topical.jsonl")},
                 "inputs": {p.name: sha256(p) for p in sorted(EXT.glob("*")) if p.is_file()},
                 "seconds": round(time.time() - t0)})
    (OUT / "manifest.json").write_text(json.dumps(info, indent=1) + "\n")
    say(f"wrote {OUT}: {n_base:,} base + {len(topical):,} topical records in {info['seconds']} s")
    return info


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "fetch":
        fetch()
    elif cmd == "build":
        build()
    else:
        raise SystemExit("usage: build_topical_corpus.py fetch|build")
