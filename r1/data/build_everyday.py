#!/usr/bin/env python3
"""Build the everyday-English corpus for an autocomplete-oriented MICA:
r1/data/everyday/{train,val,test}.jsonl (hex records, one per line, the
trainer's format) and plain-text copies for the baselines.

Sources (r1/data/external, fetched by fetch_external.py):
  Tatoeba English sentences, CC BY 2.0 FR -- one sentence per record;
    exact duplicates removed; sentences naming "Tom" (a Tatoeba convention
    that dominates the corpus) kept at most at 15% of the result
  COCO 2017 captions, CC BY 4.0 -- one caption per record, first letter
    capitalised, final full stop added
  prose from the existing training split (r1/data/bulk/train.jsonl), maths
    and list-like records removed (filter_rules.py), first 256 bytes

Splits: Tatoeba by sentence id, COCO train2017 by image id (all captions of an
image together), 90/5/5 by a hash; COCO val2017 goes to validation and test
by image id; prose only to train. Records are shuffled with a fixed seed.
"""
import bz2
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import filter_rules as fr                                   # noqa: E402

R1 = HERE.parent
EXT = HERE / "external"
OUT = HERE / "everyday"
PROSE_RECORDS = 200_000
TOM_SHARE = 0.15


def beat(msg):
    hb = R1 / "runs/soft/heartbeat"
    try:
        if hb.parent.exists():
            hb.write_text(f"everyday {msg} {int(time.time())}\n")
    except OSError:
        pass


def bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100


def split_of(key: str) -> str:
    b = bucket(key)
    return "train" if b < 90 else ("val" if b < 95 else "test")


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260926)
    out = {"train": [], "val": [], "test": []}
    stats = {}
    # Tatoeba
    seen, tom, other = set(), [], []
    with bz2.open(EXT / "tatoeba/eng_sentences.tsv.bz2", "rt", encoding="utf-8",
                  errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sid, text = parts[0], clean(parts[2])
            b = text.encode("utf-8")
            if not 8 <= len(b) <= 256 or text in seen:
                continue
            seen.add(text)
            (tom if re.search(r"\bTom\b", text) else other).append((sid, b))
    rng.shuffle(tom)
    keep_tom = int(TOM_SHARE / (1 - TOM_SHARE) * len(other))
    tat = other + tom[:keep_tom]
    for sid, b in tat:
        out[split_of("tatoeba:" + sid)].append(b)
    stats["tatoeba"] = {"unique": len(seen), "tom_total": len(tom),
                        "tom_kept": min(keep_tom, len(tom)), "used": len(tat)}
    beat("tatoeba done")
    # COCO captions
    n_coco = 0
    for fname, split in (("captions_train2017.json", None),
                         ("captions_val2017.json", "val/test")):
        ann = json.loads((EXT / "coco" / fname).read_text(encoding="utf-8"))
        for a in ann["annotations"]:
            c = clean(a["caption"])
            if not c:
                continue
            c = c[0].upper() + c[1:]
            if c[-1] not in ".!?":
                c += "."
            b = c.encode("utf-8")
            if not 8 <= len(b) <= 256:
                continue
            key = f"coco:{a['image_id']}"
            if split is None:
                s = split_of(key)
            else:
                s = "val" if bucket(key) < 50 else "test"
            out[s].append(b)
            n_coco += 1
    stats["coco_captions"] = n_coco
    beat("coco done")
    # prose from the existing training split
    prose = []
    with open(R1 / "data/bulk/train.jsonl") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = bytes.fromhex(line)[:256]
            if len(r) >= 32 and fr.keep(r):
                prose.append(r)
            if i % 200_000 == 0:
                beat(f"prose {i}")
    rng.shuffle(prose)
    prose = prose[:PROSE_RECORDS]
    out["train"] += prose
    stats["prose"] = len(prose)
    for s in out:
        rng.shuffle(out[s])
        with open(OUT / f"{s}.jsonl", "w") as f:
            for r in out[s]:
                f.write(r.hex() + "\n")
        with open(OUT / f"{s}.txt", "w", encoding="utf-8") as f:
            for r in out[s]:
                f.write(r.decode("utf-8", "replace").replace("\n", " ") + "\n")
        stats[s] = {"records": len(out[s]), "bytes": sum(map(len, out[s]))}
    (OUT / "manifest.json").write_text(json.dumps(stats, indent=1))
    print("[everyday]", json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
