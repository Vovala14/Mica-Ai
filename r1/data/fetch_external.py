#!/usr/bin/env python3
"""Download the external data and baselines the owner approved (2026-09-26)
into r1/data/external/, with a manifest of sizes and SHA-256 checksums.

  Tatoeba English sentences (CC BY 2.0 FR)
  Tatoeba English detailed sentences (owner names for source attribution)
  COCO 2017 caption annotations (CC BY 4.0)
  llama2.c (MIT) source, and the stories260K checkpoint with its tokenizer
  TinyStories V2 validation text (stories260K's own domain)
  tiny-character-transformer source
(KenLM and Presage come from PyPI and Ubuntu in the cloud workspace.)

Every download is attempted; a failure is logged and the rest continue.
Nothing downloaded here is executed by this script.
"""
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

R1 = Path(__file__).resolve().parents[1]
OUT = R1 / "data/external"
HF = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/"
GH = "https://github.com/"
ITEMS = [
    ("tatoeba/eng_sentences.tsv.bz2",
     ["https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2"]),
    ("tatoeba/eng_sentences_detailed.tsv.bz2",
     ["https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_detailed.tsv.bz2"]),
    ("coco/annotations_trainval2017.zip",
     ["http://images.cocodataset.org/annotations/annotations_trainval2017.zip"]),
    ("llama2c/llama2.c-master.zip",
     [GH + "karpathy/llama2.c/archive/refs/heads/master.zip"]),
    ("llama2c/stories260K.bin", [HF + "stories260K.bin"]),
    ("llama2c/stories260K.pt", [HF + "stories260K.pt"]),
    ("llama2c/tok512.model", [HF + "tok512.model"]),
    ("llama2c/tok512.bin", [HF + "tok512.bin"]),
    ("tinystories/TinyStoriesV2-GPT4-valid.txt",
     ["https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
      "TinyStoriesV2-GPT4-valid.txt"]),
    ("tinychar/tiny-character-transformer.zip",
     [GH + "maxpolaczuk/tiny-character-transformer/archive/refs/heads/main.zip",
      GH + "maxpolaczuk/tiny-character-transformer/archive/refs/heads/master.zip"]),
]


def beat(msg):
    hb = R1 / "runs/soft/heartbeat"
    try:
        if hb.parent.exists():
            hb.write_text(f"fetch {msg} {int(time.time())}\n")
    except OSError:
        pass


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "mica-research/0.1"})
    h = hashlib.sha256()
    n = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
            if n % (32 << 20) < (1 << 20):
                beat(f"{dest.name} {n >> 20} MB")
    tmp.replace(dest)
    return n, h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man_path = OUT / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    for rel, urls in ITEMS:
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and rel in manifest and manifest[rel].get("sha256"):
            print(f"[fetch] {rel}: already here", flush=True)
            continue
        for url in urls:
            try:
                t0 = time.time()
                n, sha = fetch(url, dest)
                manifest[rel] = {"url": url, "bytes": n, "sha256": sha,
                                 "downloaded": time.strftime("%Y-%m-%d %H:%M:%S")}
                print(f"[fetch] {rel}: {n:,} bytes, sha256 {sha} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                break
            except Exception as exc:                        # noqa: BLE001
                print(f"[fetch] {rel}: FAILED from {url}: {exc}", flush=True)
        man_path.write_text(json.dumps(manifest, indent=1))
    # the caption files only, out of the 241 MB annotation archive
    z = OUT / "coco/annotations_trainval2017.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            for name in ("annotations/captions_train2017.json",
                         "annotations/captions_val2017.json"):
                target = OUT / "coco" / Path(name).name
                if not target.exists():
                    target.write_bytes(zf.read(name))
                    print(f"[fetch] extracted {target.name}: "
                          f"{target.stat().st_size:,} bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
