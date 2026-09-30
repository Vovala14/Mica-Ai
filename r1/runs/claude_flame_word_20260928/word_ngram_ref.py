#!/usr/bin/env python3
"""Size-matched word n-gram references for Flame-W, trained on the identical
token stream (r1/data/word/v02a/train.part*.u16.gz, the mix A training split).

    python word_ngram_ref.py WORD_DIR WORK_DIR ORDER:PRUNE [ORDER:PRUNE ...]

Each config ("4:0 1 1 1") is trained with lmplz, stored as a KenLM trie with
8-bit quantisation, and its size reported. Evaluation is done separately with
word_eval.py (kenlm:WORK_DIR/<tag>.binary).
"""
from __future__ import annotations

import gzip
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

BIN = Path("/opt/src/kenlm-0.3.0/build/bin")


def shards_to_text(word_dir: Path, out: Path) -> dict:
    if out.exists():
        return json.loads(out.with_suffix(".json").read_text())
    n_rec = n_tok = 0
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as g:
        for p in sorted((word_dir / "v02a").glob("train.part*.u16.gz")):
            data = gzip.open(p, "rb").read()
            i = 0
            while i < len(data):
                n = struct.unpack_from("<H", data, i)[0]
                ids = struct.unpack_from(f"<{n // 2}H", data, i + 2)
                g.write(" ".join(f"w{t}" for t in ids) + "\n")
                n_rec += 1
                n_tok += len(ids)
                i += 2 + n
    tmp.rename(out)
    info = {"records": n_rec, "tokens": n_tok}
    out.with_suffix(".json").write_text(json.dumps(info))
    return info


def main() -> int:
    word_dir, work = Path(sys.argv[1]), Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    text = work / "train_ids.txt"
    info = shards_to_text(word_dir, text)
    print(f"[ref] text {info} ({time.time() - t0:.0f}s)", flush=True)
    for cfg in sys.argv[3:]:
        order, _, prune = cfg.partition(":")
        tag = f"w{order}-p{prune.replace(' ', '')}" if prune else f"w{order}"
        binary = work / f"{tag}.binary"
        if binary.exists():
            print(f"[ref] {tag} exists: {binary.stat().st_size} bytes", flush=True)
            continue
        arpa = work / f"{tag}.arpa"
        cmd = [str(BIN / "lmplz"), "-o", order, "-S", "20%", "-T", str(work),
               "--discount_fallback"]
        if prune:
            cmd += ["--prune"] + prune.split()
        t1 = time.time()
        subprocess.run(cmd + ["--text", str(text), "--arpa", str(arpa)], check=True,
                       stderr=subprocess.DEVNULL)
        subprocess.run([str(BIN / "build_binary"), "-q", "8", "-b", "8", "trie", str(arpa),
                        str(binary)], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        arpa.unlink()
        meta = {"tag": tag, "order": int(order), "prune": prune or None,
                "file_bytes": binary.stat().st_size, "train": info,
                "seconds": round(time.time() - t1, 1)}
        (work / f"{tag}.json").write_text(json.dumps(meta))
        print(f"[ref] {json.dumps(meta)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
