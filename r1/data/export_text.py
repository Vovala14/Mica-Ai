#!/usr/bin/env python3
"""Pack training and evaluation text for the baselines that run in the cloud
workspace (KenLM, Presage, fastText, the small transformers), so that every
system is trained on the same text as MICA.

Writes r1/data/export/:
  bulk_train256.partK.bin.gz  every record of r1/data/bulk/train.jsonl, cut to
                              its first 256 bytes (MICA trains on --record-bytes
                              256); record i goes to part i % PARTS, so each
                              part is an even sample of the whole split
  everyday_{train,val,test}.bin.gz   r1/data/everyday/*.jsonl, as they are
  manifest.json               records, bytes and SHA-256 of every file

Format: for each record a 2-byte little-endian length, then its bytes;
gzip level 6.
"""
import gzip
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

R1 = Path(__file__).resolve().parents[1]
OUT = R1 / "data/export"
PARTS = 4
CUT = 256


def beat(msg):
    hb = R1 / "runs/soft/heartbeat"
    try:
        if hb.parent.exists():
            hb.write_text(f"export {msg} {int(time.time())}\n")
    except OSError:
        pass


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(src, outs, cut=None):
    """Records of a hex .jsonl file, round-robin into the open gzip files."""
    n = [0] * len(outs)
    nbytes = [0] * len(outs)
    with open(src) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = bytes.fromhex(line)
            if cut:
                r = r[:cut]
            k = i % len(outs)
            outs[k].write(struct.pack("<H", len(r)) + r)
            n[k] += 1
            nbytes[k] += len(r)
            if i % 100_000 == 0:
                beat(f"{src.name} {i}")
    return n, nbytes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = {}
    jobs = [("bulk_train256", R1 / "data/bulk/train.jsonl", PARTS, CUT)]
    for s in ("train", "val", "test"):
        jobs.append((f"everyday_{s}", R1 / f"data/everyday/{s}.jsonl", 1, None))
    for name, src, parts, cut in jobs:
        if not src.exists():
            print(f"[export] {src} missing; skipped", flush=True)
            continue
        t0 = time.time()
        paths = ([OUT / f"{name}.part{k}.bin.gz" for k in range(parts)]
                 if parts > 1 else [OUT / f"{name}.bin.gz"])
        outs = [gzip.open(p, "wb", compresslevel=6) for p in paths]
        try:
            n, nbytes = pack(src, outs, cut)
        finally:
            for o in outs:
                o.close()
        for p, k, b in zip(paths, n, nbytes):
            man[p.name] = {"records": k, "text_bytes": b,
                           "file_bytes": p.stat().st_size, "sha256": sha(p),
                           "source": str(src.relative_to(R1.parent)).replace("\\", "/"),
                           "cut": cut}
        print(f"[export] {name}: {sum(n):,} records, {sum(nbytes):,} bytes "
              f"({time.time() - t0:.0f}s)", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
