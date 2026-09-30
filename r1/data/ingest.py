#!/usr/bin/env python3
"""Turn your own text into MICA R1 training records.

This is the "bring your own data" entry point. Point it at a folder, a file,
or a mix of both; it produces the record splits the trainer reads.

    python3 r1/data/ingest.py --input ~/my_texts --out r1/data/mydata
    python3 r1/run_search.py --records r1/data/mydata/train.jsonl \
                             --val-records r1/data/mydata/val.jsonl

What it does, following section 13's "Data boundaries":

  * reads UTF-8 text and skips anything that is not valid UTF-8
  * deduplicates whole documents by SHA-256
  * assigns **whole documents** to train/validation/test in 90/5/5, so no
    document contributes bytes to two splits
  * cuts each document into records of at most 256 bytes, never cutting inside
    a UTF-8 character
  * records every source path, size and hash in a manifest

Instruction data (section 14) is supported with --format chat: a .jsonl file
whose lines are {"user": "...", "assistant": "..."} becomes records carrying
the literal markers section 14 asks for.

    User: <question>
    Assistant: <answer>

Section 14 is worth quoting before you use that: markers supply "a task
convention; it does not create an architectural instruction-following module",
and they should only be introduced "after a language signal exists".
"""

from __future__ import annotations

import argparse, hashlib, json, random, re, sys
from pathlib import Path

TEXT_EXT = {".txt", ".md", ".rst", ".markdown", ".text", ".org", ".tex",
            ".py", ".js", ".ts", ".c", ".h", ".cpp", ".rs", ".go", ".java",
            ".sh", ".sql", ".html", ".css", ".json", ".yaml", ".yml", ".toml",
            ".csv", ".log", ""}

USER_MARK, ASSISTANT_MARK = "User: ", "Assistant: "

# Project Gutenberg wraps every text in a licence header and footer. Left in,
# they are identical across every book and become the most repeated text in the
# corpus -- exactly the "repeated templates and boilerplate" a byte model
# overfits first.
GUT_START = re.compile(rb"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
GUT_END = re.compile(rb"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)


def strip_gutenberg(raw: bytes) -> tuple[bytes, bool]:
    """Remove the Project Gutenberg licence header and footer, if present."""
    m = GUT_START.search(raw)
    start = m.end() if m else 0
    m2 = GUT_END.search(raw, start)
    end = m2.start() if m2 else len(raw)
    if start or m2:
        return raw[start:end].strip() + b"\n", True
    return raw, False


def utf8_safe_cut(buf: bytes, limit: int) -> int:
    if len(buf) <= limit:
        return len(buf)
    k = limit
    while k > 0 and (buf[k] & 0xC0) == 0x80:
        k -= 1
    return k or limit


def records_of(doc: bytes, limit: int) -> list[bytes]:
    out, i = [], 0
    while i < len(doc):
        k = utf8_safe_cut(doc[i:], limit)
        chunk = doc[i:i + k]
        if chunk.strip():
            out.append(chunk)
        i += k
    return out


def iter_inputs(paths: list[str], exts: set[str], max_bytes: int):
    for p in paths:
        path = Path(p).expanduser()
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(f for f in path.rglob("*") if f.is_file())
        else:
            print(f"  ! not found: {path}", file=sys.stderr)
            continue
        for f in files:
            if exts and f.suffix.lower() not in exts:
                continue
            try:
                raw = f.read_bytes()
            except OSError as e:
                print(f"  ! unreadable {f}: {e}", file=sys.stderr)
                continue
            if len(raw) > max_bytes:
                print(f"  ! skipping {f}: {len(raw)/1e6:.1f} MB over the limit")
                continue
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                print(f"  ! skipping {f}: not valid UTF-8")
                continue
            if b"\x00" in raw:
                continue
            yield f, raw


def load_chat(path: Path) -> list[bytes]:
    """One record per exchange, with the literal markers from section 14."""
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        u, a = obj.get("user"), obj.get("assistant")
        if not u or not a:
            continue
        docs.append(f"{USER_MARK}{u}\n{ASSISTANT_MARK}{a}\n".encode("utf-8"))
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="files and/or folders of UTF-8 text")
    ap.add_argument("--out", default="r1/data/mydata")
    ap.add_argument("--format", default="text", choices=["text", "chat"],
                    help="'chat' reads .jsonl of {user, assistant} pairs")
    ap.add_argument("--limit", type=int, default=256,
                    help="maximum record size in bytes (section 13 says 256)")
    ap.add_argument("--ext", nargs="*", default=None,
                    help="restrict to these extensions, e.g. --ext .md .txt")
    ap.add_argument("--max-file-mb", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--split", nargs=3, type=float, default=[0.90, 0.05, 0.05],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--keep-gutenberg-boilerplate", action="store_true",
                    help="do not strip Project Gutenberg licence header/footer")
    ap.add_argument("--sections", type=int, default=2000,
                    help="when the input is only a handful of documents, cut "
                         "them into sections of at least this many bytes so a "
                         "document-level split is possible at all; 0 disables")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exts = set(e.lower() if e.startswith(".") else "." + e.lower()
               for e in args.ext) if args.ext else TEXT_EXT

    docs, sources, seen = [], [], set()
    skipped_dupes = 0

    if args.format == "chat":
        for p in args.input:
            path = Path(p).expanduser()
            files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
            for f in files:
                for d in load_chat(f):
                    h = hashlib.sha256(d).hexdigest()
                    if h in seen:
                        skipped_dupes += 1
                        continue
                    seen.add(h)
                    docs.append(d)
                    sources.append({"path": str(f), "bytes": len(d), "sha256": h})
    else:
        n_gut = 0
        for f, raw in iter_inputs(args.input, exts, int(args.max_file_mb * 1e6)):
            if not args.keep_gutenberg_boilerplate:
                raw, was_gut = strip_gutenberg(raw)
                n_gut += bool(was_gut)
            if not raw.endswith(b"\n"):
                raw += b"\n"
            h = hashlib.sha256(raw).hexdigest()
            if h in seen:
                skipped_dupes += 1
                continue
            seen.add(h)
            docs.append(raw)
            sources.append({"path": str(f), "bytes": len(raw), "sha256": h})

    if not docs:
        print("no usable documents found", file=sys.stderr)
        return 1

    # A document-level split needs documents to split. One big file cannot be
    # divided 90/5/5 at document level, so cut it into sections at blank lines.
    # Sections of one source are NOT independent the way separate documents
    # are, so this leaks more than a true document split and the manifest says
    # so plainly.
    sectioned = False
    if args.sections and len(docs) < 10:
        new_docs, new_src = [], []
        for d, src in zip(docs, sources):
            parts, buf = [], b""
            for para in d.split(b"\n\n"):
                buf += para + b"\n\n"
                if len(buf) >= args.sections:
                    parts.append(buf); buf = b""
            if buf.strip():
                parts.append(buf)
            for k, part in enumerate(parts):
                new_docs.append(part)
                new_src.append({**src, "section": k, "bytes": len(part)})
        if len(new_docs) > len(docs):
            print(f"[ingest] {len(docs)} document(s) cut into {len(new_docs)} "
                  f"sections so a document-level split is possible")
            print(f"[ingest] WARNING: sections of one source are not "
                  f"independent. Held-out numbers from this split are "
                  f"optimistic; treat them as a memorisation check, not as "
                  f"generalisation.")
            docs, sources = new_docs, new_src
            sectioned = True

    total = sum(len(d) for d in docs)
    print(f"[ingest] {len(docs)} documents, {total/1e6:.2f} MB"
          + (f" ({skipped_dupes} duplicates removed)" if skipped_dupes else ""))
    if args.format == "text" and locals().get("n_gut"):
        print(f"[ingest] stripped Project Gutenberg boilerplate from "
              f"{n_gut} file(s)")
    if total < 1_000_000:
        print(f"[ingest] note: {total/1e6:.2f} MB is small. Section 13 works "
              f"from 20 MiB; less than about 1 MB will mostly teach the model "
              f"to memorise these documents rather than the language in them.")

    order = list(range(len(docs)))
    random.Random(args.seed).shuffle(order)
    n = len(order)
    tr, va, _te = args.split
    n_tr, n_va = int(n * tr), int(n * va)
    groups = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va],
              "test": order[n_tr + n_va:]}
    # A tiny corpus rounds the smaller splits to nothing. Validation is the
    # split the search selects checkpoints on, so it must never be empty.
    for name in ("val", "test"):
        if groups[name]:
            continue
        donor = "train" if len(groups["train"]) > 1 else (
            "test" if name == "val" and groups["test"] else None)
        if donor:
            groups[name] = [groups[donor].pop()]
    if not groups["val"]:
        print("[ingest] warning: too few documents to hold out a validation "
              "split. Add more documents, or the search has nothing "
              "independent to select on.")

    manifest = {"seed": args.seed, "format": args.format,
                "sectioned_single_source": sectioned,
                "record_limit": args.limit,
                "split_rule": "whole documents (section 13)",
                "inputs": args.input, "documents": len(docs),
                "bytes": total, "duplicates_removed": skipped_dupes,
                "sources": sources[:2000], "splits": {}}

    for name, idxs in groups.items():
        recs = []
        for i in idxs:
            recs.extend(records_of(docs[i], args.limit))
        path = out / f"{name}.jsonl"
        with open(path, "w") as fh:
            for r in recs:
                fh.write(r.hex() + "\n")
        b = sum(len(r) for r in recs)
        manifest["splits"][name] = {
            "documents": len(idxs), "records": len(recs), "bytes": b,
            "mean_record_bytes": round(b / max(len(recs), 1), 1),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"[ingest] {name}: {len(idxs)} docs -> {len(recs):,} records, "
              f"{b/1e6:.3f} MB")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ingest] wrote {out}/train.jsonl, val.jsonl, test.jsonl, manifest.json")
    print(f"\nNext:\n  python3 r1/run_search.py --records {out}/train.jsonl "
          f"--val-records {out}/val.jsonl \\\n"
          f"        --bias-init spread --probe-init zero --routing pairdiff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
