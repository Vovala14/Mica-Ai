#!/usr/bin/env python3
"""Turn the MICA test site's logs into training records for Ember and Flame-W.

    python r1/data/ingest_site_logs.py mica-site-logs.jsonl --out r1/data/site/2026-10-01

Input is the owner export (curl -H "Authorization: Bearer ADMIN_KEY" <site>/api/export), one JSON object per line:
"generation" rows (prompt, model output) and "feedback" rows (rating up/down
and an optional user-written continuation, the "correction").

Which text becomes training data, per generation, using its latest feedback:
  1. a correction:     prompt + correction  (a person wrote how it should go on)
  2. a thumbs-up:      prompt + model output
  3. otherwise:        the prompt alone     (text people typed; --no-prompts drops it)
A thumbs-down output is never used. Fixed rating-set prompts (prompt_id) are
excluded from training to preserve that evaluation set. Texts are deduplicated
(casefolded, whitespace squashed) and split train/val by a hash of the prompt,
so corrections to the same prompt always land on the same side.

Outputs under --out:
  ember/train.jsonl, ember/val.jsonl    UTF-8 bytes, hex, <= 256 bytes a record
                                        (the byte trainer's record format)
  flamew/train.jsonl, flamew/val.jsonl  word ids (build_word_corpus.pack), the
                                        Flame-W vocabulary and tokenizer
  manifest.json                         counts, sources and SHA-256 of every file

These records are small next to the original corpora. Mix them into the
existing training mix rather than training on them alone: a narrow set shifts
the model toward it and costs general text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_word_corpus as W  # noqa: E402
from make_records import utf8_safe_cut  # noqa: E402

VOCAB = HERE / "word" / "vocab.json"
MAX_BYTES = 256


def join(prompt: str, cont: str) -> str:
    if not cont:
        return prompt
    if cont[0].isspace() or cont[0] in ".,!?;:'\")" or not prompt or prompt[-1].isspace():
        return prompt + cont
    return prompt + " " + cont


def completed_word(prompt: str, correction: str) -> str:
    """The Ember completion UI asks for missing letters, not a new word."""
    if correction[:2].casefold() == prompt[-2:].casefold():
        correction = correction[2:]  # also accept a user who typed the full word
    return prompt + correction


def texts(rows: list[dict], with_prompts: bool) -> list[tuple[str, str, str]]:
    gens = {r["id"]: r for r in rows if r.get("kind") == "generation"}
    fb: dict[str, dict] = {}
    for r in sorted((r for r in rows if r.get("kind") == "feedback"), key=lambda r: r.get("time", "")):
        # A later vote can explicitly replace an earlier correction. Do not
        # resurrect stale text from the previous feedback event.
        fb[r["id"]] = {"rating": r.get("rating"), "correction": r.get("correction"), "row": r}
    out = []
    for gid in list(gens) + [i for i in fb if i not in gens]:
        g = gens.get(gid) or fb[gid]["row"]
        f = fb.get(gid, {})
        prompt, output = g.get("prompt", ""), g.get("output", "")
        if not isinstance(prompt, str) or not prompt or g.get("prompt_id") or f.get("row", {}).get("prompt_id"):
            continue
        if f.get("correction"):
            correction = f["correction"].strip()
            if g.get("model", "").startswith("ember-") and g.get("mode") == "complete-word":
                out.append(("correction", completed_word(prompt, correction), prompt))
            else:
                out.append(("correction", join(prompt, correction), prompt))
        elif f.get("rating") == "up":
            out.append(("thumbs_up", prompt + output, prompt))
        elif with_prompts:
            out.append(("prompt", prompt, prompt))
    return out


def key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path, help="mica-site-logs.jsonl from <site>/api/export")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-percent", type=int, default=5)
    ap.add_argument("--no-prompts", action="store_true", help="skip prompts that have no approved continuation")
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.export.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen, kept = set(), []
    for source, text, prompt in texts(rows, not a.no_prompts):
        text = text.strip()
        if len(text) < 3 or key(text) in seen:
            continue
        seen.add(key(text))
        kept.append((source, text, prompt))

    vocab = W.Vocab.load(VOCAB)
    files = {f"{m}/{s}": [] for m in ("ember", "flamew") for s in ("train", "val")}
    counts = {"train": {}, "val": {}}
    for source, text, prompt in kept:
        split = "val" if int(hashlib.sha256(key(prompt).encode()).hexdigest(), 16) % 100 < a.val_percent else "train"
        counts[split][source] = counts[split].get(source, 0) + 1
        raw = text.encode("utf-8")
        files[f"ember/{split}"].append(raw[:utf8_safe_cut(raw, MAX_BYTES)].hex())
        ids = vocab.encode(W.tokenize(text))
        if ids:
            files[f"flamew/{split}"].append(W.pack(ids))

    manifest = {"source": str(a.export), "export_sha256": W.sha256_file(a.export),
                "rows": len(rows), "texts": len(kept), "counts": counts, "files": {}}
    for name, lines in files.items():
        path = a.out / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        manifest["files"][f"{name}.jsonl"] = {"records": len(lines), "sha256": W.sha256_file(path)}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"texts": len(kept), "counts": counts}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
