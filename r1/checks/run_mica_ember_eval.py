#!/usr/bin/env python3
"""Exact integer held-out evaluation of completed native MICA Ember."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from common import paired_bootstrap


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
OUT = R1 / "runs/codex_ember_chat_20260928"
EMBER = OUT / "train"
OLD_BEST = R1 / "runs/codex_page_block_refit_20260927_v2/control"
SETS = (
    ("chat_dev1000", R1 / "data/chat/dev1000.jsonl"),
    ("chat_clean_val1000", R1 / "data/eval_clean/chat_val1000.jsonl"),
    ("everyday_clean_val1000", R1 / "data/eval_clean/val1000.jsonl"),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def status(stage: str, detail: str = "") -> None:
    save(OUT / "eval_status.json", {"stage": stage, "detail": detail,
                                    "updated": time.time()})


def main() -> None:
    if not (EMBER / "FINISHED").is_file():
        raise RuntimeError("Ember native fit is not finished")
    before = {"ember": sha(EMBER / "best.mica"),
              "old_best": sha(OLD_BEST / "best.mica")}
    paths = {name: sha(path) for name, path in SETS}
    result = {"protocol": "exact exported integer C engine, pooled bytes plus EOS from BOS; paired 5,000-resample record bootstrap for Ember minus prior mature MICA",
              "models": {"ember": str(EMBER / "best.mica"),
                         "old_best": str(OLD_BEST / "best.mica")},
              "model_sha256": before, "set_sha256": paths, "scores": {}}
    save(OUT / "eval_results.json", result)
    for name, path in SETS:
        status("evaluating", name)
        out = OUT / f"eval_{name}.json"
        cmd = [sys.executable, str(R1 / "checks/eval_int.py"),
               "--run", f"old_best={OLD_BEST}", "--run", f"ember={EMBER}",
               "--set", "val1000", "--val-file", str(path),
               "--val-skip", "0", "--out", str(out)]
        with (OUT / "eval.log").open("a", encoding="utf-8") as log:
            p = subprocess.run(cmd, cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT)
        if p.returncode:
            raise RuntimeError(f"exact integer evaluation failed for {name}: {p.returncode}")
        rows = json.loads(out.read_text(encoding="utf-8"))
        if set(rows) != {"old_best", "ember"}:
            raise RuntimeError(f"missing exact model score for {name}")
        a, b = rows["ember"], rows["old_best"]
        if a["counts"] != b["counts"] or a["records"] != 1000:
            raise RuntimeError(f"unaligned or incomplete records for {name}")
        result["scores"][name] = {
            "ember_bits": a["bits"], "old_best_bits": b["bits"],
            "ember_minus_old_best": paired_bootstrap(
                a["nats"], b["nats"], a["counts"], n_boot=5000),
            "records": a["records"], "targets": a["targets"],
        }
        save(OUT / "eval_results.json", result)
    if {key: sha(path) for key, path in
            (("ember", EMBER / "best.mica"),
             ("old_best", OLD_BEST / "best.mica"))} != before:
        raise RuntimeError("model changed during evaluation")
    status("generating")
    cmd = [sys.executable, str(R1 / "checks/eval_mica_raw_generation.py"),
           "--run", str(EMBER), "--prompts",
           str(R1 / "checks/generation_dev10.json"), "--limit", "10",
           "--byte-limit", "100", "--out", str(OUT / "generation.json")]
    with (OUT / "eval.log").open("a", encoding="utf-8") as log:
        p = subprocess.run(cmd, cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"raw generation failed: {p.returncode}")
    status("complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise
