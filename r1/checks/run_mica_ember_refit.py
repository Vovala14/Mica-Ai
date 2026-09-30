#!/usr/bin/env python3
"""One bounded low-rate native integer-rule/readout refit of MICA Ember.

The source checkpoint stays intact. Selection uses separate chat dev1000;
clean everyday and chat sets are reserved for reporting after selection.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time

from run_mica_page_block_refit import digest, sample_train, write_json


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
SOURCE = R1 / "runs/codex_ember_chat_20260928/train"
OUT = R1 / "runs/codex_ember_refit_20260928"
SOURCE_SHA = "8f64bc80df1f02b71a0959d84d3df11b033e9f6091d3fc59173684fe85d607ac"


def status(stage: str, detail: str = "") -> None:
    write_json(OUT / "status.json", {
        "stage": stage, "detail": detail, "updated": time.time()})


def main() -> None:
    existing = {item.name for item in OUT.iterdir()} if OUT.exists() else set()
    if existing - {"stdout.log", "stderr.log", "status.json"}:
        raise RuntimeError(f"refusing populated output directory: {OUT}")
    if "status.json" in existing:
        previous = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
        if previous.get("stage") != "failed":
            raise RuntimeError(f"refusing active output directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    status("preparing")
    if digest(SOURCE / "best.mica") != SOURCE_SHA:
        raise RuntimeError("Ember source checkpoint hash changed")
    if not (SOURCE / "FINISHED").is_file():
        raise RuntimeError("Ember source fit is unfinished")
    info = json.loads((SOURCE / "run_info.json").read_text(encoding="utf-8"))
    for key in list(os.environ):
        if key.startswith("MICA_"):
            del os.environ[key]
    os.environ.update({k: str(v) for k, v in info["env"].items()})
    os.environ["PYTHONUTF8"] = "1"
    sys.path.insert(0, str(R1))
    from mica_r1 import fit, serialize, spec
    from mica_r1.discretise import to_integer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for Ember low-rate refit")
    if spec.N_CANDIDATES != 256 or spec.N_PHASE != 16 or spec.VSET_WIDTH != 6:
        raise RuntimeError("unexpected original Ember cellular geometry")
    train_path = R1 / "data/chat/train.jsonl"
    preflight = R1 / "runs/codex_ember_chat_20260928/preflight.json"
    audit = json.loads(preflight.read_text(encoding="utf-8"))
    if (digest(R1 / "data/chat/manifest.json") != audit["manifest_sha256"] or
            digest(R1 / "data/chat/dev1000.jsonl") != audit["dev1000_sha256"] or
            digest(R1 / "data/eval_clean/chat_val1000.jsonl") != audit["chat_clean_sha256"] or
            digest(R1 / "data/eval_clean/val1000.jsonl") != audit["everyday_clean_sha256"]):
        raise RuntimeError("corpus or held-out set changed since preflight")
    # Uniform deterministic sample from the approved training split only.
    # Preflight verified that the whole train split has no exact dev/clean overlap.
    train = sample_train(train_path, 40000, set())
    write_json(OUT / "protocol.json", {
        "source": str(SOURCE / "best.mica"), "source_sha256": SOURCE_SHA,
        "train_file": str(train_path), "train_manifest_sha256": audit["manifest_sha256"],
        "train_sample_records": len(train),
        "train_sample_seed": 20260927, "fit_seed": 20260928,
        "steps": 1500, "max_records": 20000, "check_every": 250,
        "lr_scale": 0.01,
        "selection": "accept only if exact integer chat dev1000 delta <= -0.005 bits/target with paired 95% CI upper below zero; clean sets report only",
        "architecture": "original learned integer-rule cellular automaton; selectors fixed, VSET immediates and readout refitted",
    })
    status("refitting")
    torch.manual_seed(20260928)
    model = fit.from_integer(serialize.load(SOURCE / "best.mica"),
                             spec.MAX_TICKS).to(torch.device("cuda"))
    initial = OUT / "initial.mica"
    serialize.save(to_integer(model), initial)
    if digest(initial) != SOURCE_SHA:
        raise RuntimeError("soft conversion changed the integer checkpoint")
    with (OUT / "refit.log").open("w", encoding="utf-8") as log:
        fit.fit_readout_and_rules(
            model, train, torch.device("cuda"), spec.MAX_TICKS,
            steps=1500, max_records=20000, check_every=250,
            lr_scale=0.01,
            log=lambda line: print(line, file=log, flush=True))
    candidate = OUT / "candidate"
    candidate.mkdir()
    serialize.save(to_integer(model), candidate / "best.mica")
    shutil.copy2(SOURCE / "run_info.json", candidate / "run_info.json")
    status("complete", digest(candidate / "best.mica"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", f"{type(exc).__name__}: {exc}")
        raise
