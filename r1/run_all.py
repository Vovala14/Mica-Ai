#!/usr/bin/env python3
"""One-shot launcher: check, test, benchmark, train.

Replaces the batch script. Batch files fail silently in ways that are painful
to debug on someone else's machine; this reports every step and why it stopped.
"""

from __future__ import annotations

import os, platform, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../mica/r1
ROOT = HERE.parent                              # .../mica
LOG = ROOT.parent / "mica_run.log"
sys.path.insert(0, str(HERE))


def say(msg: str = "") -> None:
    print(msg, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


def run(args, **kw) -> int:
    """Run a child process, streaming its output to console and log."""
    say(f"$ {' '.join(str(a) for a in args)}")
    p = subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", **kw)
    for line in p.stdout:
        say(line.rstrip())
    return p.wait()


def main() -> int:
    try:
        LOG.write_text("", encoding="utf-8")
    except OSError:
        pass
    say("=" * 60)
    say(f" MICA launcher   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say("=" * 60)
    say(f"python     {sys.version.split()[0]}   {sys.executable}")
    say(f"host       {platform.platform()}")
    say(f"working in {ROOT}")

    # ---- what does this interpreter see? --------------------------------
    backend, device, score_backend = "numpy", "cpu", "numpy"
    try:
        import torch
        gpu = torch.cuda.is_available()
        say(f"torch      {torch.__version__}")
        say(f"GPU        {'YES - ' + torch.cuda.get_device_name(0) if gpu else 'no, CPU only'}")
        if gpu:
            backend, device = "torch", "cuda"
    except ImportError:
        say("torch      NOT INSTALLED in this interpreter")

    if backend == "numpy":
        # compiled CPU scorer: bit-exact, measured 5.1x faster than numpy
        try:
            from mica_r1.c_score import CompiledScorer   # noqa: F401
            score_backend = "c"
            say("scorer     compiled C (5x faster than numpy, bit-exact)")
        except Exception as exc:
            say(f"scorer     numpy ({exc})")
        say("")
        say("!" * 60)
        say(" No GPU from THIS python. On CPU the run takes weeks.")
        say(" Run INSTALL_GPU.bat, then start again.")
        say("!" * 60)
        try:
            answer = input("\nRun on CPU anyway? [y/N] ").strip().lower()
        except EOFError:          # no console attached
            answer = "n"
        if answer != "y":
            return 1

    # ---- data present? ---------------------------------------------------
    train = ROOT / "r1" / "data" / "full" / "train.jsonl"
    val = ROOT / "r1" / "data" / "full" / "val.jsonl"
    if not train.exists():
        say(f"\nMissing corpus: {train}")
        say("Unzip mica_gpu.zip next to the mica folder, or build records with")
        say("  python r1/data/ingest.py --input <your text folder> --out r1/data/mydata")
        return 1
    say(f"corpus     {train.stat().st_size/1e6:.1f} MB train, "
        f"{val.stat().st_size/1e6:.1f} MB val")

    # ---- dependencies ----------------------------------------------------
    try:
        import numpy, pytest            # noqa: F401
    except ImportError:
        say("\nInstalling numpy and pytest ...")
        run([sys.executable, "-m", "pip", "install", "--quiet", "numpy", "pytest"])

    # ---- conformance -----------------------------------------------------
    say("\n--- conformance tests (expect 11 passed) ---")
    if run([sys.executable, "-m", "pytest", "r1/tests", "-q"]) != 0:
        say("TESTS FAILED - stopping.")
        return 1

    # ---- benchmark -------------------------------------------------------
    say("\n--- timing one full round on this machine ---")
    run([sys.executable, "r1/bench.py"])

    # ---- train -----------------------------------------------------------
    say("")
    say("=" * 60)
    say(f" Training on {backend}/{device}. Ctrl-C or close to stop.")
    say(" Progress: mica/r1/runs/gpu/progress.json")
    say("=" * 60)
    cmd = [sys.executable, "r1/run_search.py",
           "--records", "r1/data/full/train.jsonl",
           "--val-records", "r1/data/full/val.jsonl",
           "--backend", backend, "--device", device,
           "--score-backend", score_backend,
           "--restarts", "1", "--rounds", "500",
           "--batch-records", "32", "--record-bytes", "256",
           "--val-records-n", "64", "--validate-every", "10", "--log-every", "5",
           "--bias-init", "spread", "--probe-init", "zero",
           "--routing", "pairdiff", "--activity-cap", "600",
           "--accept-holdout", "32",
           "--out", "r1/runs/gpu"]
    rc = run(cmd)
    say(f"\nfinished with code {rc}")
    say("Send mica_run.log and r1/runs/gpu/progress.json to Claude.")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped by user - progress is saved")
        sys.exit(130)
