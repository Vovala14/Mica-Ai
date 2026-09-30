#!/usr/bin/env python3
"""Run everything: verify, race the backends, then train with all the fixes.

Three things go into this run that are not in R1 as written, each measured
rather than assumed, and each recorded in the result file:

  compiled scorer     merges duplicate reads and drops zero coefficients, then
                      scores in C across threads. Bit-exact; measured 5.1x.
  held-out acceptance a winning child is re-tested on fresh records before it
                      is accepted. Section 13 proposes and accepts on the same
                      batch, which on a flat landscape selects the luckiest
                      child rather than the best one and walks the incumbent
                      downhill -- observed directly at 40 s/round on a 9070 XT.
  routing / init      pairdiff page routing, spread candidate biases, probe
                      BIASES seeded from the corpus unigram, and probe
                      COEFFICIENTS left random so the field reaches the output.
  activity ceiling    finding 5.4 asks for a bound on rewrites per symbol.
                      Anchored to the model's measured starting activity
                      (1.25x) rather than an absolute number: a fixed cap of
                      600 against a model that naturally runs 733 rejected
                      every child for 30 rounds without scoring one.
  readout scale       R1 section 12 divides the probe score by 16. Measured, a
                      settled field puts 2.26 nats of noise on the logits
                      before anything is learned, which is why random
                      coefficients start worse than uniform. The divisor is not
                      a field of the model file, so it is free to set. At 128
                      a random-coefficient readout costs 0.085 bits against a
                      dead one, and a typical two-mutation child moves the loss
                      by 0.00049 bits against a measured 32-record noise floor
                      of 0.00017 -- about three to one, which is what a
                      selection test needs. At 64 the child effect is twice as
                      large but the readout handicap is 0.31 bits; at 256 the
                      handicap is 0.027 but the ratio falls to 1.5.

Why this matters, twice over. Zeroing the probe coefficients to start at
uniform disconnects the field from the output, so two of section 13's three
mutation kinds cannot change the loss at all. Measured over 100 rounds:
accept_rate 0.000, loss fixed at 8.0056. Seeding the BIASES from the unigram
moved the starting point but kept the coefficients at zero, so the field was
still disconnected: over 500 rounds the search tuned a byte histogram, held-out
loss drifted from 4.9461 up to 4.9495, and of six trial children with two
mutations each, five changed the loss by less than 1e-5 bits.

run_search.py now runs an exact liveness check at startup -- children mutated
only in ways that reach the loss through the cell field -- and says so loudly
if none of them can move it.
"""

from __future__ import annotations

import json, os, platform, subprocess, sys, time
from pathlib import Path

# Must be set before mica_r1.spec is imported anywhere, including in the
# training subprocess, which inherits this environment.
os.environ.setdefault("MICA_LOGIT_DIVISOR", "128")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "data"))
LOG = ROOT.parent / "mica_run.log"


def say(msg: str = "") -> None:
    print(msg, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


def run(args) -> int:
    say(f"$ {' '.join(str(a) for a in args)}")
    p = subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    for line in p.stdout:
        say(line.rstrip())
    return p.wait()


def time_backend(models, records, backend: str, score: str, device: str) -> float:
    """Time one real full round. No extrapolation."""
    if backend == "torch":
        from mica_r1.torch_batch import TorchModelStack, evaluate
        import torch
        dev = torch.device(device)
        ms = TorchModelStack(models[:2], dev, routing="pairdiff")
        evaluate(ms, records[:2])                      # warm up kernels
        ms = TorchModelStack(models, dev, routing="pairdiff")
        torch.cuda.synchronize() if device == "cuda" else None
        t = time.perf_counter()
        evaluate(ms, records)
        torch.cuda.synchronize() if device == "cuda" else None
        return time.perf_counter() - t
    from mica_r1.batch import ModelStack, evaluate
    ms = ModelStack(models, score_backend=score)
    t = time.perf_counter()
    evaluate(ms, records)
    return time.perf_counter() - t


def main() -> int:
    try:
        LOG.write_text("", encoding="utf-8")
    except OSError:
        pass
    say("=" * 64)
    say(f" MICA   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say("=" * 64)
    say(f"python     {sys.version.split()[0]}")
    say(f"host       {platform.platform()}")

    have_gpu = False
    try:
        import torch
        have_gpu = torch.cuda.is_available()
        say(f"torch      {torch.__version__}")
        say(f"GPU        {torch.cuda.get_device_name(0) if have_gpu else 'none'}")
    except ImportError:
        say("torch      not installed")

    from mica_r1 import engine
    engine.ROUTING_MODE = "pairdiff"
    from mica_r1.engine import random_model
    from make_records import load_records

    train = ROOT / "r1" / "data" / "full" / "train.jsonl"
    if not train.exists():
        say(f"\nMissing corpus: {train}")
        return 1

    # ---- verify ----------------------------------------------------------
    say("\n--- conformance tests ---")
    if run([sys.executable, "-m", "pytest", "r1/tests", "-q"]) != 0:
        say("TESTS FAILED - stopping.")
        return 1

    # ---- race the backends on one real round -----------------------------
    say("\n--- racing the backends on one full round ---")
    say("    17 models x 32 records x 256 bytes, the real thing")
    from mica_r1.search import XorShift32, make_child
    recs = [r for r in load_records(str(train))[:3000] if len(r) >= 256][:32]
    recs = [r[:256] for r in recs]
    base = random_model(1)
    rng = XorShift32(1)
    models = [base] + [make_child(base, rng) for _ in range(16)]

    results = {}
    try:
        results["numpy + C scorer"] = time_backend(models, recs, "numpy", "c", "cpu")
        say(f"    numpy + C scorer : {results['numpy + C scorer']:7.1f} s/round")
    except Exception as exc:
        say(f"    numpy + C scorer : unavailable ({exc})")
    if have_gpu:
        try:
            results["torch GPU"] = time_backend(models, recs, "torch", "", "cuda")
            say(f"    torch on GPU     : {results['torch GPU']:7.1f} s/round")
        except Exception as exc:
            say(f"    torch on GPU     : failed ({exc})")

    if not results:
        say("No usable backend. Stopping.")
        return 1
    best = min(results, key=results.get)
    say(f"\n    winner: {best} at {results[best]:.1f} s/round")
    say(f"    500 rounds = {results[best]*500/3600:.1f} hours")

    if best == "torch GPU":
        backend, score, device = "torch", "numpy", "cuda"
    else:
        backend, score, device = "numpy", "c", "cpu"

    # ---- train -----------------------------------------------------------
    say("")
    say("=" * 64)
    say(f" Training: {best}, held-out acceptance, 256-byte records")
    say(" Progress: mica/r1/runs/best/progress.json")
    say(" Ctrl-C to stop; checkpoints are saved on validation improvements.")
    say("=" * 64)
    cmd = [sys.executable, "r1/run_search.py",
           "--records", "r1/data/full/train.jsonl",
           "--val-records", "r1/data/full/val.jsonl",
           "--backend", backend, "--score-backend", score, "--device", device,
           "--restarts", "1", "--rounds", "500",
           "--batch-records", "32", "--record-bytes", "256",
           "--accept-holdout", "32",
           "--bias-init", "spread", "--probe-init", "unigram-live",
           "--mutations", "2",
           "--routing", "pairdiff", "--activity-headroom", "1.25",
           "--val-records-n", "64", "--validate-every", "10", "--log-every", "5",
           "--out", "r1/runs/best"]
    rc = run(cmd)

    say("\n" + "=" * 64)
    say(" What to look for")
    say("=" * 64)
    say(" Two lines to read before anything else.")
    say("   'liveness' must say 5/8 or more. 0/8 means the field is")
    say("     disconnected from the output and the run is worthless.")
    say("   'activity' must NOT print a warning. A cap at or below the")
    say("     model's own activity rejects every child before scoring.")
    say("   Either one, stop the run and tell Claude.")
    say("")
    say(" The model STARTS around 5.14 bits: the corpus unigram at 4.93 plus")
    say("   about 0.09 for the random readout it has to tidy up. The unigram")
    say("   part is free - a byte histogram - and is not progress.")
    say("")
    say(" Watch the distance BELOW 4.96. That is the only number that is the")
    say("   model's own work. Reference points at the same 86,820 bytes:")
    say("     unigram floor .......... 4.93   (free)")
    say("     storage-matched 2-gram . 3.29   <- the section 15 gate")
    say("     byte GRU ............... 2.98")
    say("")
    say(" accept_rate should fall over time as the held-out test starts")
    say("   refusing lucky children; holdout_rejections counts those.")
    say(" The failure to watch for: accept_rate climbing while VALIDATION")
    say("   gets worse. That is the search fitting the proposing batch. It is")
    say("   what the last run did - 232 accepts, all of them noise.")
    say("\nSend mica_run.log and r1/runs/best/progress.json to Claude.")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped - progress saved")
        sys.exit(130)
