#!/usr/bin/env python3
"""Launcher for gradient training at the scaled geometry.

Sets the geometry, times one real step, then trains. Everything it configures
was measured rather than guessed; see docs/r1-findings.md 5.8.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from procs import lock_holder   # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOG = ROOT.parent / "mica_train.log"

CORPUS_FILES = {
    "everyday": ("r1/data/everyday/train.jsonl",
                 "r1/data/everyday/val.jsonl"),
    "bulk": ("r1/data/bulk/train.jsonl", "r1/data/bulk/val.jsonl"),
    "gsm8k": ("r1/data/gsm8k/train.jsonl", "r1/data/gsm8k/val.jsonl"),
}


def unused_sweep_dir(sweep: Path, name: str) -> Path:
    """Keep each sweep checkpoint, including repeated configuration names."""
    first = sweep / name
    if not first.exists():
        return first
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    for index in range(1000):
        candidate = sweep / f"{name}_{stamp}_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot allocate a new sweep folder for {name}")


def resume_corpus_problem(run_dir: Path, records: str,
                          val_records: str, root: Path = ROOT) -> str | None:
    """Refuse to reuse an existing run with a different training corpus.

    train_soft.py's resume signature does not bind the checkpoint to its data,
    so a shape-compatible checkpoint can otherwise silently change corpora.
    Check before launching even the GPU timing run, and before train_soft.py
    replaces run_info.json with the newly requested arguments.
    """
    if not any((run_dir / name).exists() for name in ("resume.pt", "best.mica")):
        return None
    info_path = run_dir / "run_info.json"
    try:
        previous = json.loads(info_path.read_text(encoding="utf-8"))["args"]
        old_train, old_val = previous["records"], previous["val_records"]
        if not isinstance(old_train, str) or not isinstance(old_val, str):
            raise TypeError("corpus paths must be strings")
    except (OSError, ValueError, KeyError, TypeError):
        return (f"{run_dir} contains a model or checkpoint but has no usable "
                "run_info.json; its training corpus cannot be verified")

    def absolute(path: str) -> Path:
        p = Path(path)
        return (p if p.is_absolute() else root / p).resolve()

    if (absolute(old_train), absolute(old_val)) != \
            (absolute(records), absolute(val_records)):
        return (f"{run_dir} was trained on {old_train} and {old_val}; "
                f"requested {records} and {val_records}")
    return None

# ---- geometry: ~4.25 MB ------------------------------------------------
# cells x channels is the working field and costs ZERO bytes in the model
# file. pages x candidates is the rule book and is 99% of it. Eight routing
# bits give 1,024 pages where R1's four give 64.
GEOM = {"MICA_CELLS": "768",
        # 16 tape channels + 96 work channels: sixteen groups of six, one per
        # phase, each written whole by a VSET rule (below).
        "MICA_CHANNELS": "112",
        # 256 candidates per page on 512 pages (below: five routing bits):
        # the same 131,072 rules and the same file size as 128 x 1,024.
        # Exact integer scores on 1,000 validation records, one self term
        # each: 128 candidates / six bits 2.3136, 256 / five bits 2.2438.
        "MICA_CANDIDATES": "256", "MICA_INJECT": "24",
        # 240 probes per symbol: 48 read the byte tape, 192 read rule-written
        # channels (all 96, one byte back and two). The probe table is 6% of
        # the file; the model file is 4,467,396 bytes.
        "MICA_PROBE": "240",
        # Vector SET: a winning rule writes six values at once -- its own
        # immediate plus five stored in the candidate record's spare bytes,
        # so the rule book does not grow. On the PC, same everything else:
        # one value per rule 3.25-3.31 bits/byte, six values 3.02; with four
        # ticks per byte 2.975.
        "MICA_VSET": "6",
        # Sixteen phases (R1: four) with five routing bits: 512 pages, each
        # byte class gets sixteen rule pages, and with sixteen ticks per byte
        # the head cell carries sixteen different rule features. Measured,
        # first round: 4 phases 2.92, 8 phases 2.77, 16 phases 2.70 bits/byte.
        # A page holds ~8 byte values; each candidate's self term (below)
        # tells them apart.
        "MICA_PHASES": "16", "MICA_TICKS": "16",
        "MICA_ROUTING_CHANNELS": "0,1,2,3,4",
        # The soft model always routes by pairdiff; say so, so the integer
        # checks and the saved file run the same machine (until 2026-09-25
        # they silently ran R1's sign routing).
        "MICA_ROUTING": "pairdiff",
        # Probes read relative to the write head (spec.ROLLING_READOUT). With
        # R1's absolute probes the first 4.25 MB run could not see recent
        # bytes and sat at the byte-frequency level for 1,100 steps.
        "MICA_ROLLING_READOUT": "1",
        # ---- the tape machine (spec.py, "tape extensions") ----------------
        # Run 2 plateaued at 4.06-4.09 bits: after 4,800 steps its field did
        # not even hold the previous byte cleanly. Now channels 0-15 of each
        # cell are a byte tape no rule can overwrite, the rules work only on
        # the 16 cells behind the write head (48x less work per byte, and the
        # same machine in training and in use), probes read the last 32
        # bytes with int8 coefficients, and neighbours look only backwards.
        # Rules update only the head cell. With the structured rule book and
        # as many ticks per byte as phases, the cells behind the head would
        # recompute exactly the values they already hold (their scoring reads
        # only the frozen tape), so a 1-cell window gives the identical model
        # -- measured: same validation loss to 6 decimals -- at 1/16 of the
        # work, in training and on the user's machine alike.
        "MICA_TAPE": "16", "MICA_WINDOW": "1", "MICA_PROBE_WINDOW": "32",
        "MICA_PROBE_COEF": "int8", "MICA_LOGIT_DIVISOR": "1024",
        "MICA_OFFSETS": "-1,-2,-3,-4,-6,-8"}
os.environ.update(GEOM)
# (expandable_segments is unsupported by ROCm on Windows -- the log said so
#  -- so it is not set. The VRAM cap in train_soft.py is the real guard.)


def say(msg: str = "") -> None:
    print(msg, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


def run(args, keep: list | None = None, env: dict | None = None) -> int:
    say(f"$ {' '.join(str(a) for a in args)}")
    # Below-normal priority for the training child, so the desktop always
    # gets the CPU first. (Windows children also inherit it from us.)
    flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    p = subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace",
                         creationflags=flags, env=env)
    for line in p.stdout:
        say(line.rstrip())
        if keep is not None:
            keep.append(line.rstrip())
    return p.wait()


def dry_run_batch(lines: list) -> int | None:
    """The batch the dry run fell back to: '[dry-run] batch reduced to N'."""
    for ln in lines:
        if ln.startswith("[dry-run] batch reduced to"):
            try:
                return int(ln.split("reduced to")[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def dry_run_hours(lines: list) -> float | None:
    """The trainer's own estimate: '[dry-run] N steps + M validations = H hours'."""
    for ln in lines:
        if ln.startswith("[dry-run]") and ln.rstrip().endswith("hours"):
            try:
                return float(ln.split("=")[-1].split()[0])
            except ValueError:
                return None
    return None


# Longest record length whose whole run fits in about this many hours.
# The run is launch-bound: each byte of a record is one more sequential step
# of ~1,600 small GPU operations, so a step's time is set by record LENGTH far
# more than by batch size. The gradient only reaches 8 bytes back (--tbptt),
# so shorter records cost little for learning; they cut the field's training
# context from 1,024 bytes to 512 or 256.
TARGET_HOURS = 16

# Measured on the first 64 validation records, trimmed to each length, with
# the tables fitted on other held-out documents (bits per byte).
REFERENCE = {"1024": (4.77, 3.80, 3.29), "512": (4.78, 3.82, 3.29),
             "256": (4.79, 3.84, 3.32)}
# Run 2 (before the tape machine), same 64 records at 256 bytes.
RUN2 = "4.25 at step 1,000; 4.06-4.09 from step 3,600 to 4,800 (plateau)"


LOCK = ROOT.parent / "mica_train.lock"


def spec_bytes() -> int:
    """The model file size under the current geometry (spec.TOTAL_BYTES)."""
    try:
        sys.path.insert(0, str(HERE))
        from mica_r1 import spec
        return spec.TOTAL_BYTES
    except Exception:
        return 0


def take_lock() -> bool:
    """One training run at a time.

    Two launches eight seconds apart put two identical runs on one GPU: each
    at half speed, both writing the same best.mica and progress.json. This is
    what happened on the first protected run. With the auto-restart watcher
    and START_TRAIN.bat both able to launch, the lock is created atomically,
    so of two simultaneous starts exactly one wins; and a lock left behind by
    a shutdown counts only while the process that wrote it is still alive
    (procs.lock_holder -- Windows reuses process ids).
    """
    for _ in range(3):
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = lock_holder(LOCK)
            if holder:
                say(f"training is already running (pid {holder}). Not "
                    f"starting a second copy.")
                return False
            try:
                text = LOCK.read_text().strip()
                young = time.time() - LOCK.stat().st_mtime < 30
            except OSError:
                text, young = "", False
            if not text and young:
                # Another launcher created it a moment ago and has not
                # written its pid yet. It is starting; let it.
                say("another launch is starting right now. Not starting a "
                    "second copy.")
                return False
            say(f"removing a stale lock ({text or 'empty'})")
            try:
                LOCK.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    say("could not take the training lock")
    return False


def start_new_run_if_asked() -> None:
    """NEW_RUN_MICA in PycharmProjects: move the current run's files to
    r1/runs/archive/ so the next training starts from scratch, then remove
    the marker. Created remotely (with RESTART_MICA) when a change makes the
    saved state meaningless, e.g. a different readout. The record-length memo
    is kept: the new run times only the length already in use."""
    marker = ROOT.parent / "NEW_RUN_MICA"
    if not marker.exists():
        return
    run = ROOT / "r1/runs/soft"
    memo = run / "record_bytes.txt"
    keep = memo.read_text() if memo.exists() else None
    if run.exists() and any(run.iterdir()):
        dest = ROOT / "r1/runs/archive" / time.strftime("soft_%Y%m%d_%H%M%S")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run), str(dest))
        say(f"NEW_RUN_MICA: previous run moved to {dest}; starting from scratch")
    run.mkdir(parents=True, exist_ok=True)
    if keep:
        memo.write_text(keep)
    marker.unlink()


def run_plan(base, chosen: str) -> None:
    """r1/runs/plan.json: a list of short experiments to run before training,
    written remotely by Claude. Each is a train_soft.py run in its own folder
    under r1/runs/sweep/, with its own geometry and arguments, for a few fit
    rounds; the results go to the log and to r1/runs/sweep/results.json.
    The plan file is renamed as soon as it is read, so a restart never runs
    it twice. The main run's heartbeat is touched between experiments, so the
    auto-restart watcher does not mistake a sweep for a hung run.

      {"configs": [{"name": "...", "env": {"MICA_CHANNELS": "48"},
                    "args": ["--steps", "3"]}, ...]}

    An entry with "script" runs that Python file with "args" instead of a
    training run (the evaluation checks in r1/checks/ are run this way).
    """
    import json
    plan_f = ROOT / "r1/runs/plan.json"
    if not plan_f.exists():
        return
    try:
        plan = json.loads(plan_f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        say(f"plan.json unreadable ({exc}); ignored")
        return
    done = plan_f.with_name(time.strftime("plan_done_%Y%m%d_%H%M%S.json"))
    plan_f.replace(done)
    sweep = ROOT / "r1/runs/sweep"
    sweep.mkdir(parents=True, exist_ok=True)
    hb = ROOT / "r1/runs/soft/heartbeat"
    results = []
    say("")
    say("=" * 66)
    say(f" Experiments before training: {len(plan.get('configs', []))}")
    say("=" * 66)
    for cfg in plan.get("configs", []):
        if (ROOT.parent / "STOP_MICA").exists():
            say("STOP_MICA found: experiments stopped")
            break
        name = str(cfg.get("name", "x")).replace("/", "_")
        env = dict(os.environ)
        env.update(GEOM)
        env.update({k: str(v) for k, v in cfg.get("env", {}).items()})
        run_dir = None
        if cfg.get("script"):
            # a check script (r1/checks/...), not a training run: it is
            # run as given, with this Python, from the mica folder
            args = [base(chosen)[0], str(cfg["script"])] + \
                [str(a) for a in cfg.get("args", [])]
        else:
            run_dir = unused_sweep_dir(sweep, name)
            args = base(chosen) + ["--out", str(run_dir),
                                   "--fresh"] + \
                [str(a) for a in cfg.get("args", [])]
        t = time.time()
        try:
            hb.parent.mkdir(parents=True, exist_ok=True)
            hb.write_text(f"sweep {name} {int(t)}\n")
        except OSError:
            pass
        rc = run(args, env=env)
        best, last = None, None
        if run_dir is not None:
            try:
                prog = json.loads((run_dir / "progress.json").read_text())
                vals = [h["val_bits"] for h in prog.get("history", [])]
                best, last = (min(vals), vals[-1]) if vals else (None, None)
            except (OSError, ValueError, KeyError):
                pass
        results.append({"name": name, "rc": rc, "best_val": best,
                        "last_val": last, "minutes": round((time.time() - t) / 60, 1),
                        "env": cfg.get("env", {}), "args": cfg.get("args", []),
                        "folder": str(run_dir) if run_dir is not None else None})
        say(f"*** experiment {name}: best val {best}  ({results[-1]['minutes']} min)")
        try:
            (sweep / "results.json").write_text(json.dumps(results, indent=2))
            hb.write_text(f"sweep {name} done {int(time.time())}\n")
        except OSError:
            pass
    say("")
    for r in sorted(results, key=lambda r: r["best_val"] or 99):
        say(f"   {r['name']:28s} {r['best_val']}")
    say("")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=tuple(CORPUS_FILES), default="everyday",
                    help="training corpus (default: everyday English; bulk "
                         "includes math Q/A and must be selected explicitly)")
    args = ap.parse_args(argv)

    say("=" * 66)
    say(f" MICA gradient training   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say("=" * 66)
    if not take_lock():
        return 0
    start_new_run_if_asked()

    recs, vals = CORPUS_FILES[args.corpus]
    train_file, val_file = ROOT / recs, ROOT / vals
    missing = [p for p in (train_file, val_file)
               if not p.is_file() or p.stat().st_size == 0]
    if missing:
        say(f"missing or empty {args.corpus} corpus file(s): " +
            ", ".join(str(p) for p in missing))
        if args.corpus == "everyday":
            say("Build it with python r1/data/build_everyday.py after "
                "preparing its source data.")
        return 1
    problem = resume_corpus_problem(ROOT / "r1/runs/soft", recs, vals, ROOT)
    if problem:
        say(f"refusing to resume: {problem}")
        say("To start a new corpus, create NEW_RUN_MICA in PycharmProjects "
            "and relaunch. The existing run will be archived first.")
        return 2
    say(f"corpus: {args.corpus} ({train_file.stat().st_size/1e6:.0f} MB of "
        "training records)")

    try:
        import torch
        say(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            say(f"GPU   {torch.cuda.get_device_name(0)}")
        else:
            say("")
            say("NO GPU VISIBLE. This geometry is 59.8 million soft parameters;")
            say("one step on CPU takes over fifteen minutes. Fix the GPU first.")
    except Exception as exc:
        say(f"torch unavailable: {exc}")
        return 1

    batch = 32

    def base(record_bytes: str) -> list:
        return [sys.executable, "r1/train_soft.py",
                "--records", recs, "--val-records", vals,
                "--record-bytes", record_bytes, "--ticks", "16", "--tau", "0.5",
                # Selectors start near-uniform in the backward pass so every
                # cell and channel gets a gradient (soft.py, sel_init). With
                # the original peaked start only the few initially-top
                # options ever did, and nothing learned from context.
                "--sel-init", "0.05", "--sel-tau", "1.0",
                # Probes start on the tape (a quarter on rule-written
                # channels), and the byte codes, probe coefficients and biases
                # are fitted to the data directly before step 1: the tape
                # readout is an ordinary additive model and needs no
                # straight-through search (train_soft.fit_tape_readout). The
                # rule book starts STRUCTURED -- every candidate SETs a work
                # channel to its own immediate, picked by the byte at the cell
                # and a hash of the bytes before it -- so the immediates can be
                # fitted exactly too (mica_r1/fit.py). Measured locally: tape
                # readout ~3.6 bits/byte on the validation records, tape plus
                # fitted rules 3.27, below a two-letter-context table (3.32).
                "--init", "fit-rules",
                # Training is then rounds of exact refits on fresh samples of
                # 20,000 records (a "step" is a round). Straight-through steps
                # made the fitted model worse: run 3 went 3.60 -> 3.69 in 750.
                "--mode", "fit", "--fit-records", "40000",
                "--fit-steps", "1500", "--round-lr", "0.3",
                # rules: VSET programs, 48 work probes (every work channel, one
                # and two bytes back), scoring reads 1-3 bytes back by phase,
                # biases balanced so a page's candidates win equally often
                "--work-probes", "192",
                "--rule-max-back", "1,2,2,3,2,3,3,4,1,2,3,4,2,3,4,4",
                "--rule-scoring", "balance",
                # One of each candidate's six scoring terms reads the byte at
                # its own cell. Six routing bits put ~4 bytes on a page; this
                # lets the candidates tell them apart. Measured on the PC,
                # 2 rounds: 2.6804 without, 2.6460 with.
                "--rule-self-terms", "1",
                # The windowed machine's graph for 8 bytes is small, so each
                # 8-byte segment is backpropagated and freed at once: no
                # recomputation (checkpointing off), and memory for a batch
                # of 32 records instead of 12. If 32 does not fit the dry run
                # tries 24, 16, ... and training uses what it found.
                "--segments", "--checkpoint-every", "0",
                "--batch", str(batch),
                "--lr", "0.02", "--steps", "600",
                "--val-every", "250", "--save-every", "250",
                "--out", "r1/runs/soft"]

    # The field (cells x channels) is the model's working memory and costs
    # ZERO bytes in the model file -- 768 cells and 384 cells both produce a
    # 4,252,740-byte model, because the rule book is 99% of it. So if the GPU
    # cannot hold the wider field, halving it is nearly free: the deliverable
    # is the same size, only the scratch space shrinks.
    say("")
    say("--- timing one real step ---")
    # Remember the record length between starts. Timing all three lengths
    # costs ~6 minutes of GPU time, and the auto-restart watcher restarts
    # runs; after the first start, only the length already in use (and
    # anything shorter, if it has become too slow) is timed.
    memo = ROOT / "r1/runs/soft/record_bytes.txt"
    lengths = ("1024", "512", "256")
    try:
        prev = memo.read_text().strip()
        if prev in lengths:
            lengths = lengths[lengths.index(prev):]
            say(f"(this run uses {prev}-byte records: timing that length)")
    except OSError:
        pass
    chosen = None
    for cells in ("768", "384", "256"):
        os.environ["MICA_CELLS"] = cells
        if cells != GEOM["MICA_CELLS"]:
            say("")
            say(f"*** retrying with a {cells}-cell field. The model file stays")
            say(f"*** {spec_bytes():,} bytes -- the field costs nothing in the file,")
            say("*** only in GPU memory.")
        for rb in lengths:
            lines: list = []
            if run(base(rb) + ["--dry-run"], keep=lines) != 0:
                continue
            chosen, hours = rb, dry_run_hours(lines)
            fitted = dry_run_batch(lines)
            if fitted and fitted < batch:
                say(f"*** batch {batch} does not fit this GPU; using {fitted}")
                batch = fitted
            if hours is None or hours <= TARGET_HOURS or rb == "256":
                break
            say("")
            say(f"*** {hours:.0f} hours at {rb}-byte records is too long for one")
            say("*** run. Timing shorter records (same data, same model file).")
        if chosen:
            break
    if not chosen:
        say("")
        say("no field size fits this GPU. Send mica_train.log to Claude.")
        return 1
    say("")
    say(f"record length for this run: {chosen} bytes")
    try:
        memo.parent.mkdir(parents=True, exist_ok=True)
        memo.write_text(chosen)
    except OSError:
        pass

    say("")
    say("=" * 66)
    say(" What to look for")
    say("=" * 66)
    say(" You can stop this any time: make an empty file called STOP_MICA")
    say("   in PycharmProjects. The next START_TRAIN.bat resumes from where")
    say("   it stopped - progress is saved every 50 steps.")
    say(" Starting a game pauses training by itself; quitting the game lets")
    say("   it carry on. You do not need to stop it first.")
    say(" With auto-restart on (INSTALL_AUTORESTART.bat) a crash, a hang or a")
    say("   reboot restarts training by itself from the last save; STOP_TRAIN.bat")
    say("   still stops it until START_TRAIN.bat. Its log: mica_supervisor.log")
    say("")
    say(" 'forward pass: HARD' must appear. SOFT trains to a better number")
    say("   and loses ~6 bits the moment it is written to disk.")
    say("")
    say(" discretisation_gap must stay near zero. It is the cost of turning")
    say(f"   the trained model into the {spec_bytes():,}-byte file. If it grows past")
    say("   a few tenths of a bit, stop and say so.")
    say("")
    u, b2, t3 = REFERENCE.get(chosen, REFERENCE["1024"])
    say(" val_bits is the real score. Reference points on the same 64")
    say(f" validation records at {chosen} bytes, tables fitted on other")
    say(" held-out documents:")
    say("     random guessing ............. 8.01")
    say(f"     letter frequencies (1-gram) . {u:.2f}   <- first real milestone")
    say(f"     previous-letter table ....... {b2:.2f}   (a 66 KB table)")
    say(f"     two-letter context (3-gram) . {t3:.2f}")
    say(f"     run 2, the previous machine . {RUN2}")
    say("     run 3, tape machine, readout fitted . 3.61 at step 1, then")
    say("        straight-through steps made it worse (3.69 by step 750)")
    say("     run 4, rule immediates fitted too ... 3.31")
    say("     run 5, vector rules, 4 ticks ........ 2.91")
    say("     experiments, 8 phases, 8 ticks ...... 2.77")
    say("     experiments, 16 phases, 16 ticks .... 2.70")
    say(" This run fits the readout AND the rule immediates exactly, then")
    say(" refits them round after round on fresh records. One line per round.")
    say("")
    say(" A progress line with the clock time and s/step is printed every")
    say("   10 steps. If none appears for 30 minutes and no validation is")
    say("   running, the run has stalled: r1/runs/soft/stall_trace.txt")
    say("   then shows exactly where (it stays empty on a healthy run).")
    say("")
    say(" r1/runs/soft/best.mica is the model, and it is a real integer")
    say("   MICA file the exact engine can run.")
    say("")
    say("=" * 66)
    say(" Training. Progress: mica/r1/runs/soft/progress.json")
    say(" Ctrl-C to stop; the best model is saved as you go.")
    say("=" * 66)
    run_plan(base, chosen)
    rc = run(base(chosen))
    say("")
    say(f"training process exited (code {rc}). Best model: "
        f"mica/r1/runs/soft/best.mica")

    return rc


if __name__ == "__main__":
    try:
        rc = main()
    except KeyboardInterrupt:
        say("\nstopped - best model kept")
        rc = 130
    finally:
        try:
            if LOCK.exists() and LOCK.read_text().split()[0] == str(os.getpid()):
                LOCK.unlink()
        except Exception:
            pass
    sys.exit(rc)
