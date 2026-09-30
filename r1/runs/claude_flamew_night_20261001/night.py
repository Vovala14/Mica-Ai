#!/usr/bin/env python3
"""Overnight Flame-W job: no-TinyStories continuation of full40, then evaluation.

    python r1\\runs\\claude_flamew_night_20261001\\night.py --hours 8

Run it from the mica folder with the ROCm venv's python. It does, in order:

  1. corpus  Rebuild the word corpus from r1/data/mix/v02a WITHOUT the
             TinyStories records (source label 4), into
             r1/data/word/v02a_nostories/. A self-check first rebuilds the
             first 5,000 records WITH stories and compares them to
             r1/data/word/v02a/train.jsonl byte for byte, so the tokenizer and
             vocabulary are proven identical.
  2. train   Two arms, one after the other, each continuing full40's saved
             state (resume.pt) on the new corpus:
               A  --round-lr 0.03     B  --round-lr 0.01
             The "best so far" is reset, so a new best.mica is written as soon
             as the new validation set improves (the day chain never saved one
             because its old best was measured on different records).
  3. eval    CPU only, after the GPU work: full40, the 2026-09-30 lr 0.03 run
             (if present), A and B, each with
               * next-word top-1/3 on chat dev1000 and everyday dev_fresh1000
                 (the first 1,000 fixed positions, same for every model),
               * bits/word on 500 no-TinyStories validation records,
               * sentences for 50 prompts with the current decoder
                 (w-sent-mmi.3) and the new one (w-sent-bos.5f).
  4. report  night_report.zip in this folder: logs, progress, scores and
             sentences, but no model files. Send that zip to Claude.

Limits (defaults, all adjustable):
  --gpu-mem-fraction 0.75  VRAM hard cap: 0.75 x 15.9 GiB = 11.9 GiB
  --ram-cap-gb 12          the training process is stopped cleanly if its
                           own memory passes this
  --min-free-ram-gb 4      ... or if the PC's free RAM drops below this
  below-normal CPU priority, at most 8 CPU threads, pause while a game uses
  the GPU; an empty file named STOP_MICA in PycharmProjects stops it cleanly
  at any time (state is saved; nothing is lost).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
R1 = HERE.parents[1]
ROOT = R1.parent                                  # the mica folder
STOP = ROOT.parent / "STOP_MICA"                  # same file train_soft.py watches
FULL40 = R1 / "runs/codex_flame_word_full40_20260928/train"
LR030 = R1 / "runs/claude_flame_word_20260930/a_full40_lr030/train"
MIX = R1 / "data/mix/v02a"
WORD = R1 / "data/word"
NOSTORY = WORD / "v02a_nostories"
TINYSTORIES = 4                                   # build_chat_corpus.SOURCES index
ARMS = {"A_lr030": 0.03, "B_lr010": 0.01}
OVERRIDE = {"records", "val_records", "steps", "round_lr", "out", "fresh", "dry_run",
            "val_records_n", "gpu_mem_fraction", "min_free_ram_gb"}
LOG = HERE / "night.log"


def say(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ memory
def process_ram_gb(pid: int) -> float:
    """Private memory of one process in GB (psutil if present, else Win32)."""
    try:
        import psutil
        return psutil.Process(pid).memory_info().rss / 1e9
    except ImportError:
        pass
    except Exception:
        return 0.0
    if os.name != "nt":
        try:
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1e6
        except OSError:
            return 0.0
        return 0.0
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = wintypes.HANDLE
    h = k32.OpenProcess(0x1000 | 0x0010, False, pid)
    if not h:
        return 0.0
    try:
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        return max(pmc.WorkingSetSize, pmc.PagefileUsage) / 1e9 if ok else 0.0
    finally:
        k32.CloseHandle(h)


# ------------------------------------------------------------------ 1 corpus
def build_corpus() -> dict:
    sys.path.insert(0, str(R1 / "data"))
    import build_word_corpus as W
    vocab = W.Vocab.load(WORD / "vocab.json")

    def records(split):
        src = (MIX / f"{split}.src").read_bytes()
        with open(MIX / f"{split}.jsonl", encoding="ascii") as fh:
            lines = [l for l in fh if l.strip()]
        if len(lines) != len(src):
            raise SystemExit(f"[corpus] {split}: {len(lines)} records but {len(src)} source labels")
        for line, code in zip(lines, src):
            toks = W.tokenize(W.record_text(bytes.fromhex(line.strip())))
            if toks:                       # build_word_corpus.convert skips empty records too
                yield code, W.pack(vocab.encode(toks))

    ref = WORD / "v02a/train.jsonl"
    with open(ref, encoding="ascii") as fh:
        want = [next(fh).strip() for _ in range(5000)]
    got = []
    for _, packed in records("train"):
        got.append(packed)
        if len(got) == 5000:
            break
    if got != want:
        bad = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
        raise SystemExit(f"[corpus] self-check FAILED at record {bad}: the rebuild does not match "
                         f"{ref}. Nothing was trained.")
    say("[corpus] self-check passed: 5,000 rebuilt records match the existing word corpus")

    NOSTORY.mkdir(parents=True, exist_ok=True)
    info = {}
    for split in ("train", "val"):
        kept = dropped = 0
        tmp = NOSTORY / f"{split}.jsonl.tmp"
        with open(tmp, "w", encoding="ascii", newline="\n") as out:
            for code, packed in records(split):
                if code == TINYSTORIES:
                    dropped += 1
                    continue
                out.write(packed + "\n")
                kept += 1
        os.replace(tmp, NOSTORY / f"{split}.jsonl")
        info[split] = {"records": kept, "tinystories_dropped": dropped,
                       "sha256": sha256(NOSTORY / f"{split}.jsonl")}
        say(f"[corpus] {split}: kept {kept:,} records, dropped {dropped:,} TinyStories records")
    info["source"] = {f: sha256(MIX / f) for f in ("train.jsonl", "train.src", "val.jsonl", "val.src")}
    (NOSTORY / "manifest.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


# ------------------------------------------------------------------ 2 train
def prepare_arm(out: Path) -> None:
    """Copy full40's state into a new folder with the 'best so far' reset."""
    import torch
    out.mkdir(parents=True, exist_ok=True)
    if (out / "resume.pt").exists():
        say(f"[train] {out.parent.name}: resume.pt exists, continuing it")
        return
    ck = torch.load(FULL40 / "resume.pt", map_location="cpu", weights_only=False)
    say(f"[train] {out.parent.name}: starting from full40 round {ck['step']} "
        f"(its best {ck['best']:.4f} was on different validation records; reset)")
    ck["best"], ck["hist"] = float("inf"), []
    torch.save(ck, out / "resume.pt")
    shutil.copy2(FULL40 / "run_info.json", out / "run_info.json")


def train_command(info: dict, out: Path, round_lr: float, steps: int, a) -> list[str]:
    cmd = [sys.executable, str(R1 / "train_soft.py")]
    for key, val in info["args"].items():
        if key in OVERRIDE or val is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd += [flag, str(val)]
    cmd += ["--records", str(NOSTORY / "train.jsonl"), "--val-records", str(NOSTORY / "val.jsonl"),
            "--val-records-n", "128", "--round-lr", str(round_lr), "--steps", str(steps),
            "--gpu-mem-fraction", str(a.gpu_mem_fraction), "--min-free-ram-gb", str(a.min_free_ram_gb),
            "--out", str(out)]
    return cmd


def run_arm(name: str, round_lr: float, deadline: float, a) -> Path | None:
    out = HERE / name / "train"
    prepare_arm(out)
    info = json.loads((FULL40 / "run_info.json").read_text(encoding="utf-8"))
    env = dict(os.environ, **{k: str(v) for k, v in info["env"].items()}, PYTHONUNBUFFERED="1")
    cmd = train_command(info, out, round_lr, 40 + a.max_rounds, a)
    (HERE / name).mkdir(exist_ok=True)
    (HERE / name / "command.txt").write_text(" ".join(cmd) + "\n")
    say(f"[train] {name}: round-lr {round_lr}, until {dt.datetime.fromtimestamp(deadline):%H:%M} "
        f"or {a.max_rounds} rounds")
    created_stop, reason = False, "finished"
    with open(HERE / name / "train.log", "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
        peak = 0.0
        while proc.poll() is None:
            time.sleep(15)
            ram = process_ram_gb(proc.pid)
            peak = max(peak, ram)
            why = None
            if ram > a.ram_cap_gb:
                why = f"training process uses {ram:.1f} GB RAM (cap {a.ram_cap_gb} GB)"
            elif time.time() > deadline:
                why = "time budget for this arm used up"
            if why and not created_stop and not STOP.exists():
                say(f"[train] {name}: stopping cleanly -- {why}")
                STOP.write_text("created by night.py\n")
                created_stop, reason = True, why
            if created_stop and time.time() > deadline + 900:
                say(f"[train] {name}: did not stop 15 min after STOP_MICA; terminating")
                proc.terminate()
        if created_stop and STOP.exists():
            STOP.unlink()
    say(f"[train] {name}: exit code {proc.returncode}, {reason}, peak RAM {peak:.1f} GB")
    best = out / "best.mica"
    return best if best.exists() else None


# ------------------------------------------------------------------ 3 eval
def evaluate(models: dict[str, Path], deadline: float) -> dict:
    sys.path.insert(0, str(HERE))
    import decode_bos as B
    E, D = B.E, B.D
    orig_positions = E.letter_positions
    E.letter_positions = lambda path, n=1000, seed=0: orig_positions(path, n=n, seed=seed)
    info = json.loads((FULL40 / "run_info.json").read_text(encoding="utf-8"))
    vocab = E.W.Vocab.load(WORD / "vocab.json")
    prompts = (json.loads((R1 / "checks/sentence_holdout_20260926.json").read_text())["prompts"]
               + json.loads((HERE / "dev30_prompts.json").read_text()))
    sets = {"chat_dev1000": R1 / "data/chat/dev1000.jsonl",
            "everyday_dev_fresh1000": R1 / "runs/codex_flame_scaling_20260928/dev_fresh1000.jsonl"}
    val500 = HERE / "eval" / "val500.jsonl"
    val500.parent.mkdir(exist_ok=True)
    with open(NOSTORY / "val.jsonl", encoding="ascii") as fh:
        val500.write_text("".join(l for _, l in zip(range(500), fh)))
    results = {}
    for name, path in models.items():
        if time.time() > deadline:
            say(f"[eval] out of time before {name}")
            break
        pkg = HERE / "eval" / "pkg" / name
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "package.json").write_text(json.dumps(
            {"code_dir": str(R1), "model": str(path), "model_sha256": sha256(path),
             "geometry_env": info["env"]}, indent=1))
        model = E.load_model(f"mica:{pkg}")
        res = {"model": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        t0 = time.time()
        for sname, spath in sets.items():
            if spath.exists():
                r = E.nextword(model, vocab, str(spath))
                res[f"nextword_{sname}"] = {k: r[k] for k in ("positions", "top1", "top3", "top10")}
        res["bits_val500"] = round(E.bits(model, str(val500))["bits_per_token"], 4)
        gens = {}
        for dname, dec in (("w-sent-mmi.3", D.WordDecoder(model, vocab, **D.MODES["sentence"])),
                           ("w-sent-bos.5f", B.BosDecoder(model, vocab, **B.BOS5F))):
            gens[dname] = [{"prompt": p, "continuation": dec.complete(p)["continuation"]} for p in prompts]
        res["sentences"] = gens
        res["eval_seconds"] = round(time.time() - t0)
        results[name] = res
        (HERE / "eval" / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        nw = res.get("nextword_chat_dev1000", {}).get("top1")
        say(f"[eval] {name}: bits/word {res['bits_val500']}, chat next-word top-1 {nw}, "
            f"{res['eval_seconds']} s")
    return results


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=8.0, help="total time for everything")
    ap.add_argument("--eval-hours", type=float, default=1.5, help="reserved at the end for evaluation")
    ap.add_argument("--max-rounds", type=int, default=400, help="rule rounds per arm at most")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.75)
    ap.add_argument("--ram-cap-gb", type=float, default=12.0)
    ap.add_argument("--min-free-ram-gb", type=float, default=4.0)
    ap.add_argument("--skip-train", action="store_true", help="only (re)run the evaluation")
    a = ap.parse_args()
    start = time.time()
    end = start + a.hours * 3600
    say(f"[night] start; everything ends by {dt.datetime.fromtimestamp(end):%H:%M}")
    try:
        if STOP.exists():
            raise SystemExit(f"{STOP} exists; delete it first")
        for need in (FULL40 / "resume.pt", FULL40 / "run_info.json", FULL40 / "best.mica",
                     MIX / "train.jsonl", MIX / "train.src", MIX / "val.jsonl", MIX / "val.src",
                     WORD / "vocab.json", WORD / "v02a/train.jsonl"):
            if not need.exists():
                raise SystemExit(f"missing {need}")
        models = {"full40": FULL40 / "best.mica"}
        if LR030.joinpath("best.mica").exists():
            models["lr030_20260930"] = LR030 / "best.mica"
        if not a.skip_train:
            if not (NOSTORY / "manifest.json").exists():
                build_corpus()
            train_end = end - a.eval_hours * 3600
            for i, (name, lr) in enumerate(ARMS.items()):
                left = train_end - time.time()
                if left < 1800:
                    say(f"[train] skipping {name}: under 30 min left")
                    continue
                best = run_arm(name, lr, time.time() + left / (len(ARMS) - i), a)
                if best:
                    models[name] = best
        else:
            for name in ARMS:
                if (HERE / name / "train/best.mica").exists():
                    models[name] = HERE / name / "train/best.mica"
        evaluate(models, end)
    except SystemExit as exc:
        say(f"[night] stopped: {exc}")
    except Exception:
        say("[night] ERROR\n" + traceback.format_exc())
    finally:
        report = HERE / "night_report.zip"
        with zipfile.ZipFile(report, "w", zipfile.ZIP_DEFLATED) as z:
            for p in HERE.rglob("*"):
                if p.is_file() and p.suffix in (".json", ".log", ".txt") and "pkg" not in p.parts \
                        and p.name != "val500.jsonl":
                    z.write(p, p.relative_to(HERE))
        say(f"[night] done in {(time.time() - start) / 3600:.1f} h; send {report} to Claude")
    return 0


if __name__ == "__main__":
    sys.exit(main())
