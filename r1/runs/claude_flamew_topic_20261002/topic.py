#!/usr/bin/env python3
"""Flame-W topic register experiment: one command, then send the zip.

    & ..\\.venv-rocm\\Scripts\\python.exe r1\\runs\\claude_flamew_topic_20261002\\topic.py --hours 5

What it does, in order (everything ends by --hours):

  1. codes   topic_codes.py on the no-TinyStories training records (CPU, a few
             minutes, once): an 8-number "what is this about" code for each
             content word; function words get zeros.
  2. train   two matched arms, the same start (B: the night launcher's
             B_lr010 best.mica, or --start), the same data, the same rounds:
               T_topic  the topic register on (8 extra channels): in phases
                        8-15, two of each rule's six scoring terms read the
                        register, so which rule fires depends on the recent
                        content words, not only on the last few words
               T_zero   the same change with all-zero codes: the register is
                        always 0, so those two terms read nothing. This is the
                        control: T_topic minus T_zero is the topic's effect,
                        separate from the cost of changing the rules.
  3. eval    CPU, each model in its own process (the geometry is fixed per
             process): next-word top-1/3/10 on chat dev1000 and everyday
             dev_fresh1000, bits/word on 500 no-TinyStories validation
             records, and sentences for the same 50 prompts with both decoders.
  4. report  topic_report.zip in this folder (logs, scores, sentences; no model
             files). Send it to Claude.

Same guards as the night launcher: GPU memory cap (--gpu-mem-fraction),
training process RAM cap (--ram-cap-gb), stop if free RAM falls below
--min-free-ram-gb, pause while a game uses the GPU, and an empty file named
STOP_MICA in PycharmProjects stops cleanly at any time. Running the same
command again continues both arms from where they stopped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
R1 = HERE.parents[1]
ROOT = R1.parent
NIGHT_DIR = R1 / "runs/claude_flamew_night_20261001"
sys.path.insert(0, str(NIGHT_DIR))
import night as N  # noqa: E402  (shares the training command, guards, data paths)

LOG = HERE / "topic.log"
DATA = N.corpus_dir("nostories")
TOPICAL = R1 / "data/word/v03_topical"          # build_topical_corpus.py
START = NIGHT_DIR / "B_lr010/train/best.mica"
# codes: the .npz file; data: the training corpus; from: the records the codes
# are learned from, and topic_codes.py options for them
ARMS = {
    "T_topic": dict(codes=HERE / "topic_codes.npz", data=DATA,
                    source=DATA / "train.jsonl", opts=[]),
    "T_zero": dict(codes=HERE / "zero_codes.npz", data=DATA, source=None, opts=[]),
    # v03: multi-turn windows that stay on one subject; codes learned from
    # those windows only, with a wider window and more skipped function words
    "T2_topic": dict(codes=HERE / "topic_codes_v03.npz", data=TOPICAL,
                     source=TOPICAL / "topical.jsonl",
                     opts=["--window", "16", "--skip", "250", "--content", "8000",
                           "--contexts", "3000"]),
}
CODES = {k: v["codes"] for k, v in ARMS.items()}
TOPIC = 8
SHIFT = 3
PHASES = "8,9,10,11,12,13,14,15"
TERMS = 2


def say(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def base_info() -> dict:
    return json.loads((N.FULL40 / "run_info.json").read_text(encoding="utf-8"))


def topic_env(info: dict) -> dict:
    env = {k: str(v) for k, v in info["env"].items()}
    env["MICA_CHANNELS"] = str(int(env["MICA_CHANNELS"]) + TOPIC)
    env["MICA_TOPIC"], env["MICA_TOPIC_SHIFT"] = str(TOPIC), str(SHIFT)
    return env


# ------------------------------------------------------------------ 1 codes
def make_data(arms) -> None:
    """The v03 topical corpus, downloaded and built once, if an arm needs it."""
    if not any(ARMS[n]["data"] == TOPICAL for n in arms) or (TOPICAL / "manifest.json").exists():
        return
    if not (DATA / "manifest.json").exists():
        N.build_corpus("nostories")
    for step in ("fetch", "build"):
        say(f"[data] build_topical_corpus.py {step} (downloads about 300 MB once)")
        r = subprocess.run([sys.executable, str(R1 / "data/build_topical_corpus.py"), step],
                           cwd=ROOT, capture_output=True, text=True)
        for line in (r.stdout + r.stderr).splitlines()[-25:]:
            say(line)
        if r.returncode:
            raise SystemExit(f"build_topical_corpus.py {step} failed")


def make_codes(arms) -> None:
    import numpy as np
    for name in arms:
        a = ARMS[name]
        if a["codes"].exists() or a["source"] is None:
            continue
        say(f"[codes] {name}: building topic codes from {a['source'].name} (CPU, a few minutes)")
        r = subprocess.run([sys.executable, str(HERE / "topic_codes.py"), str(a["source"]),
                            str(N.WORD / "vocab.json"), str(a["codes"]),
                            "--channels", str(TOPIC), "--shift", str(SHIFT)] + a["opts"],
                           cwd=ROOT, capture_output=True, text=True)
        for line in (r.stdout + r.stderr).splitlines():
            say(line)
        if r.returncode:
            raise SystemExit("topic_codes.py failed")
    if "T_zero" in arms and not CODES["T_zero"].exists():
        if not CODES["T_topic"].exists():
            make_codes(["T_topic"])
        codes = np.load(CODES["T_topic"])["codes"]
        np.savez(CODES["T_zero"], codes=np.zeros_like(codes))


# ------------------------------------------------------------------ 2 train
def run_arm(name: str, deadline: float, a) -> Path | None:
    info = base_info()
    out = HERE / name / "train"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **topic_env(info), PYTHONUNBUFFERED="1")
    done = 0
    if (out / "progress.json").exists():
        hist = json.loads((out / "progress.json").read_text()).get("history", [])
        done = hist[-1]["step"] if hist else 0
    cmd = N.train_command(info, out, a.round_lr, done + a.max_rounds, a, ARMS[name]["data"])
    cmd += ["--init-mica", str(a.start), "--topic-codes", str(CODES[name]),
            "--topic-phases", PHASES, "--topic-terms", str(TERMS)]
    (HERE / name / "command.txt").write_text(" ".join(cmd) + "\n")
    say(f"[train] {name}: from {a.start.name} ({N.sha256(a.start)[:12]}), round-lr {a.round_lr}, "
        f"until {dt.datetime.fromtimestamp(deadline):%H:%M} or {a.max_rounds} more rounds")
    created_stop, reason, peak = False, "finished", 0.0
    with open(HERE / name / "train.log", "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(15)
            ram = N.process_ram_gb(proc.pid)
            peak = max(peak, ram)
            why = None
            if ram > a.ram_cap_gb:
                why = f"training process uses {ram:.1f} GB RAM (cap {a.ram_cap_gb} GB)"
            elif time.time() > deadline:
                why = "time budget for this arm used up"
            if why and not created_stop and not N.STOP.exists():
                say(f"[train] {name}: stopping cleanly -- {why}")
                N.STOP.write_text("created by topic.py\n")
                created_stop, reason = True, why
            if created_stop and time.time() > deadline + 900:
                say(f"[train] {name}: did not stop 15 min after STOP_MICA; terminating")
                proc.terminate()
        if created_stop and N.STOP.exists():
            N.STOP.unlink()
    say(f"[train] {name}: exit code {proc.returncode}, {reason}, peak RAM {peak:.1f} GB")
    if proc.returncode:
        tail = (HERE / name / "train.log").read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        say("[train] last lines of the log:\n" + "\n".join(tail))
    best = out / "best.mica"
    return best if best.exists() else None


# ------------------------------------------------------------------ 3 eval
def eval_one(name: str, model: Path, env: dict) -> None:
    """Runs in its own process: the geometry is fixed when mica_r1 is imported."""
    os.environ.update(env)
    sys.path.insert(0, str(NIGHT_DIR))
    import decode_bos as B
    E, D = B.E, B.D
    orig_positions = E.letter_positions
    E.letter_positions = lambda path, n=1000, seed=0: orig_positions(path, n=n, seed=seed)
    vocab = E.W.Vocab.load(N.WORD / "vocab.json")
    prompts = (json.loads((R1 / "checks/sentence_holdout_20260926.json").read_text())["prompts"]
               + json.loads((NIGHT_DIR / "dev30_prompts.json").read_text())
               + json.loads((HERE / "topic24_prompts.json").read_text()))
    sets = {"chat_dev1000": R1 / "data/chat/dev1000.jsonl",
            "everyday_dev_fresh1000": R1 / "runs/codex_flame_scaling_20260928/dev_fresh1000.jsonl"}
    val500 = HERE / "eval" / "val500.jsonl"
    pkg = HERE / "eval" / "pkg" / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "package.json").write_text(json.dumps(
        {"code_dir": str(R1), "model": str(model), "model_sha256": N.sha256(model),
         "geometry_env": env}, indent=1))
    m = E.load_model(f"mica:{pkg}")
    res = {"model": str(model.relative_to(ROOT)), "sha256": N.sha256(model), "env": env}
    t0 = time.time()
    for sname, spath in sets.items():
        if spath.exists():
            r = E.nextword(m, vocab, str(spath))
            res[f"nextword_{sname}"] = {k: r[k] for k in ("positions", "top1", "top3", "top10")}
    res["bits_val500"] = round(E.bits(m, str(val500))["bits_per_token"], 4)
    gens = {}
    for dname, dec in (("w-sent-mmi.3", D.WordDecoder(m, vocab, **D.MODES["sentence"])),
                       ("w-sent-bos.5f", B.BosDecoder(m, vocab, **B.BOS5F))):
        gens[dname] = [{"prompt": p, "continuation": dec.complete(p)["continuation"]} for p in prompts]
    res["sentences"] = gens
    res["eval_seconds"] = round(time.time() - t0)
    (HERE / "eval" / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def evaluate(models: dict[str, tuple[Path, dict]], deadline: float) -> None:
    val500 = HERE / "eval" / "val500.jsonl"
    val500.parent.mkdir(exist_ok=True)
    with open(DATA / "val.jsonl", encoding="ascii") as fh:
        val500.write_text("".join(l for _, l in zip(range(500), fh)))
    for name, (path, env) in models.items():
        if time.time() > deadline:
            say(f"[eval] out of time before {name}")
            break
        r = subprocess.run([sys.executable, str(HERE / "topic.py"), "--eval-one", name, str(path),
                            json.dumps(env)], cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            say(f"[eval] {name} FAILED:\n{(r.stdout + r.stderr)[-3000:]}")
            continue
        res = json.loads((HERE / "eval" / f"{name}.json").read_text(encoding="utf-8"))
        say(f"[eval] {name}: bits/word {res['bits_val500']}, chat next-word top-1 "
            f"{res.get('nextword_chat_dev1000', {}).get('top1')}, {res['eval_seconds']} s")


# ------------------------------------------------------------------ main
def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "--eval-one":
        eval_one(sys.argv[2], Path(sys.argv[3]), json.loads(sys.argv[4]))
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=5.0, help="total time for everything")
    ap.add_argument("--eval-hours", type=float, default=0.8, help="reserved at the end for evaluation")
    ap.add_argument("--max-rounds", type=int, default=400, help="rule rounds per arm at most")
    ap.add_argument("--round-lr", type=float, default=0.01)
    ap.add_argument("--start", type=Path, default=START, help="the model both arms start from")
    ap.add_argument("--arms", default="T2_topic")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.75)
    ap.add_argument("--ram-cap-gb", type=float, default=12.0)
    ap.add_argument("--min-free-ram-gb", type=float, default=4.0)
    ap.add_argument("--skip-train", action="store_true", help="only (re)run the evaluation")
    a = ap.parse_args()
    a.start = a.start.resolve()
    start = time.time()
    end = start + a.hours * 3600
    say(f"[topic] start; everything ends by {dt.datetime.fromtimestamp(end):%H:%M}")
    try:
        if N.STOP.exists():
            raise SystemExit(f"{N.STOP} exists; delete it first")
        for need in (a.start, N.FULL40 / "run_info.json", N.WORD / "vocab.json",
                     DATA / "train.jsonl", DATA / "val.jsonl"):
            if not need.exists():
                raise SystemExit(f"missing {need}")
        info = base_info()
        base = {k: str(v) for k, v in info["env"].items()}
        models = {"start": (a.start, base)}
        arms = [n for n in a.arms.split(",") if n]
        for n in arms:
            if n not in ARMS:
                raise SystemExit(f"unknown arm {n}; arms are {', '.join(CODES)}")
        if not a.skip_train:
            make_data(arms)
            make_codes(arms)
            train_end = end - a.eval_hours * 3600
            for i, name in enumerate(arms):
                left = train_end - time.time()
                if left < 1800:
                    say(f"[train] skipping {name}: under 30 min left")
                    continue
                run_arm(name, time.time() + left / (len(arms) - i), a)
        for name in ARMS:                      # every arm with a model, for comparison
            best = HERE / name / "train/best.mica"
            if best.exists():
                models[name] = (best, topic_env(info))
        evaluate(models, end)
    except SystemExit as exc:
        say(f"[topic] stopped: {exc}")
    except Exception:
        say("[topic] ERROR\n" + traceback.format_exc())
    finally:
        report = HERE / "topic_report.zip"
        with zipfile.ZipFile(report, "w", zipfile.ZIP_DEFLATED) as z:
            for p in HERE.rglob("*"):
                if p.is_file() and p.suffix in (".json", ".log", ".txt") and "pkg" not in p.parts \
                        and p.name != "val500.jsonl":
                    z.write(p, p.relative_to(HERE))
        say(f"[topic] done in {(time.time() - start) / 3600:.1f} h; send {report} to Claude")
    return 0


if __name__ == "__main__":
    sys.exit(main())
