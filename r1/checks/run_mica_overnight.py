#!/usr/bin/env python3
"""Resume the existing integer-rule MICA run and leave an overnight audit trail.

This supervisor changes no model code, geometry, corpus, or learned-rule method.
It copies the original checkpoint, trains in a new directory, and evaluates
immutable snapshots with the exact integer engine every two hours.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "r1"
STOP = ROOT.parent / "STOP_MICA"
PYTHON = Path(sys.executable)
PROMPTS = R1 / "checks/generation_dev10.json"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def record_event(out: Path, message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    with (out / "monitor.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        # Redirected stdout may inherit a Windows legacy code page.
        sys.stdout.buffer.write((line + "\n").encode("utf-8"))
        sys.stdout.flush()


def progress(out: Path) -> tuple[int, float | None]:
    path = out / "progress.json"
    if not path.exists():
        return 0, None
    try:
        hist = json.loads(path.read_text(encoding="utf-8"))["history"]
        return (hist[-1]["step"] if hist else 0,
                min(row["val_bits"] for row in hist) if hist else None)
    except (OSError, ValueError, KeyError):
        return 0, None


def render_report(out: Path, status: str, samples: list[dict]) -> None:
    step, selected = progress(out)
    rows = ["# MICA — דוח אימון לילה", "",
            f"עודכן: {datetime.now().astimezone().isoformat(timespec='minutes')}",
            f"מצב: {status}; סבב אחרון: {step}/145.", "",
            "הארכיטקטורה והנתונים זהים למודל המקורי: Minimal Inference Cellular Automaton, "
            "כללים שלמים נלמדים, חלון 32 בתים. לא שימש מודל חיצוני ליצירה.", "",
            "יעד: להתקרב ל־1 bit/target בבדיקת ולידציה נקייה, וגם ליצור משפטים חדשים והגיוניים. "
            "אין קידום לאתר עד ששני התנאים מתקיימים.", "",
            "נקודת פתיחה: 1.790994 bits/target ב־1,000 רשומות ולידציה נקיות; "
            "יצירה ישירה טרם הייתה מספקת.", "",
            f"הולידציה הפנימית הטובה ביותר: {selected:.4f} bits/target."
            if selected is not None else "אין עדיין מדידת ולידציה חדשה.", "",
            "| דגימה | סבב | bits/target נקי | hash מודל |", "|---|---:|---:|---|" ]
    for item in samples:
        bits = f"{item['bits']:.6f}" if item.get("bits") is not None else "שגיאת בדיקה"
        rows.append(f"| {item['label']} | {item['step']} | {bits} | `{item['sha256'][:12]}` |")
    if samples:
        last = samples[-1]
        rows += ["", "## דוגמאות יצירה ישירה מן המודל השלם", "",
                 "הדוגמאות הן פלט גולמי של MICA; עצירה מוקדמת וחזרתיות אינן מוסתרות.", ""]
        for row in last.get("generation", []):
            rows.append(f"- `{row['prompt']}` → `{row['full_text']}`")
        rows += ["", "הערכה אנושית של הגיון, חידוש ותקינות עדיין נדרשת. "
                 "מדד הדחיסה לבדו אינו מעיד על איכות המשפטים."]
    rows += ["", f"קובצי הריצה והדגימות: `{out}`", ""]
    write_atomic(out / "REPORT_HE.md", "\n".join(rows))
    write_atomic(out / "samples.json", json.dumps(samples, ensure_ascii=False, indent=2) + "\n")


def sample(out: Path, label: str, samples: list[dict]) -> None:
    step, _ = progress(out)
    source = out / "best.mica"
    if not source.exists():
        record_event(out, f"sample {label}: best.mica missing")
        return
    snap = out / "samples" / label
    snap.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, snap / "best.mica")
    shutil.copy2(out / "run_info.json", snap / "run_info.json")
    digest = file_hash(snap / "best.mica")
    item = {"label": label, "step": step, "sha256": digest,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "bits": None, "generation": []}
    if samples and samples[-1]["sha256"] == digest:
        item["bits"] = samples[-1].get("bits")
        item["generation"] = samples[-1].get("generation", [])
    else:
        eval_out = snap / "clean_val1000.json"
        cmd = [str(PYTHON), str(R1 / "checks/eval_int.py"),
               "--run", f"night={snap}", "--set", "val1000",
               "--out", str(eval_out), "--val-file",
               str(R1 / "data/eval_clean/val1000.jsonl"), "--val-skip", "0"]
        ev = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                            timeout=1800)
        (snap / "eval.log").write_text(ev.stdout + ev.stderr, encoding="utf-8")
        if ev.returncode == 0 and eval_out.exists():
            item["bits"] = json.loads(eval_out.read_text(encoding="utf-8"))["night"]["bits"]
        cmd = [str(PYTHON), str(R1 / "checks/eval_mica_raw_generation.py"),
               "--run", str(snap), "--prompts", str(PROMPTS),
               "--out", str(snap / "generation_dev10.json"),
               "--limit", "10", "--byte-limit", "100"]
        gen = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                             timeout=300)
        (snap / "generation.log").write_text(gen.stdout + gen.stderr, encoding="utf-8")
        if gen.returncode == 0:
            item["generation"] = json.loads((snap / "generation_dev10.json").read_text(
                encoding="utf-8"))["rows"]
    samples.append(item)
    render_report(out, "אימון פעיל", samples)
    record_event(out, f"sample {label}: round {step}, clean bpt {item['bits']}, model {digest[:12]}")


def train_command(source_info: dict, out: Path, steps: int) -> list[str]:
    a = source_info["args"]
    values = ["records", "val_records", "batch", "record_bytes", "ticks", "tau",
              "sel_init", "sel_tau", "lr", "int_lr", "init", "work_probes",
              "bias_init", "warmup", "clip", "checkpoint_every", "mode",
              "rule_max_back", "rule_scoring", "rule_self_terms", "rule_state",
              "fit_records", "fit_steps", "round_lr", "min_batch", "val_every",
              "val_records_n", "device", "gpu_mem_fraction", "min_free_ram_gb",
              "tbptt", "stall_minutes", "log_every", "yield_gb"]
    cmd = [str(PYTHON), str(R1 / "train_soft.py")]
    for key in values:
        cmd += ["--" + key.replace("_", "-"), str(a[key])]
    cmd += ["--steps", str(steps), "--save-every", "50", "--out", str(out)]
    if a["segments"]:
        cmd.append("--segments")
    if a["choose_channels"]:
        cmd.append("--choose-channels")
    if a["soft"]:
        cmd.append("--soft")
    if a["no_yield"]:
        cmd.append("--no-yield")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=145)
    ap.add_argument("--report-at", type=str, required=True,
                    help="local ISO time, e.g. 2026-09-27T05:45:00")
    a = ap.parse_args()
    source, out = a.source.resolve(), a.out.resolve()
    if STOP.exists():
        raise SystemExit(f"Training blocked by {STOP}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing nonempty output directory {out}")
    info = json.loads((source / "run_info.json").read_text(encoding="utf-8"))
    expected = ("resume.pt", "best.pt", "best.mica", "progress.json")
    out.mkdir(parents=True, exist_ok=True)
    for name in expected:
        shutil.copy2(source / name, out / name)
    shutil.copy2(source / "run_info.json", out / "run_info.json")
    # The old run saved resume.pt at round 15, while best.pt improved at
    # round 17 and progress.json reached round 19. Fit rounds do not use the
    # outer optimizer, so reconstruct the continuation from the exported
    # best state and all completed-round metadata instead of rolling back.
    import torch
    state = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
    history = json.loads((out / "progress.json").read_text(encoding="utf-8"))["history"]
    if history and history[-1]["step"] > state["step"]:
        state["model"] = torch.load(out / "best.pt", map_location="cpu",
                                    weights_only=True)
        state["step"] = history[-1]["step"]
        state["hist"] = history
        state["best"] = min(row["val_bits"] for row in history)
        repaired = out / "resume.repaired.tmp"
        torch.save(state, repaired)
        os.replace(repaired, out / "resume.pt")
    del state
    env = dict(os.environ)
    env.update({key: str(value) for key, value in info["env"].items()})
    env["PYTHONUTF8"] = "1"
    cmd = train_command(info, out, a.steps)
    write_atomic(out / "launch.json", json.dumps({
        "source": str(source), "source_model_sha256": file_hash(source / "best.mica"),
        "command": cmd, "env": info["env"], "report_at": a.report_at,
    }, indent=2) + "\n")
    samples: list[dict] = []
    render_report(out, "הוכן, ממתין לתחילת אימון", samples)
    record_event(out, f"starting original MICA training, target round {a.steps}")
    with (out / "train.log").open("a", encoding="utf-8") as log:
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                             stderr=subprocess.STDOUT, creationflags=flags)
        write_atomic(out / "train.pid", str(p.pid) + "\n")
        start = time.monotonic()
        due = [start + 2 * 3600, start + 4 * 3600]
        report_at = datetime.fromisoformat(a.report_at).astimezone().timestamp()
        final_done = False
        while True:
            now = time.monotonic()
            for index, when in enumerate(due):
                if when is not None and now >= when:
                    try:
                        sample(out, f"plus_{2 * (index + 1)}h", samples)
                    except Exception as exc:
                        record_event(out, f"sample failed: {type(exc).__name__}: {exc}")
                    due[index] = None
            if not final_done and time.time() >= report_at:
                try:
                    sample(out, "before_6am", samples)
                except Exception as exc:
                    record_event(out, f"final sample failed: {type(exc).__name__}: {exc}")
                final_done = True
            rc = p.poll()
            if rc is not None:
                try:
                    sample(out, "on_exit", samples)
                except Exception as exc:
                    record_event(out, f"exit sample failed: {type(exc).__name__}: {exc}")
                status = "האימון הסתיים" if (out / "FINISHED").exists() else f"האימון נעצר (exit {rc})"
                render_report(out, status, samples)
                record_event(out, status)
                return rc
            time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
