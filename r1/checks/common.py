"""Shared helpers for r1/checks/*: record sets, run geometry, metrics.

A model's geometry (MICA_* environment) must be set BEFORE mica_r1 is
imported, so drivers that handle several models run one worker process per
model, with the geometry recorded in that run's run_info.json.

Record sets (all trimmed to the training length, 256 bytes):
  val64     the first 64 records of the val split: exactly the records the
            trainer validates on (and picks best.mica with)
  val1000   1,000 records spread evenly over the REST of the val split
            (records 64 onward): for choosing between configurations
  test2000  2,000 records spread evenly over the test split. Documents are
            assigned whole to one split (data/ingest.py), so these share no
            document with training or validation. Used once, for the final
            report, never for tuning.

Metric: bits per target, pooled -- total nats over every predicted target
(each byte, then EOS) divided by the number of targets and by ln 2. The
trainer's own "val" line averages per-record means instead; both are kept.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
sys.path.insert(0, str(R1 / "data"))

RECORD_BYTES = 256
SPLIT_FILE = {"val": R1 / "data/bulk/val.jsonl", "test": R1 / "data/bulk/test.jsonl"}


def record_set(name: str, root: Path | None = None, *,
               val_file: Path | None = None, test_file: Path | None = None,
               val_skip: int = 64) -> list[bytes]:
    from make_records import load_records
    files = dict(SPLIT_FILE)
    if root is not None:
        files = {k: Path(root) / v.relative_to(R1) for k, v in SPLIT_FILE.items()}
    if val_file is not None:
        files["val"] = Path(val_file)
    if test_file is not None:
        files["test"] = Path(test_file)
    if val_skip < 0:
        raise ValueError("val_skip must be nonnegative")
    if name == "val64":
        recs = load_records(files["val"])[:64]
    elif name.startswith("val"):
        n = int(name[3:])
        pool = load_records(files["val"])[val_skip:]
        recs = pool[::max(1, len(pool) // n)][:n]
    elif name.startswith("test"):
        n = int(name[4:])
        pool = load_records(files["test"])
        recs = pool[::max(1, len(pool) // n)][:n]
    else:
        raise ValueError(name)
    recs = [r[:RECORD_BYTES] for r in recs]
    assert all(len(r) for r in recs), "empty record in set"
    return recs


def run_geometry(run_dir) -> dict:
    info = json.loads((Path(run_dir) / "run_info.json").read_text())
    return dict(info["env"])


def worker_env(geom: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MICA_")}
    env.update(geom)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def pooled_bits(nats, counts) -> float:
    return float(sum(nats)) / float(sum(counts)) / math.log(2)


def mean_record_bits(nats, counts) -> float:
    return sum(n / c for n, c in zip(nats, counts)) / len(nats) / math.log(2)


def paired_bootstrap(nats_a, nats_b, counts, n_boot: int = 2000, seed: int = 0):
    """Pooled bits(A) - bits(B) on the same records, with a 95% bootstrap
    interval over records (records resampled with replacement)."""
    import numpy as np
    a, b, c = (np.asarray(x, float) for x in (nats_a, nats_b, counts))
    d = (a.sum() - b.sum()) / c.sum() / math.log(2)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), (n_boot, len(c)))
    ds = (a[idx].sum(1) - b[idx].sum(1)) / c[idx].sum(1) / math.log(2)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return float(d), float(lo), float(hi)


def beat(msg: str = "check") -> None:
    """Touch the main run's heartbeat: the auto-restart watcher kills a run
    whose heartbeat has been silent for 45 minutes, and a long check runs
    inside the training process tree."""
    import time
    hb = R1 / "runs/soft/heartbeat"
    try:
        if hb.parent.exists():
            hb.write_text(f"{msg} {int(time.time())}\n")
    except OSError:
        pass


def peak_rss_bytes() -> int | None:
    """Peak resident memory of THIS process (Linux/macOS: getrusage; Windows:
    PeakWorkingSetSize from GetProcessMemoryInfo)."""
    import sys as _s
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r) if _s.platform == "darwin" else int(r) * 1024
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return int(pmc.PeakWorkingSetSize)
    except Exception:
        pass
    return None


def cpu_name() -> str:
    import platform
    import sys as _s
    name = platform.processor() or ""
    if _s.platform.startswith("linux"):
        try:
            for ln in open("/proc/cpuinfo"):
                if ln.startswith("model name"):
                    name = ln.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    elif _s.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return f"{name} ({os.cpu_count()} logical CPUs, {platform.system()} {platform.release()})"


def utf8_stdout() -> None:
    """Sample text can hold any character; Windows pipes default to cp1252."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def hard_exit(code: int = 0) -> None:
    """End this process now, skipping interpreter and DLL teardown. On the PC
    (Windows, ROCm) a worker that has used the GPU was seen twice to hang
    after its last line, even through os._exit; TerminateProcess skips the
    DLL detach notifications where such a hang would sit."""
    import sys as _s
    for f in (_s.stdout, _s.stderr):
        try:
            f.flush()
        except Exception:
            pass
    if _s.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.TerminateProcess(
                ctypes.windll.kernel32.GetCurrentProcess(), code)
        except Exception:
            pass
    os._exit(code)
