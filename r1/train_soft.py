#!/usr/bin/env python3
"""Gradient training for MICA, and the integer model that comes out of it.

Direct mutation search (R1 section 13) cannot reach megabyte scale: one
accepted change was measured at ~44 s, and a 4.25 MB model has 3.4 million
tunable slots. This trains the same machine by backpropagation instead.

THE ONE THING TO UNDERSTAND BEFORE RUNNING THIS

The forward pass is HARD by default: argmax selectors, rounded parameters,
int8 field. Gradients flow through a softmax version in the backward pass
only. That is deliberate and it is the difference between a model and a
mirage. Measured on the same four-record overfit:

    soft forward   trained to 3.01 bits -> 9.52 bits as an integer model
    hard forward   trained to 6.10 bits -> 6.12 bits as an integer model

Soft training reaches a much better number and then loses six and a half bits
when you save the file, because a soft read of a distribution over 768 cells
is a weighted average the integer machine cannot compute. --soft is available
for comparison; do not ship anything trained with it.
"""

from __future__ import annotations

import argparse, gc, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from make_records import load_records
from mica_r1 import spec, serialize
from mica_r1.soft import SoftMica
from mica_r1.discretise import measure_gap, to_integer


# ===========================================================================
# Keeping the desktop alive
#
# The earlier runs froze the machine. The error said "29.47 GiB is allocated
# by PyTorch" on a 15.92 GiB card: on Windows the GPU can oversubscribe into
# system RAM, so PyTorch quietly took ~13 GB of the machine's 32 GB, Windows
# started paging, and everything stalled for minutes per attempt. Four guards:
#
#   1. a hard VRAM cap, so an allocation that does not fit fails INSTANTLY
#      and cleanly instead of spilling into system memory
#   2. below-normal priority and a CPU-thread limit, so the desktop always
#      wins the CPU
#   3. a system-RAM watchdog that stops the run before the machine pages
#   4. a STOP_MICA file in PycharmProjects that ends the run cleanly, so it
#      can be stopped remotely without killing anything
# ===========================================================================
STOP_FILE = Path(__file__).resolve().parents[2] / "STOP_MICA"


def free_ram_gb() -> float:
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _MS()
            m.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 1e9
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return float("inf")


def harden(dev, frac: float) -> None:
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.GetCurrentProcess.restype = ctypes.c_void_p
            k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            ok = k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)
            print(f"[guard] process priority: "
                  f"{'below normal' if ok else 'unchanged (call failed)'}",
                  flush=True)
        except Exception as exc:
            print(f"[guard] could not lower priority: {exc}", flush=True)
        # A GPU job does not count as "activity" to Windows: the PC can go
        # to sleep under a running training job, and a windowless background
        # process can be put into efficiency mode. Both are switched off for
        # THIS process only, until it exits. The screen can still turn off;
        # priority stays below normal.
        try:
            k.SetThreadExecutionState.argtypes = [ctypes.c_ulong]
            k.SetThreadExecutionState.restype = ctypes.c_ulong
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            ok = k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            print("[guard] the PC will not go to sleep while training runs "
                  "(the screen can still turn off)" if ok else
                  "[guard] *** could not stop the PC from sleeping; a sleep "
                  "pauses training", flush=True)
        except Exception as exc:
            print(f"[guard] could not stop sleep: {exc}", flush=True)
        try:
            class _Throttle(ctypes.Structure):
                _fields_ = [("Version", ctypes.c_ulong),
                            ("ControlMask", ctypes.c_ulong),
                            ("StateMask", ctypes.c_ulong)]
            k.SetProcessInformation.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
            st = _Throttle(1, 0x1, 0x0)   # EXECUTION_SPEED: not throttled
            ok = k.SetProcessInformation(k.GetCurrentProcess(), 4,
                                         ctypes.byref(st), ctypes.sizeof(st))
            print(f"[guard] Windows efficiency mode: "
                  f"{'off for this process' if ok else 'could not turn off'}",
                  flush=True)
        except Exception as exc:
            print(f"[guard] could not turn off efficiency mode: {exc}",
                  flush=True)
    n = max(1, min(8, (os.cpu_count() or 4) // 3))
    torch.set_num_threads(n)
    print(f"[guard] torch CPU threads: {n} of {os.cpu_count()}", flush=True)
    if dev.type == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(frac)
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"[guard] VRAM capped at {frac:.0%} of {total:.1f} GiB = "
                  f"{frac * total:.1f} GiB. Anything larger fails at once "
                  f"instead of spilling into system RAM.", flush=True)
        except Exception as exc:
            print(f"[guard] *** COULD NOT CAP VRAM ({exc}). The run may spill "
                  f"into system memory. Refusing to continue.", flush=True)
            raise SystemExit(2)
    print(f"[guard] free system RAM: {free_ram_gb():.1f} GB", flush=True)


# Step sizes for the integer parameters, in integer units per step (times
# --int-lr). The readout bias is scaled by the readout divisor so that it
# moves at most ~0.01 nat per step whatever the divisor: 1.28 units at 128.
INT_LR = {"pr_bias": 0.01 * spec.LOGIT_DIVISOR, "sc_bias": 0.5,
          "inj_delta": 0.25, "op_b": 0.25,
          # int8 probe coefficients (spec.WIDE_PROBE_COEF). One unit times a
          # tape code of ~40 is 40/divisor of a nat: 0.04 at 1024. Fitted
          # coefficients (--init fit) have a median of ~6, and at an optimum
          # Adam still walks every parameter ~0.2 lr per step on noise alone,
          # so a larger step blurs what the fit found.
          "pr_w": 0.25}


def unigram_bias(records, max_records: int = 50_000,
                 alpha: float = 0.5) -> torch.Tensor:
    """LOGIT_DIVISOR * log p(symbol), centred, from a spread-out sample of the
    training records: the 256 bytes plus one EOS per record -- exactly the
    targets the loss scores. BOS is never a target (it is masked out of the
    softmax), so it just gets the minimum."""
    from mica_r1.spec import N_SYMBOLS, LOGIT_DIVISOR, BOS, EOS
    stride = max(1, len(records) // max_records)
    sample = records[::stride]
    counts = np.zeros(N_SYMBOLS)
    if N_SYMBOLS == 258:
        counts[:256] = np.bincount(np.frombuffer(b"".join(sample), dtype=np.uint8),
                                   minlength=256)
    else:
        for record in sample:
            counts[:BOS] += np.bincount(np.asarray(record, dtype=np.int32),
                                        minlength=BOS)
    counts[EOS] = len(sample)
    targets = np.ones(N_SYMBOLS, dtype=bool)
    targets[BOS] = False
    logp = np.log((counts[targets] + alpha) /
                  (counts[targets] + alpha).sum())
    b = np.zeros(N_SYMBOLS)
    b[targets] = LOGIT_DIVISOR * (logp - logp.mean())
    b[BOS] = b[targets].min()
    return torch.tensor(np.clip(np.round(b), -32767, 32767),
                        dtype=torch.float32)


VAL_BATCH = 16
SIG_STEPS = 9          # index of args.steps in the resume signature
INIT_KEYS = (
    "init", "mode", "sel_init", "bias_init", "work_probes",
    "tape_lags", "work_lags", "rule_max_back", "rule_lag_coverage",
    "rule_scoring", "template_records", "template_phases",
    "rule_self_terms", "rule_state", "reversible_multiplier",
    "probe_reversible_state", "choose_channels",
)
# These options were added after older run_info.json files were written.
# Missing values in those files used exactly these old behaviours.
LEGACY_INIT_DEFAULTS = {
    "rule_lag_coverage": False,
    "template_records": 5000,
    "template_phases": "all",
    "reversible_multiplier": 4,
    "probe_reversible_state": False,
}


def same_run(saved, now) -> bool:
    """Resume only into the same model and schedule shape; the one field
    allowed to differ is the total step count, so a run can be extended."""
    if not isinstance(saved, list) or len(saved) != len(now):
        return False
    return all(a == b for i, (a, b) in enumerate(zip(saved, now))
               if i != SIG_STEPS)


def initialization_signature(args) -> dict:
    """Settings that determine the initial rule book and readout layout."""
    return {key: getattr(args, key) for key in INIT_KEYS}


def resume_problem(saved_sig, now_sig, saved_init, now_init,
                   legacy_args=None) -> str | None:
    """Reject incompatible resumes; older checkpoints use prior run_info."""
    if not same_run(saved_sig, now_sig):
        return "saved geometry or schedule differs from this run"
    prior = (saved_init if saved_init is not None else
             ({**LEGACY_INIT_DEFAULTS, **legacy_args}
              if isinstance(legacy_args, dict) else None))
    if not isinstance(prior, dict) or any(key not in prior for key in INIT_KEYS):
        return "saved initialization settings cannot be verified"
    changed = [key for key in INIT_KEYS if prior[key] != now_init[key]]
    if changed:
        return "saved initialization differs: " + ", ".join(changed)
    return None


def tape_init(model, sel_init: float, margin: float = 3.0,
              seed: int = 1) -> None:
    """Start the field as a clean record of recent bytes.

    Measured on the first model that learned (run 2, step 4,600): a dense
    linear readout of its whole field predicted the next byte WORSE (3.99
    bits) than a table of the previous byte alone (3.72) -- the field was not
    even keeping the last byte cleanly. Its bytes wrote 24 deltas each to
    cells scattered over the whole ring, on top of older writes, and random
    rules rewrote everything twice per byte.

    Here every byte writes its own code (one signed value per injection slot,
    slot e in channel e) into the cell under the write head, which moves on
    by one cell per byte, so with the rolling readout "j bytes ago" is always
    relative cell -(j+1). Every rule candidate starts as HOLD, so nothing
    disturbs that record until learning finds a reason to. Every probe starts
    pointed at one of the last few bytes, reading one code channel, with its
    coefficient OFF, so the model starts exactly at the byte-frequency level
    and each probe turns on only when its read helps.

    The margins are small against the near-uniform selector start: the
    argmax is fixed, but the backward softmax stays nearly flat, so every
    alternative still gets a gradient.
    """
    from mica_r1.spec import N_CELLS, N_CHANNELS, N_INJECT, N_PROBE, HOLD
    g = torch.Generator().manual_seed(seed)
    def noise(t):
        return torch.randn(t.shape, generator=g) * sel_init
    with torch.no_grad():
        dev = model.inj_cell.device
        ic = noise(model.inj_cell); ic[:, :, 0] += margin
        ch = noise(model.inj_chan)
        for e in range(N_INJECT):
            ch[:, e, e % N_CHANNELS] += margin
        mag = 24 + 40 * torch.rand(model.inj_delta.shape, generator=g)
        sign = torch.where(torch.rand(model.inj_delta.shape, generator=g) < 0.5,
                           -1.0, 1.0)
        oc = noise(model.op_code); oc[:, :, HOLD] += margin
        # probes: half on the last byte, a quarter on the one before, ...
        lags = []
        for p in range(N_PROBE):
            lag, share = 1, N_PROBE // 2
            while p >= share and lag < 8:
                p -= share; share = max(1, share // 2); lag += 1
            lags.append(lag)
        pc = noise(model.pr_cell); pch = noise(model.pr_chan)
        for p, lag in enumerate(lags):
            pc[:, p, (N_CELLS - lag) % N_CELLS] += margin
            chans = torch.randint(0, min(N_INJECT, N_CHANNELS), (pch.shape[0],),
                                  generator=g)
            pch[torch.arange(pch.shape[0]), p, chans] += margin
        co = noise(model.pr_co); co[:, :, 1] += 1.0        # TERNARY[1] == 0
        for name, val in (("inj_cell", ic), ("inj_chan", ch),
                          ("inj_delta", sign * mag), ("op_code", oc),
                          ("pr_cell", pc), ("pr_chan", pch), ("pr_co", co)):
            getattr(model, name).copy_(val.to(dev))


def _lag_schedule(n: int, max_lag: int = 8, cap: int = 10**9) -> list:
    """Half the probes on the last byte, half the rest on the one before, and
    so on -- but never more at one lag than there are channels to read (`cap`):
    a second probe on the same channel of the same cell adds nothing."""
    lags, left, lag = [], n, 1
    while left > 0:
        k = left if lag >= max_lag else min(cap, (left + 1) // 2)
        lags += [lag] * k
        left -= k
        lag += 1
    return lags


def _parse_tape_lags(layout: str, count: int, limit: int) -> list[int]:
    """Expand ``lag:probe-count`` groups for an explicit tape probe layout."""
    lags = []
    for group in layout.split(","):
        parts = group.strip().split(":")
        if len(parts) != 2:
            raise ValueError("--tape-lags needs comma-separated lag:count groups")
        try:
            lag, n = (int(part.strip()) for part in parts)
        except ValueError as exc:
            raise ValueError("--tape-lags needs integer lag:count groups") from exc
        if not 1 <= lag <= limit:
            raise ValueError(f"--tape-lags lag {lag} must be in 1..{limit}")
        if n <= 0:
            raise ValueError("--tape-lags counts must be positive")
        if len(lags) + n > count:
            raise ValueError(f"--tape-lags specifies more than {count} probes")
        lags.extend([lag] * n)
    if len(lags) != count:
        raise ValueError(f"--tape-lags specifies {len(lags)} probes; "
                         f"this run needs exactly {count}")
    return lags


def tape_init_ext(model, sel_init: float, margin: float = 3.0,
                  seed: int = 1, work_probes: int = 8,
                  work_layout: list | None = None,
                  tape_lags: str = "") -> list:
    """Start for the tape extensions (spec.TAPE_CHANNELS and friends).

    The tape itself needs no help: it is structural. What needs a start is
    who reads it. Measured with a float fit on the bulk corpus (256-byte
    records): 32 probes reading a clean tape of 16-channel byte codes at lags
    1..6 reach 3.47 bits/byte, against 4.90 for byte frequencies and 3.83 for
    a full bigram table.

    ``tape_lags`` optionally replaces only the tape probe lag schedule. An
    empty string preserves the original initialization exactly.

      * every byte's code: a random signed value of 24..64 on each tape
        channel, so all 256 codes differ and every routing comparison is
        decisive from the start. Work-channel injections start at 0.
      * most probes read a tape channel at lags 1..6, half on the last byte;
        at each lag they take distinct channels, so the last byte is read at
        full rank. A few read WORK channels near the head -- without them no
        gradient ever reaches the rules. Coefficients start at 0, so the
        model starts exactly at the byte-frequency level.
      * rules start random (the selectors' own init). Starting them as HOLD
        would leave the work channels at zero, the work probes with nothing
        to learn from, and the rules with no gradient: a deadlock.
    """
    from mica_r1.spec import (N_CELLS, N_CHANNELS, N_PROBE, TAPE_CHANNELS as K,
                              PROBE_WINDOW)
    g = torch.Generator().manual_seed(seed)
    def noise(t):
        return torch.randn(t.shape, generator=g) * sel_init
    # at least 32 probes stay on the tape
    n_work = min(work_probes, max(0, N_PROBE - 32)) if N_CHANNELS > K else 0
    max_lag = min(8, PROBE_WINDOW or 8)
    tape_schedule = (_parse_tape_lags(
        tape_lags, N_PROBE - n_work, min(PROBE_WINDOW or N_CELLS - 1,
                                        N_CELLS - 1))
        if tape_lags else _lag_schedule(N_PROBE - n_work, max_lag, cap=K))
    lags = ([(lag, "tape") for lag in tape_schedule]
            + [(lag, "work") for lag in _lag_schedule(n_work, max_lag,
                                                      cap=N_CHANNELS - K)])
    if work_layout is not None:
        # explicit (lag, channel) for the work probes (fit.structured_work_probes)
        work_layout = list(work_layout)[:n_work]
        lags = lags[:N_PROBE - n_work] + [(l, "work") for l, _ in work_layout]
    with torch.no_grad():
        dev = model.inj_delta.device
        mag = 24 + 40 * torch.rand(model.inj_delta.shape, generator=g)
        sign = torch.where(torch.rand(model.inj_delta.shape, generator=g) < 0.5,
                           -1.0, 1.0)
        delta = sign * mag
        delta[:, K:] = 0.0
        pc = noise(model.pr_cell)
        pch = noise(model.pr_chan)
        used = {}
        n_tape = N_PROBE - n_work
        for p, (lag, kind) in enumerate(lags):
            k = used.get((lag, kind), 0)
            used[(lag, kind)] = k + 1
            chan = k % K if kind == "tape" else K + k % (N_CHANNELS - K)
            if kind == "work" and work_layout is not None:
                chan = work_layout[p - n_tape][1]
            pc[:, p, (N_CELLS - lag) % N_CELLS] += margin
            pch[:, p, chan] += margin
        for name, val in (("inj_delta", delta), ("pr_cell", pc),
                          ("pr_chan", pch)):
            getattr(model, name).copy_(val.to(dev))
        if spec.WIDE_PROBE_COEF:
            model.pr_w.zero_()
        else:
            co = noise(model.pr_co); co[:, :, 1] += 1.0     # TERNARY[1] == 0
            model.pr_co.copy_(co.to(dev))
    return lags


def apply_optional_ppmi_codes(model) -> str:
    """Copy an int8 tape-code table onto the tape channels, if asked.

    FLAMEW_PPMI_CODES unset or empty does nothing, so an ordinary run is
    unchanged. The file holds an int8 ``codes`` array of shape
    (N_SYMBOLS, TAPE_CHANNELS). Work-channel injections are not written.
    """
    path = os.environ.get("FLAMEW_PPMI_CODES", "").strip()
    if not path:
        return ""
    try:
        codes = np.load(path)["codes"]
    except Exception as exc:
        raise SystemExit(
            f"[train] FLAMEW_PPMI_CODES {path} unreadable: {exc}") from exc
    k = int(spec.TAPE_CHANNELS)
    expect = (int(spec.N_SYMBOLS), k)
    if tuple(getattr(codes, "shape", ())) != expect or codes.dtype != np.int8:
        raise SystemExit(
            f"[train] FLAMEW_PPMI_CODES {path} is "
            f"{tuple(getattr(codes, 'shape', ()))} {getattr(codes, 'dtype', None)}; "
            f"need {expect} int8")
    lo, hi = int(codes.min()), int(codes.max())
    if lo < -127 or hi > 127:
        raise SystemExit(
            f"[train] FLAMEW_PPMI_CODES {path} range {lo}..{hi} "
            "is outside -127..127")
    with torch.no_grad():
        model.inj_delta[:, :k].copy_(
            torch.from_numpy(np.ascontiguousarray(codes)).to(
                device=model.inj_delta.device, dtype=model.inj_delta.dtype))
    print("[train] tape codes loaded from FLAMEW_PPMI_CODES "
          f"{path} after tape_init_ext and before the tape readout fit; "
          "work-channel injections unchanged", flush=True)
    return path


def fit_tape_readout(model, records, lags, dev, steps: int = 3000,
                     batch: int = 16384, max_records: int = 40_000,
                     reserve_word_route: bool = False,
                     reserve_letter_route: bool = False,
                     log=print) -> float:
    """Fit the byte codes, probe coefficients and biases to the data directly.

    With the tape extensions the readout of the tape is an ordinary additive
    model: logit(v) = bias[v] + sum_p w[v,p] * code[byte at lag_p][chan_p] /
    divisor. Nothing about it needs the straight-through estimator, so it is
    fitted here in closed-loop float arithmetic -- with the same rounding and
    ranges the model file has -- in seconds, instead of being discovered one
    noisy hard step at a time over thousands of steps. The rules, whose
    gradient does need the machine, start learning from there.

    Only the tape probes are fitted; the work probes (rule-written channels)
    keep coefficient 0. The optional route constraints pin MICA tape codes
    so word-state pages can distinguish word bytes or individual letters.
    Returns the fitted bits/byte on held-out positions of the sample, which
    the step-1 validation should reproduce.
    """
    from mica_r1.spec import (N_SYMBOLS, BOS, EOS, LOGIT_DIVISOR,
                              TAPE_CHANNELS as K)
    if N_SYMBOLS > 258:
        if reserve_word_route or reserve_letter_route:
            raise ValueError("byte boundary and letter routes are invalid for word symbols")
        batch = min(batch, 512)
    tape = [(p, lag) for p, (lag, kind) in enumerate(lags) if kind == "tape"]
    if not tape:
        return float("nan")
    max_lag = max(l for _, l in tape)
    with torch.no_grad():
        chans = model.logits_of("pr_chan").argmax(-1)[0]      # (P,) same for all v
    stride = max(1, len(records) // max_records)
    sample = [r for r in records[::stride] if len(r)]
    ZERO = N_SYMBOLS                     # "no byte here yet": code 0
    ulags = sorted({lag for _, lag in tape})
    ctx, tgt = [], []
    for r in sample:
        a = (np.frombuffer(r, np.uint8).astype(np.int32) if N_SYMBOLS == 258
             else np.asarray(r, dtype=np.int32))
        seq = np.concatenate([np.full(max_lag, ZERO, np.int32),
                              np.array([BOS], np.int32), a])
        n = len(a) + 1                                      # bytes, then EOS
        t = np.arange(n)
        # the symbol `lag` back from target t is seq[max_lag + t + 1 - lag]
        ctx.append(np.stack([seq[max_lag + t + 1 - lag] for lag in ulags], 1))
        tgt.append(np.concatenate([a, np.array([EOS], np.int32)]))
    ctx = torch.from_numpy(np.concatenate(ctx)).long().to(dev)   # (n,lags)
    tgt = torch.from_numpy(np.concatenate(tgt)).long().to(dev)
    lcol = torch.tensor([ulags.index(lag) for _, lag in tape], device=dev)
    n_hold = min(len(tgt) // 10, 4096 if N_SYMBOLS > 258 else 200_000)
    perm = torch.randperm(len(tgt), generator=torch.Generator().manual_seed(0))
    perm = perm.to(dev)
    hold, fit = perm[:n_hold], perm[n_hold:]
    pidx = torch.tensor([p for p, _ in tape], device=dev)
    pch = chans.to(dev)[pidx]                                # (Pt,)
    elig = torch.full((N_SYMBOLS,), float("-inf"), device=dev)
    elig[:BOS] = 0.0
    elig[EOS] = 0.0

    def rnd(x, lo, hi):                  # the file's rounding, straight-through
        return x + (x.detach().round().clamp(lo, hi) - x.detach())

    with torch.no_grad():
        code0 = model.inj_delta[:, :K].detach().clone().to(dev)
        w0 = torch.zeros(N_SYMBOLS, len(tape), device=dev)
        b0 = model.pr_bias.detach().clone().to(dev)
    code = code0.clone().requires_grad_()
    w = w0.clone().requires_grad_()
    bias = b0.clone().requires_grad_()
    # Empty keeps the old fit. "all" freezes every tape channel. A positive
    # integer freezes that many leading channels (the routing channels are 0..4).
    freeze_spec = os.environ.get("FLAMEW_FREEZE_TAPE_CODES", "").strip()
    if not freeze_spec:
        n_freeze = 0
    elif freeze_spec == "all":
        n_freeze = K
    else:
        n_freeze = int(freeze_spec)
        if not 0 < n_freeze <= K:
            raise SystemExit(
                f"[train] FLAMEW_FREEZE_TAPE_CODES {freeze_spec} "
                f"is outside 1..{K}")
    if n_freeze:
        log(f"[fit] freezing {n_freeze} leading tape-code channels")
    opt = torch.optim.Adam([{"params": [code], "lr": 1.0},
                            {"params": [w], "lr": 1.0},
                            {"params": [bias], "lr": 0.01 * LOGIT_DIVISOR}])

    def logits(rows):
        effective = rnd(code, -127, 127)
        if reserve_word_route:
            from mica_r1.fit import reserve_word_boundary_route
            effective = reserve_word_boundary_route(effective)
        if reserve_letter_route:
            from mica_r1.fit import reserve_letter_routes
            effective = reserve_letter_routes(effective)
        c = torch.cat([effective,
                       torch.zeros(N_SYMBOLS + 1 - code.shape[0], K,
                                   device=dev)])            # up to ZERO
        vals = c[ctx[rows][:, lcol], pch[None, :]]           # (n,Pt)
        sc = rnd(bias, -32767, 32767) + vals @ rnd(w, -127, 127).T
        return sc / LOGIT_DIVISOR + elig

    t0 = time.time()

    def held_out():
        with torch.no_grad():
            chunk = 512 if N_SYMBOLS > 258 else 65536
            return sum(float(Fn.cross_entropy(logits(hold[i:i + chunk]),
                                              tgt[hold[i:i + chunk]],
                                              reduction="sum"))
                       for i in range(0, n_hold, chunk)) / n_hold / math.log(2)
    # early stopping: keep the best parameters seen on the held-out positions
    best_h, keep = float("inf"), None
    for it in range(steps):
        rows = fit[torch.randint(0, len(fit), (batch,), device=dev)]
        loss = Fn.cross_entropy(logits(rows), tgt[rows])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if n_freeze and code.grad is not None:
            code.grad[:, :n_freeze] = 0
        opt.step()
        if (it + 1) % 250 == 0 or it == steps - 1:
            h = held_out()
            if h < best_h:
                best_h = h
                keep = [t.detach().clone() for t in (code, w, bias)]
            if it % 1000 == 999 or it == steps - 1:
                log(f"[fit] step {it + 1:5d}  held-out {h:.4f} bits/byte "
                    f"(best {best_h:.4f}, {time.time() - t0:.0f}s)")
    code, w, bias = keep
    if n_freeze:
        code = code.clone()
        code[:, :n_freeze] = code0[:, :n_freeze]
    if reserve_word_route:
        from mica_r1.fit import reserve_word_boundary_route
        code = reserve_word_boundary_route(code)
    if reserve_letter_route:
        from mica_r1.fit import reserve_letter_routes
        code = reserve_letter_routes(code)
    with torch.no_grad():
        model.inj_delta[:code.shape[0], :K] = code.round().clamp(-127, 127).to(
            model.inj_delta)
        wr = w.round().clamp(-127, 127).to(model.pr_w)
        model.pr_w[:, pidx.to(model.pr_w.device)] = wr
        model.pr_bias.copy_(bias.round().clamp(-32767, 32767).to(model.pr_bias))
    return best_h


def mark_finished(out: Path, step: int) -> None:
    """Tells the auto-restart watcher the run is complete, so it stops
    relaunching it. Removed again as soon as a run has steps left to do."""
    try:
        (out / "FINISHED").write_text(
            f"finished at step {step}, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except OSError:
        pass


def should_stop(min_ram_gb: float):
    if STOP_FILE.exists():
        return f"{STOP_FILE.name} found"
    r = free_ram_gb()
    if r < min_ram_gb:
        return f"free system RAM fell to {r:.1f} GB (floor {min_ram_gb} GB)"
    return None


def batches(recs, size, rng, device):
    """Length-bucketed batches: padding a 200-byte record out to 1,024 wastes
    most of the compute on BOS filler."""
    order = sorted(range(len(recs)), key=lambda i: len(recs[i]))
    blocks = [order[i:i + size] for i in range(0, len(order), size)]
    rng.shuffle(blocks)
    for bl in blocks:
        rs = [recs[i] for i in bl]
        L = max(len(r) for r in rs)
        b = torch.zeros(len(rs), L, dtype=torch.long, device=device)
        ln = torch.tensor([len(r) for r in rs], device=device)
        for j, r in enumerate(rs):
            b[j, :len(r)] = torch.tensor(list(r), device=device)
        yield b, ln


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="r1/data/gsm8k/train.jsonl")
    ap.add_argument("--val-records", default="r1/data/gsm8k/val.jsonl")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--record-bytes", type=int, default=1024)
    ap.add_argument("--ticks", type=int, default=2,
                    help="ticks per symbol during training. Each one is "
                         "another sequential step to backpropagate through, so "
                         "this is the depth knob, not a quality knob.")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--sel-init", type=float, default=2.0,
                    help="scale of the random selector logits at the start. "
                         "Near 0 (with --sel-tau ~1) gives every option a "
                         "gradient; 2.0 (the original) gives almost all of it "
                         "to the few initially-top options")
    ap.add_argument("--sel-tau", type=float, default=None,
                    help="temperature of the selector softmax in the backward "
                         "pass (default: --tau)")
    ap.add_argument("--lr", type=float, default=0.02,
                    help="step size for the selector logits (which cell, "
                         "channel, opcode, coefficient)")
    ap.add_argument("--int-lr", type=float, default=1.0,
                    help="scale for the step sizes of the parameters that are "
                         "INTEGERS in the model file -- biases, injection "
                         "deltas, operands -- which live in integer units: "
                         "readout bias 0.01 nat/step (1.28 units at divisor "
                         "128), candidate bias 0.5, injection delta and "
                         "operand 0.25, all times this")
    ap.add_argument("--init", choices=["random", "tape", "fit", "fit-rules"],
                    default="random",
                    help="how a FRESH run starts. 'tape': every byte writes a "
                         "distinct code into the cell under the write head, "
                         "every rule starts as HOLD, and every probe points "
                         "at one of the last few bytes' cells with its "
                         "coefficient off -- the field starts as a clean "
                         "history of recent bytes instead of a scramble. "
                         "Ignored when resuming.")
    ap.add_argument("--work-probes", type=int, default=16,
                    help="with the tape extensions: how many of each symbol's "
                         "probes start on rule-written work channels instead "
                         "of the tape (at most a quarter). They are the "
                         "rules' only path to the loss.")
    ap.add_argument("--tape-lags", default="",
                    help="optional comma-separated lag:count layout for the "
                         "tape probes, e.g. 1:16,2:8,3:4,4:4,8:4,16:4,24:4,32:4. "
                         "Counts must total N_PROBE minus active work probes; "
                         "lags must fit MICA_PROBE_WINDOW. Empty uses the "
                         "original layout.")
    ap.add_argument("--work-lags", default="",
                    help="optional comma-separated lag:VSET-group-count layout "
                         "for fit-rules work probes, e.g. 1:16,2:8,8:4,16:4 "
                         "with 192 probes and VSET width 6. Empty preserves "
                         "the original layout.")
    ap.add_argument("--bias-init", choices=["unigram", "zero"],
                    default="unigram",
                    help="start the readout biases at the training data's "
                         "byte frequencies (what R1's mutation search learned "
                         "to do) instead of at zero")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--checkpoint-every", type=int, default=1,
                    help="non-zero recomputes each symbol's field instead of "
                         "storing its activations. 0 stores everything and "
                         "will run out of memory on long records.")
    ap.add_argument("--mode", choices=["st", "fit"], default="st",
                    help="st: straight-through gradient steps. fit: rounds of "
                         "exact refits of the readout and the rule immediates "
                         "on fresh samples (needs --init fit-rules and the "
                         "tape extensions); a --steps is then a round")
    ap.add_argument("--selector-flips-per-phase", type=int, default=0,
                    help="ST mode: gradient-guided hard score-selector changes "
                         "per phase and selector kind at each projection; "
                         "the exported integer MICA engine is unchanged")
    ap.add_argument("--selector-project-every", type=int, default=5,
                    help="ST mode: project scoring selectors every N steps")
    ap.add_argument("--selector-project-margin", type=float, default=1.0,
                    help="new hard selector argmax margin after projection")
    ap.add_argument("--rule-max-back", default="1,2,2,2",
                    help="fit-rules: per phase, how many of the nearest older "
                         "neighbours a candidate's scoring may read")
    ap.add_argument("--rule-lag-coverage", action="store_true",
                    help="fit-rules: use distinct allowed neighbour offsets "
                         "per candidate before repeating any; the exported "
                         "integer cellular engine is unchanged")
    ap.add_argument("--rule-scoring", default="balance",
                    choices=["balance", "template", "template+balance",
                             "informed+balance",
                             "random"],
                    help="fit-rules: how candidates are scored. balance: "
                         "random terms, biases set so candidates win equally "
                         "often; template: each candidate matches one of the "
                         "page's most frequent contexts")
    ap.add_argument("--template-records", type=int, default=5000,
                    help="maximum training records used to select rule "
                         "context templates; fitting still uses --fit-records")
    ap.add_argument("--template-phases", default="all",
                    help="comma-separated phase indices to receive context "
                         "templates; remaining phases keep random rules")
    ap.add_argument("--rule-self-terms", type=int, default=0,
                    help="fit-rules: how many of a candidate's six scoring "
                         "terms read the cell's own byte (useful when few "
                         "routing bits make a page cover several bytes)")
    ap.add_argument("--rule-state", choices=["none", "word", "word-reserved",
                                             "stream", "reversible",
                                             "reversible2", "lexical2",
                                             "lexical2-letters",
                                             "lexical2-track", "lexical2-s1"],
                    default="none",
                    help="fit-rules with VSET: 'word' hashes bytes within "
                         "a word and resets at separators; 'word-reserved' "
                         "reserves one routing bit so those resets are exact; "
                         "'stream' hashes separators too")
    ap.add_argument("--reversible-multiplier", type=int, default=4,
                    help="reversible rule selection: invertible state multiplier "
                         "modulo 243; compare candidates on held-out text")
    ap.add_argument("--probe-reversible-state", action="store_true",
                    help="include the existing reversible state channels in "
                         "the MICA readout probes")
    ap.add_argument("--choose-channels", action="store_true",
                    help="fit-rules: let each rule candidate pick the channel "
                         "of its group it writes (fit.choose_channels)")
    ap.add_argument("--fit-records", type=int, default=20000,
                    help="fit mode: records drawn per round")
    ap.add_argument("--fit-steps", type=int, default=1500,
                    help="fit mode: Adam steps per round")
    ap.add_argument("--round-lr", type=float, default=0.3,
                    help="fit mode: step-size scale for the refit rounds "
                         "(the first fit uses 1). Smaller steps make a round a "
                         "nudge toward its sample rather than a jump to it, "
                         "so the rounds average over the corpus.")
    ap.add_argument("--word-start-weight", type=float, default=1.0,
                    help="fit-rules: weight the first letter after a space "
                         "or BOS in the fitted readout loss; 1 is unchanged")
    ap.add_argument("--segments", action="store_true",
                    help="backpropagate each --tbptt segment as soon as it "
                         "ends (same gradient, far less memory, no "
                         "recomputation). Meant for the windowed machine "
                         "(MICA_WINDOW), whose segments are small.")
    ap.add_argument("--min-batch", type=int, default=1,
                    help="on an out-of-memory error the batch is halved and "
                         "the step retried, down to this. An unattended run "
                         "that dies on the first step has wasted the night.")
    ap.add_argument("--soft", action="store_true",
                    help="soft forward pass. Trains to a better number and "
                         "does NOT survive discretisation. Comparison only.")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-records-n", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.75,
                    help="hard cap on VRAM, as a fraction of the card. The "
                         "rest stays free for Windows and the display.")
    ap.add_argument("--min-free-ram-gb", type=float, default=4.0,
                    help="stop cleanly if free system RAM drops below this")
    ap.add_argument("--tbptt", type=int, default=8,
                    help="let the gradient flow back at most this many bytes. "
                         "The field still carries memory across the whole "
                         "record; only the learning signal is truncated. "
                         "Measured on the reference geometry at 256 bytes: "
                         "8 -> norm 38, 16 -> 2e7, 32 -> 3e19, 64 -> 2e27, "
                         "untruncated -> infinite.")
    ap.add_argument("--save-every", type=int, default=50,
                    help="write the full resumable state every N steps")
    ap.add_argument("--stall-minutes", type=float, default=25,
                    help="if one step (or one validation) takes longer than "
                         "this, write every thread's Python stack to "
                         "stall_trace.txt in --out, repeating while it lasts. "
                         "Nothing is stopped; it records WHERE a hang is.")
    ap.add_argument("--log-every", type=int, default=10,
                    help="print a short progress line (time, speed, train "
                         "loss) every N steps. Without it the log is silent "
                         "between validations, and a stalled run looks "
                         "exactly like a slow one.")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any saved state and start from step 0")
    ap.add_argument("--no-yield", action="store_true",
                    help="do NOT pause when another program (a game) takes "
                         "the GPU")
    ap.add_argument("--yield-gb", type=float, default=3.0,
                    help="pause when other programs use this many GiB of GPU "
                         "memory more than they did when training started")
    ap.add_argument("--out", default="r1/runs/soft")
    ap.add_argument("--dry-run", action="store_true",
                    help="time one step and report memory, then stop")
    args = ap.parse_args()
    if args.word_start_weight < 1:
        ap.error("--word-start-weight must be at least 1")
    if args.rule_lag_coverage and args.init != "fit-rules":
        ap.error("--rule-lag-coverage requires --init fit-rules")
    if args.template_records < 1:
        ap.error("--template-records must be positive")
    if args.template_phases != "all" and args.rule_scoring not in (
            "template", "template+balance", "informed+balance"):
        ap.error("--template-phases requires template rule scoring")
    if args.word_start_weight != 1 and (args.mode != "fit" or args.init != "fit-rules"):
        ap.error("--word-start-weight requires --mode fit --init fit-rules")
    if args.word_start_weight != 1 and args.choose_channels:
        ap.error("--word-start-weight is not supported with --choose-channels")
    if args.rule_state != "none" and args.choose_channels:
        ap.error("--choose-channels changes recurrent state routing; it is "
                 "not supported with --rule-state")
    if args.rule_state == "word-reserved" and (args.mode != "fit" or
                                               args.init != "fit-rules"):
        ap.error("--rule-state word-reserved requires --mode fit --init fit-rules")
    if (args.rule_state in ("reversible", "reversible2") or
            args.rule_state.startswith("lexical2")) and args.init != "fit-rules":
        ap.error("integer-rule state requires --init fit-rules")
    if args.rule_state in ("reversible", "reversible2") and math.gcd(
            args.reversible_multiplier, 243) != 1:
        ap.error("--reversible-multiplier must be coprime to 243")
    if args.rule_state not in ("reversible", "reversible2") and \
            args.reversible_multiplier != 4:
        ap.error("--reversible-multiplier requires reversible rule state")
    if args.probe_reversible_state and args.rule_state != "reversible":
        ap.error("--probe-reversible-state requires --rule-state reversible")
    if args.selector_flips_per_phase < 0 or args.selector_project_every < 1 or \
            args.selector_project_margin <= 0:
        ap.error("invalid selector projection budget, interval, or margin")
    if args.selector_flips_per_phase and args.mode != "st":
        ap.error("--selector-flips-per-phase requires --mode st")
    if args.work_lags and (args.mode != "fit" or args.init != "fit-rules"):
        ap.error("--work-lags requires --mode fit --init fit-rules")
    if spec.N_SYMBOLS > 258:
        if args.mode != "fit" or args.init != "fit-rules" or args.dry_run:
            ap.error("word symbols currently require native --mode fit --init fit-rules")
        if args.word_start_weight != 1 or args.rule_state != "none":
            ap.error("byte-only word-start weighting and rule states are invalid for word symbols")
        if args.rule_scoring == "informed+balance":
            ap.error("dense information histograms are invalid for word symbols")

    # SoftMica always routes by pairdiff (F[c] >= F[c + offset]). The integer
    # engine defaults to R1's sign routing, so until 2026-09-25 every
    # "integer_bits" this script printed came from a DIFFERENT machine than
    # the one trained. Make them the same one.
    from mica_r1 import engine as _engine
    if _engine.ROUTING_MODE != "pairdiff":
        print(f"[train] integer checks: routing {_engine.ROUTING_MODE} -> "
              f"pairdiff, the rule the soft model trains", flush=True)
        _engine.ROUTING_MODE = "pairdiff"

    dev = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
           if args.device == "auto" else torch.device(args.device))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    harden(dev, args.gpu_mem_fraction)
    if STOP_FILE.exists():
        print(f"[guard] {STOP_FILE} exists -- delete it to allow a run.")
        return 0
    rng = __import__("random").Random(1)
    torch.manual_seed(1)

    trim = lambda rs: [r[:args.record_bytes] for r in rs]
    train = trim(load_records(args.records))
    val = trim(load_records(args.val_records))[:args.val_records_n]

    model = SoftMica(ticks=args.ticks, tau=args.tau, hard=not args.soft,
                     sel_init=args.sel_init, sel_tau=args.sel_tau,
                     compact_selectors=spec.N_SYMBOLS > 258).to(dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"[train] {spec.N_CELLS} cells x {spec.N_CHANNELS} channels, "
          f"{spec.N_PAGES} pages x {spec.N_CANDIDATES} candidates")
    print(f"[train] {n:,} soft parameters ({n * 4 / 1e6:.0f} MB fp32) "
          f"-> {spec.TOTAL_BYTES:,} byte integer model "
          f"({n * 4 / spec.TOTAL_BYTES:.0f}x collapse)")
    print(f"[train] forward pass: {'SOFT (will not discretise)' if args.soft else 'HARD (straight-through)'}")
    print(f"[train] {len(train):,} train records, {len(val):,} validation, "
          f"device {dev}")

    # Two kinds of parameter, two step sizes. Adam moves a parameter by about
    # `lr` per step whatever its gradient's size. The selector logits are
    # unit-scale, so 0.02 is sensible there. The integer parameters are not:
    # the readout divides by 128, so one unit of probe bias is 1/128 of a nat,
    # and a byte-frequency table needs a spread of ~1,800 units. At 0.02 per
    # step they never moved -- in the step-100 model of the first real run
    # every probe bias, candidate bias and operand was still exactly 0.
    groups = [{"params": [q for nme, q in model.named_parameters()
                          if nme not in INT_LR], "lr": args.lr}]
    for nme, q in model.named_parameters():
        if nme in INT_LR:
            groups.append({"params": [q], "lr": INT_LR[nme] * args.int_lr})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    def step_with_retry(b, ln):
        """One forward+backward, halving the batch on out-of-memory.

        The retry happens OUTSIDE the except block on purpose. Inside it, the
        exception's traceback still references every frame of the failed step
        and therefore every tensor it allocated, so empty_cache() can free
        nothing and the smaller retry fails too -- which is exactly what the
        last run did at batch 4, then 2, then 1.
        """
        while True:
            oom = False
            loss = None
            try:
                if args.segments:
                    loss, _ = model.backward_tbptt(b, ln, args.tbptt)
                else:
                    loss, _ = model.sequence_loss(b, ln, args.checkpoint_every,
                                                  tbptt=args.tbptt)
                    loss.backward()
                return loss, b.shape[0]
            except torch.OutOfMemoryError:
                oom = True
            # An OOM inside backward() leaves `loss` holding the whole failed
            # graph. Drop it before retrying, or every smaller batch fails
            # too: on 2026-09-24 a batch-16 OOM cascaded all the way to 1.
            loss = None
            opt.zero_grad(set_to_none=True)
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            if b.shape[0] <= args.min_batch:
                raise RuntimeError(
                    f"out of memory even at batch {b.shape[0]} under the "
                    f"{args.gpu_mem_fraction:.0%} VRAM cap")
            half = max(args.min_batch, b.shape[0] // 2)
            print(f"  out of memory at batch {b.shape[0]}, retrying at "
                  f"{half}", flush=True)
            b, ln = b[:half], ln[:half]

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    if args.dry_run:
        b0, ln0 = next(batches(train, args.batch, rng, dev))
        # Try the requested batch, then 8, 4, 2, 1, each attempt from a clean
        # slate, and report the time and peak memory of the one that fits --
        # not of the failed attempts before it.
        sizes = [b0.shape[0]] + [x for x in (24, 16, 12, 8, 4, 2, 1)
                                 if x < b0.shape[0]]
        used = dt = peak = None
        for size in sizes:
            b, ln = b0[:size], ln0[:size]
            loss = None
            opt.zero_grad(set_to_none=True)
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            # Wait for the GPU before reading the clock, both ends: a kernel
            # launch returns at once, so without synchronize() this measured
            # how fast Python could QUEUE a step (14.8 s), not how long the
            # GPU took to run it (~88 s).
            sync(); t = time.time()
            try:
                if args.segments:
                    loss, _ = model.backward_tbptt(b, ln, args.tbptt)
                else:
                    loss, _ = model.sequence_loss(b, ln, args.checkpoint_every,
                                                  tbptt=args.tbptt)
                    loss.backward()
                sync()
                dt = time.time() - t
                used = size
            except torch.OutOfMemoryError:
                print(f"  out of memory at batch {size}", flush=True)
            loss = None
            if used:
                if dev.type == "cuda":
                    peak = torch.cuda.max_memory_allocated() / 2**30
                break
        opt.zero_grad(set_to_none=True)
        if not used:
            print("[dry-run] out of memory even at batch 1")
            return 1
        if used != b0.shape[0]:
            print(f"[dry-run] batch reduced to {used} to fit")
        b = b0[:used]
        print(f"\n[dry-run] batch {tuple(b.shape)}  one step {dt:.2f}s")
        # Validation is forward-only, but it runs every record in the
        # validation set, every --val-every steps. Time one batch of it.
        vbs = max(1, min(args.batch, VAL_BATCH))
        vb, vln = next(batches(val, vbs, __import__("random").Random(0), dev))
        model.eval()
        with torch.no_grad():
            sync(); t = time.time()
            model.sequence_loss(vb, vln)
            sync(); dv = time.time() - t
        model.train()
        n_val = args.steps // args.val_every + 1
        one_val = math.ceil(len(val) / vbs) * dv
        print(f"[dry-run] one validation ({len(val)} records): "
              f"{one_val / 60:.1f} min")
        if peak is not None:
            print(f"[dry-run] peak GPU memory at batch {used}: {peak:.2f} GiB")
        print(f"[dry-run] free system RAM after the step: "
              f"{free_ram_gb():.1f} GB")
        print(f"[dry-run] {args.steps:,} steps + {n_val} validations = "
              f"{(dt * args.steps + n_val * one_val) / 3600:.1f} hours")
        return 0

    # =====================================================================
    # Resume
    #
    # A run measured in hours will be interrupted -- a game, a Windows
    # update, a reboot. Without resume every interruption meant step 0 again.
    # The full state (weights, Adam's moments, the learning-rate schedule, the
    # step, the best score so far) is written every --save-every steps, and
    # written atomically: a crash in the middle of a save can never destroy
    # the only checkpoint.
    # =====================================================================
    RESUME = out / "resume.pt"
    RUN_INFO = out / "run_info.json"
    existing = (RESUME, out / "best.mica", out / "best.pt")
    if args.fresh and any(path.exists() for path in existing):
        raise SystemExit("[resume] --fresh needs a new --out folder; "
                         "existing checkpoints are preserved")
    if not RESUME.exists() and any(path.exists() for path in existing[1:]):
        raise SystemExit("[resume] found a best model without resume.pt; "
                         "use a new --out folder to preserve it")
    try:
        previous_info = json.loads(RUN_INFO.read_text())
    except (OSError, ValueError):
        previous_info = {}
    if not isinstance(previous_info, dict):
        previous_info = {}
    init_sig = initialization_signature(args)
    sig = [spec.N_CELLS, spec.N_CHANNELS, spec.N_PAGES, spec.N_CANDIDATES,
           spec.N_INJECT, spec.N_PROBE, spec.LOGIT_DIVISOR, args.ticks,
           args.tau, args.steps, bool(args.soft), bool(spec.ROLLING_READOUT)]
    if spec.EXTENDED:
        # appended, so SIG_STEPS still indexes args.steps
        sig += [spec.TAPE_CHANNELS, spec.WINDOW, spec.PROBE_WINDOW,
                bool(spec.WIDE_PROBE_COEF), list(spec.OFFSETS)]
    if spec.N_SYMBOLS != 258:
        sig.append("symbols=" + str(spec.N_SYMBOLS))
    if args.tape_lags:
        # Keep older default-layout checkpoints resumable. An explicit layout
        # must never resume a shape-compatible checkpoint with different probes.
        sig.append("tape_lags=" + args.tape_lags)
    if args.work_lags:
        sig.append("work_lags=" + args.work_lags)
    if args.rule_lag_coverage:
        sig.append("rule_lag_coverage=" + args.rule_max_back +
                   "/self=" + str(args.rule_self_terms))
    if args.rule_scoring in ("template", "template+balance",
                             "informed+balance") and args.template_records != 5000:
        sig.append(f"template_records={args.template_records}")
    if args.template_phases != "all":
        sig.append(f"template_phases={args.template_phases}")
    if args.rule_state == "stream":
        # A recurrent-state run must not resume a same-geometry tape-only or
        # word-reset checkpoint. Keep older signatures valid for those modes.
        sig.append("rule_state=stream")
    if args.rule_state == "reversible":
        sig.append("rule_state=reversible")
        if args.reversible_multiplier != 4:
            sig.append(f"reversible_multiplier={args.reversible_multiplier}")
        if args.probe_reversible_state:
            sig.append("probe_reversible_state=1")
    if args.rule_state == "reversible2":
        sig.append("rule_state=reversible2")
        if args.reversible_multiplier != 4:
            sig.append(f"reversible_multiplier={args.reversible_multiplier}")
    if args.rule_state.startswith("lexical2"):
        sig.append(f"rule_state={args.rule_state}")
    if args.rule_state == "word-reserved":
        sig.append("rule_state=word-reserved")
    if args.word_start_weight != 1:
        sig.append(f"word_start_weight={args.word_start_weight:g}")
    def save_state(step, best, hist):
        tmp = out / "resume.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "best": best,
                    "hist": hist, "sig": sig, "init_sig": init_sig}, tmp)
        os.replace(tmp, RESUME)

    hist, best, step = [], float("inf"), 0
    if RESUME.exists() and not args.fresh:
        try:
            ck = torch.load(RESUME, map_location=dev, weights_only=False)
        except Exception as exc:
            raise SystemExit(f"[resume] cannot read {RESUME}: {exc}; "
                             "existing checkpoints are preserved") from exc
        # The total step count only shapes the learning-rate schedule, so
        # a run can be extended by resuming with a larger --steps.
        problem = resume_problem(ck.get("sig"), sig, ck.get("init_sig"),
                                 init_sig, previous_info.get("args"))
        if problem:
            raise SystemExit(f"[resume] {problem}; use the original options "
                             "or a new --out folder. Existing checkpoints "
                             "are preserved")
        model.load_state_dict(ck["model"])
        step, best, hist = ck["step"], ck["best"], ck["hist"]
        note = ""
        if len(ck["opt"]["param_groups"]) == len(opt.param_groups):
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
        else:
            # Saved before the integer parameters had their own step sizes.
            # The weights carry over; Adam's moments and warm-up start again.
            note = (" -- optimizer reset for the new step sizes, "
                    "warm-up starts again")
        print(f"[resume] continuing from step {step} (best validation "
              f"so far {best:.4f} bits){note}", flush=True)
        del ck
    # Write metadata only after compatibility has been checked. A failed
    # resume must leave even the old run_info.json untouched.
    try:
        RUN_INFO.write_text(json.dumps({
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rolling_readout": bool(spec.ROLLING_READOUT),
            "logit_divisor": spec.LOGIT_DIVISOR,
            "env": {k: v for k, v in sorted(os.environ.items())
                    if k.startswith("MICA_")},
            "args": vars(args), "init_sig": init_sig}, indent=2))
    except OSError:
        pass
    if args.init in ("tape", "fit", "fit-rules") and step == 0 and \
            spec.TAPE_CHANNELS:
        from mica_r1 import fit as _fit
        layout = (_fit.structured_work_probes(
                      args.work_probes, ticks=args.ticks,
                      skip_groups=(0, 1) if args.rule_state == "reversible2" or args.rule_state.startswith("lexical2")
                      else () if args.probe_reversible_state
                      else (0,) if args.rule_state in ("word", "word-reserved", "stream", "reversible")
                      else (), work_lags=args.work_lags)
                  if args.init == "fit-rules" else None)
        lags = tape_init_ext(model, args.sel_init,
                             work_probes=args.work_probes, work_layout=layout,
                             tape_lags=args.tape_lags)
        tape = [l for l, k in lags if k == "tape"]
        work = [l for l, k in lags if k == "work"]
        print(f"[train] tape start: {len(tape)} probes read the byte tape at "
              f"lags {sorted(set(tape))}, {len(work)} read rule-written work "
              f"channels at lags {sorted(set(work))}; all coefficients 0",
              flush=True)
        apply_optional_ppmi_codes(model)
        if args.init in ("fit", "fit-rules") and spec.WIDE_PROBE_COEF:
            if args.bias_init == "unigram":
                with torch.no_grad():
                    model.pr_bias.copy_(unigram_bias(train).to(model.pr_bias))
            print("[train] fitting the byte codes and the tape probes' "
                  "coefficients to the data directly (fit_tape_readout)",
                  flush=True)
            h = fit_tape_readout(model, train, lags, dev,
                                 steps=2 * args.fit_steps,
                                 max_records=2 * args.fit_records,
                                 reserve_word_route=args.rule_state in ("word-reserved", "lexical2", "lexical2-track", "lexical2-s1"),
                                 reserve_letter_route=args.rule_state == "lexical2-letters",
                                 log=lambda m: print(m, flush=True))
            print(f"[train] the tape readout starts at {h:.4f} bits/byte on "
                  f"held-out positions; the rules learn from there",
                  flush=True)
        if args.init == "fit-rules" and spec.WIDE_PROBE_COEF:
            mb = tuple(int(x) for x in args.rule_max_back.split(","))
            _fit.structured_rules(model, args.sel_init, max_back=mb,
                                  self_terms=args.rule_self_terms,
                                  lag_coverage=args.rule_lag_coverage)
            if args.rule_scoring in ("template", "template+balance",
                                     "informed+balance"):
                try:
                    phases = (tuple(range(spec.N_PHASE)) if
                              args.template_phases == "all" else
                              tuple(int(p) for p in
                                    args.template_phases.split(",")))
                except ValueError as exc:
                    ap.error(f"invalid --template-phases: {exc}")
                _fit.template_rules(model, train, dev, max_back=mb,
                                    max_records=min(args.fit_records,
                                                    args.template_records),
                                    ranking="information" if
                                    args.rule_scoring == "informed+balance"
                                    else "frequency",
                                    phases=phases,
                                    log=lambda m: print(m, flush=True))
            if args.rule_state in ("word", "word-reserved", "stream", "reversible", "reversible2") or args.rule_state.startswith("lexical2"):
                if args.rule_state.startswith("lexical2"):
                    info = _fit.lexical_state_rules(
                        model, args.sel_init,
                        decode_state_features=args.rule_state in ("lexical2", "lexical2-letters"))
                    if args.rule_state == "lexical2-s1":
                        coupled = _fit.condition_local_buckets_on_previous_word(model)
                        print(f"[train] S1 state-conditioned hard buckets in "
                              f"phases {coupled['phases']}", flush=True)
                    print(f"[train] lexical state: {info['word_pages']} word "
                          f"routing pages, {info['separator_pages']} separator "
                          "pages; previous-word and current-word integer-rule "
                          f"tracks with {info['feature_phases']} decoded feature phases", flush=True)
                elif args.rule_state in ("reversible", "reversible2"):
                    info = _fit.reversible_state_rules(
                        model, args.sel_init,
                        multiplier=args.reversible_multiplier,
                        tracks=2 if args.rule_state == "reversible2" else 1)
                    print(f"[train] reversible state: {info['tracks']} track(s) "
                          f"of {info['states']} "
                          "distinct integer codes; phase-0 pages permute "
                          "the previous state with multiplier "
                          f"{info['permutation_multiplier']}, phase-1 "
                          "features are fitted",
                          flush=True)
                else:
                    info = _fit.word_state_rules(
                        model, args.sel_init,
                        reset_on_nonword=args.rule_state != "stream")
                    print(f"[train] {args.rule_state} state: phase-0 rules "
                          f"hash (byte, previous state) on "
                          f"{info['word_pages']} pages and reset on "
                          f"{info['reset_pages']} others; phase-1 rules turn "
                          f"the state into fitted features", flush=True)
                if "balance" in args.rule_scoring:
                    _fit.balance_by_simulation(
                        model, train, dev, args.ticks,
                        # All recurrent-state modes simulate long streams.
                        # Keeping every score row can exceed the VRAM cap;
                        # 31 is coprime to 16 phases and samples them all.
                        row_stride=31,
                        log=lambda m: print(m, flush=True))
            elif "balance" in args.rule_scoring:
                _fit.balance_buckets(model, train, dev,
                                     log=lambda m: print(m, flush=True))
            print("[train] rule book: every candidate SETs a work channel to "
                  "its own immediate, chosen by the byte at the cell (its "
                  "page) and a hash of the bytes before it (its scores); "
                  "fitting the immediates and all probe coefficients "
                  "together (fit.fit_readout_and_rules)", flush=True)
            h = _fit.fit_readout_and_rules(model, train, dev, args.ticks,
                                           steps=2 * args.fit_steps,
                                           max_records=args.fit_records,
                                           freeze_phase0_state=args.rule_state in ("word", "word-reserved", "stream", "reversible", "reversible2") or args.rule_state.startswith("lexical2"),
                                           freeze_phase1_state=args.rule_state == "reversible2" or args.rule_state.startswith("lexical2"),
                                           word_start_weight=args.word_start_weight,
                                           log=lambda m: print(m, flush=True))
            if args.choose_channels:
                _fit.choose_channels(model, train, dev, args.ticks,
                                     steps=2 * args.fit_steps,
                                     max_records=args.fit_records,
                                     log=lambda m: print(m, flush=True))
                h = _fit.fit_readout_and_rules(
                    model, train, dev, args.ticks, steps=2 * args.fit_steps,
                    max_records=args.fit_records,
                    freeze_phase0_state=args.rule_state in ("word", "word-reserved", "stream", "reversible", "reversible2") or args.rule_state.startswith("lexical2"),
                    freeze_phase1_state=args.rule_state == "reversible2" or args.rule_state.startswith("lexical2"),
                    log=lambda m: print(m, flush=True))
            print(f"[train] readout + rule features start at {h:.4f} "
                  f"bits/byte on held-out positions", flush=True)
    elif args.init == "tape" and step == 0:
        tape_init(model, args.sel_init)
        print("[train] tape start: bytes write codes under the write head, "
              "rules start as HOLD, probes read the last few bytes", flush=True)
    if args.bias_init == "unigram":
        with torch.no_grad():
            u = unigram_bias(train).to(model.pr_bias)
            have = float(model.pr_bias.max() - model.pr_bias.min())
            need = float(u.max() - u.min())
            # Fresh, or a bias that never really learned: the first real run
            # drifted to a spread of a few units in 246 steps, against the
            # ~1,700 a byte-frequency table needs (a "< 1 unit" test missed
            # it). Whatever small deviations it did learn are kept on top.
            if have < 0.1 * need:
                model.pr_bias.copy_(u + (model.pr_bias - model.pr_bias.mean()))
                print(f"[train] readout biases set from the byte frequencies "
                      f"of the training data (spread {need:.0f} units, was "
                      f"{have:.1f}); the model starts at the letter-frequency "
                      f"level, so everything below it is the field's work",
                      flush=True)
    if step >= args.steps:
        print(f"[train] this run already finished at step {step}. The model "
              f"is {out / 'best.mica'}. Delete {out} to train again.",
              flush=True)
        mark_finished(out, step)
        return 0
    # A different shuffle after a resume, so it does not replay records.
    rng = __import__("random").Random(1 + step)
    vrng = __import__("random").Random(7)

    # =====================================================================
    # Validation that cannot fail silently
    #
    # The first protected run printed "validation skipped (memory)" and never
    # saved a model, because the model is only saved after a validation that
    # works -- and that one message covered two different failures, running
    # out of memory and the loss going NaN. Now: try a small batch, then a
    # batch of one, and say exactly which problem it was.
    # =====================================================================
    def validate():
        if spec.N_SYMBOLS > 258:
            # Compact fit selectors are expanded only in the exported integer
            # model. Score that exact machine here; the soft forward would
            # expand a vocabulary-wide selector tensor on every token.
            from mica_r1 import engine as exact_engine
            m = to_integer(model)
            losses = []
            for record in val:
                session = exact_engine.new_session(m)
                nats = 0.0
                for target in list(record) + [spec.EOS]:
                    scores = exact_engine.probe_scores(m, session).astype(np.float64)
                    scores /= spec.LOGIT_DIVISOR
                    scores[spec.BOS] = -np.inf
                    nats += float(np.logaddexp.reduce(scores) - scores[target])
                    if target != spec.EOS:
                        exact_engine.ingest(m, session, target)
                losses.append(nats / (len(record) + 1))
            v = float(np.mean(losses)) / math.log(2)
            return (v, None) if math.isfinite(v) else (None, "non-finite integer word loss")
        # Validation is forward-only, so it can run at a larger batch than
        # training; the step is paced by the number of GPU operations, not
        # their size. Smaller batches are the fallback if memory runs out.
        sizes = sorted({max(1, min(args.batch, VAL_BATCH)),
                        max(1, min(args.batch, 4)), 1}, reverse=True)
        for vb_size in sizes:
            oom, losses = False, []
            model.eval()
            try:
                with torch.no_grad():
                    losses = [float(model.sequence_loss(vb, vln)[0])
                              for vb, vln in batches(val, vb_size, vrng, dev)]
            except torch.OutOfMemoryError:
                oom = True
            finally:
                model.train()
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            if oom:
                print(f"  validation did not fit at batch {vb_size}; "
                      f"retrying smaller", flush=True)
                continue
            v = float(np.mean(losses)) / math.log(2)
            if not math.isfinite(v):
                return None, ("the validation loss is NaN -- a numerical "
                              "problem, not memory")
            return v, None
        return None, "out of GPU memory even at batch 1"

    # =====================================================================
    # Step aside for games
    #
    # A game and this run both want the whole card; together they overflow
    # it, Windows pages graphics memory into RAM, and the machine stutters or
    # freezes. So: measure how much GPU memory OTHER programs use when
    # training starts, and if that jumps by --yield-gb (a game launching),
    # save, release our cached memory, and wait. When it drops back, carry on.
    # =====================================================================
    limit = None
    if dev.type == "cuda" and not args.no_yield:
        try:
            free, total = torch.cuda.mem_get_info()
            other = (total - free - torch.cuda.memory_reserved()) / 2**30
            if 0 <= other < 0.8 * total / 2**30:
                limit = max(4.0, other + args.yield_gb)
                print(f"[guard] other programs use {other:.1f} GiB of GPU "
                      f"memory now; training pauses itself if that passes "
                      f"{limit:.1f} GiB (a game starting)", flush=True)
            else:
                print(f"[guard] GPU memory reading looks wrong ({other:.1f} "
                      f"GiB used by others); pause-for-games is off",
                      flush=True)
        except Exception as exc:
            print(f"[guard] cannot measure other programs' GPU use ({exc}); "
                  f"pause-for-games is off", flush=True)

    def others_gb():
        free, total = torch.cuda.mem_get_info()
        return (total - free - torch.cuda.memory_reserved()) / 2**30

    def maybe_pause(step, best, hist):
        """Returns a reason to stop, or None to carry on."""
        if limit is None:
            return None
        try:
            o = others_gb()
        except Exception:
            return None
        if o <= limit:
            return None
        save_state(step, best, hist)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n[guard] another program is using {o:.1f} GiB of GPU memory "
              f"(a game?). State saved; training is paused and will carry on "
              f"by itself when it closes.", flush=True)
        t, calm = time.time(), 0
        while calm < 2:
            time.sleep(30)
            rearm()                      # a pause is not a stall
            if STOP_FILE.exists():
                return f"{STOP_FILE.name} found while paused"
            try:
                o = others_gb()
            except Exception:
                o = 0.0
            calm = calm + 1 if o <= limit else 0
        print(f"[guard] GPU free again after {(time.time() - t) / 60:.0f} "
              f"min -- continuing from step {step}", flush=True)
        hb["t"], hb["s"] = time.time(), step   # the pause is not step time
        return None

    t0, bad_steps = time.time(), 0
    # Heartbeat clock: speed is measured between progress lines, with
    # validation and game pauses excluded, so s/step is the real rate.
    hb = {"t": t0, "s": step, "val_s": 0.0}
    # Stall recorder. The 2026-09-23 run went silent after its first
    # validation: no error, no checkpoint, and nothing on disk to say whether
    # it was slow, hung in the GPU driver, or suspended. faulthandler's timer
    # is re-armed at every step; if it ever fires, the stack of every thread
    # lands in stall_trace.txt (it records only -- the run is not touched).
    import faulthandler
    stall_fh = open(out / "stall_trace.txt", "a", encoding="utf-8")

    # Heartbeat for the auto-restart watcher (supervisor.py): rewritten at
    # every step, at the start of every validation and every 30 s during a
    # game pause. If it goes quiet for 45 minutes the watcher treats the run
    # as hung, kills it and restarts it from the last save.
    heartbeat = out / "heartbeat"

    def rearm():
        if args.stall_minutes > 0:
            faulthandler.dump_traceback_later(args.stall_minutes * 60,
                                              repeat=True, file=stall_fh)
        try:
            heartbeat.write_text(f"{step} {int(time.time())}\n")
        except OSError:
            pass
    rearm()
    (out / "FINISHED").unlink(missing_ok=True)    # there is work to do
    print(f"[train] saving resumable state every {args.save_every} steps to "
          f"{RESUME}", flush=True)
    # =====================================================================
    # Fit mode: training by repeated exact fits instead of straight-through
    # steps. Each round draws --fit-records fresh records, runs the machine
    # over them once to see which rule immediate every work probe reads, and
    # refits every probe coefficient, readout bias and rule immediate from
    # where the last round left them. A round digests ~20,000 records in
    # about a minute; a straight-through step digested 32 in ~3.5 s, and
    # made the fitted model WORSE (run 3: 3.60 -> 3.66 in 500 steps).
    # =====================================================================
    if args.mode == "fit":
        from mica_r1 import fit as _fit
        frng = __import__("random").Random(1000 + step)
        while step < args.steps:
            why = should_stop(args.min_free_ram_gb) or \
                maybe_pause(step, best, hist)
            if why:
                print(f"\n[guard] stopping cleanly: {why}", flush=True)
                save_state(step, best, hist)
                print(f"[guard] state saved at round {step}; the next start "
                      f"resumes from here. Best model so far: "
                      f"{out / 'best.mica'}", flush=True)
                return 0
            step += 1
            rearm()
            tr = time.time()
            sample = frng.sample(train, min(args.fit_records, len(train)))
            h = _fit.fit_readout_and_rules(
                model, sample, dev, args.ticks, steps=args.fit_steps,
                max_records=len(sample), lr_scale=args.round_lr,
                freeze_phase0_state=args.rule_state in ("word", "word-reserved", "stream", "reversible", "reversible2") or args.rule_state.startswith("lexical2"),
                freeze_phase1_state=args.rule_state == "reversible2" or args.rule_state.startswith("lexical2"),
                word_start_weight=args.word_start_weight,
                log=lambda m: None)
            rearm()
            v, problem = validate()
            if problem:
                print(f"  round {step}: validation failed -- {problem}",
                      flush=True)
                continue
            rec = {"step": step, "fit_heldout_bits": round(h, 4),
                   "val_bits": round(v, 4),
                   "seconds": round(time.time() - t0, 1)}
            if v < best:
                best = v
                torch.save(model.state_dict(), out / "best.pt")
                if spec.N_SYMBOLS > 258:
                    rec["integer_bits"] = v
                else:
                    try:
                        gap = measure_gap(model, val[:4], device=dev, chunk=1)
                        rec["integer_bits"] = gap["integer_bits"]
                        rec["discretisation_gap"] = gap["discretisation_gap"]
                    except torch.OutOfMemoryError:
                        pass
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                serialize.save(to_integer(model), out / "best.mica")
            hist.append(rec)
            (out / "progress.json").write_text(json.dumps(
                {"steps": args.steps, "mode": "fit",
                 "history": hist[-500:]}, indent=2))
            if step % max(1, args.save_every // 50) == 0 or step == 1:
                save_state(step, best, hist)
            extra = (f"  integer {rec['integer_bits']:.4f} "
                     f"(gap {rec['discretisation_gap']:+.4f})"
                     if "discretisation_gap" in rec else "")
            print(f"  {time.strftime('%H:%M')}  round {step:5d}/{args.steps}"
                  f"  fit {h:.4f}  val {v:.4f}{extra}  "
                  f"({time.time() - tr:.0f}s, {len(sample):,} records)",
                  flush=True)
        faulthandler.cancel_dump_traceback_later()
        save_state(step, best, hist)
        mark_finished(out, step)
        print(f"\n[train] finished: best validation {best:.4f} bits")
        return 0
    while step < args.steps:
        for b, ln in batches(train, args.batch, rng, dev):
            if step >= args.steps:
                break
            why = should_stop(args.min_free_ram_gb) or \
                maybe_pause(step, best, hist)
            if why:
                print(f"\n[guard] stopping cleanly: {why}", flush=True)
                save_state(step, best, hist)
                print(f"[guard] state saved at step {step}; the next start "
                      f"resumes from here. Best model so far: "
                      f"{out / 'best.mica'}", flush=True)
                return 0
            step += 1
            rearm()
            opt.zero_grad(set_to_none=True)
            # Once a smaller batch has been found to fit, use it from the
            # start of every step instead of failing down to it each time.
            b, ln = b[:args.batch], ln[:args.batch]
            loss, used = step_with_retry(b, ln)
            if used != args.batch:
                args.batch = used       # stick with what fits
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                # A non-finite gradient would turn every weight into NaN on
                # the next optimizer step and silently ruin the whole run.
                opt.zero_grad(set_to_none=True)
                bad_steps += 1
                what = "loss" if not torch.isfinite(loss) else "gradient"
                print(f"  step {step}: skipped, non-finite {what} "
                      f"({bad_steps} so far)", flush=True)
                if bad_steps >= 10:
                    print("[guard] ten non-finite steps -- stopping before "
                          "the model is damaged", flush=True)
                    return 1
                continue
            opt.step(); sched.step()
            if args.selector_flips_per_phase and \
                    step % args.selector_project_every == 0:
                from mica_r1.selector_search import project_score_selectors
                moved = project_score_selectors(
                    model, per_phase=args.selector_flips_per_phase,
                    margin=args.selector_project_margin)
                print(f"  step {step}: projected hard rule selectors "
                      f"{moved}", flush=True)

            if step % args.save_every == 0:
                save_state(step, best, hist)

            is_val = step % args.val_every == 0 or step == 1
            if args.log_every and step % args.log_every == 0 and not is_val:
                now = time.time()
                rate = (now - hb["t"]) / max(1, step - hb["s"])
                left = (rate * (args.steps - step) +
                        hb["val_s"] * ((args.steps - step) // args.val_every))
                # What else is running: the same code has measured 8 s/step
                # and 30 s/step on the same card, switching abruptly with no
                # event in this log -- other programs' GPU and RAM use.
                load = f"  ram free {free_ram_gb():.1f}G"
                if dev.type == "cuda":
                    try:
                        load += f"  gpu other {others_gb():.1f}G"
                    except Exception:
                        pass
                print(f"  {time.strftime('%H:%M')}  step {step:6d}/"
                      f"{args.steps}  train {loss.item() / math.log(2):.4f}  "
                      f"grad {float(gn):.3g}  {rate:.0f}s/step  "
                      f"~{left / 3600:.1f}h to go{load}", flush=True)
                hb["t"], hb["s"] = now, step

            if is_val:
                tv = time.time()
                rearm()
                v, problem = validate()
                if problem:
                    print(f"  step {step}: validation failed -- {problem}",
                          flush=True)
                    hb["t"], hb["s"] = time.time(), step
                    continue
                rec = {"step": step,
                       "grad_norm": round(float(gn), 3),
                       "train_bits": round(loss.item() / math.log(2), 4),
                       "val_bits": round(v, 4),
                       "seconds": round(time.time() - t0, 1)}
                if v < best:
                    best = v
                    torch.save(model.state_dict(), out / "best.pt")
                    try:
                        gap = measure_gap(model, val[:4], device=dev, chunk=1)
                        rec["integer_bits"] = gap["integer_bits"]
                        rec["discretisation_gap"] = gap["discretisation_gap"]
                    except torch.OutOfMemoryError:
                        print("  (rounding check did not fit in memory this "
                              "time)", flush=True)
                    gc.collect()
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    serialize.save(to_integer(model), out / "best.mica")
                hist.append(rec)
                (out / "progress.json").write_text(json.dumps(
                    {"steps": args.steps, "history": hist[-200:]}, indent=2))
                extra = (f"  integer {rec['integer_bits']:.4f} "
                         f"(gap {rec['discretisation_gap']:+.4f})"
                         if "integer_bits" in rec else "")
                hb["val_s"] = time.time() - tv
                print(f"  {time.strftime('%H:%M')}  step {step:6d}/"
                      f"{args.steps}  train "
                      f"{rec['train_bits']:.4f}  val {v:.4f}{extra}  "
                      f"grad {rec['grad_norm']:.3g}  {rec['seconds']:.0f}s "
                      f"(validation {hb['val_s'] / 60:.1f} min)", flush=True)
                hb["t"], hb["s"] = time.time(), step

    faulthandler.cancel_dump_traceback_later()
    save_state(step, best, hist)
    mark_finished(out, step)
    print(f"\n[train] finished: best validation {best:.4f} bits")
    print(f"[train] integer model: {out / 'best.mica'} "
          f"({spec.TOTAL_BYTES:,} bytes)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
