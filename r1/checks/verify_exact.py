#!/usr/bin/env python3
"""Does the training implementation compute exactly what the exported integer
file computes?  Measured, not rounded.

For one run (best.mica + run_info.json, and best.pt when present) and a fixed
set of records, every implementation of the machine runs on the same bytes and
is compared position by position and tick by tick:

  soft      mica_r1/soft.py, the training model, hard forward, on the chosen
            device. With best.pt: the TRAINED parameters exactly as saved.
            Without it: parameters rebuilt from the file (fit.from_integer).
  soft-legacy  the same with the winner picked by a plain argmax, as soft.py
            did before 2026-09-25
  numpy     mica_r1/batch.py, the batched integer evaluator, numpy scorer
  c         mica_r1/batch.py with the compiled scorer (what evaluations use)
  engine    mica_r1/engine.py, the single-session reference engine
  fit       fit._simulate, the integer machine the fitting code runs

Reported: the largest absolute difference of any output score from the numpy
integer scores (integer units; 1 unit = 1/1024 of a logit), the loss to 1e-9,
rule winners / pages / top scores that differ (in total, and among ticks where
two or more candidates tie), and the FIRST mismatch in detail. With best.pt,
every parameter of the trained model is compared with the file field by field
as the training forward pass reads it, and the export is re-done and compared
byte for byte. A second model with FORCED ties (candidate c+C/2 scores exactly
like candidate c on every page) exercises the tie rule on every tick. Finally
the trainer's own validation protocol is re-run, so a printed "val" can be
traced to its cause.

The driver streams each worker's output as it comes, touches the watcher's
heartbeat, and stops a worker that runs past --timeout minutes.

    python r1/checks/verify_exact.py --run p16_self1=r1/runs/sweep/p16_self1 \
        --set val64 --n 64 --out r1/runs/checks/verify.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import utf8_stdout, record_set, run_geometry, worker_env, beat, hard_exit


def _pad(recs):
    import numpy as np
    L = max(len(r) for r in recs)
    padded = np.full((len(recs), L), -1, np.int64)
    for j, r in enumerate(recs):
        padded[j, :len(r)] = np.frombuffer(r, np.uint8)
    return padded, np.array([len(r) for r in recs]), L


def soft_run(sm, recs, dev, legacy: bool):
    """Scores (L+1,B,V) in integer units, and per tick: winner, top score,
    number of candidates at the top, page -- each (ticks_total, B)."""
    import numpy as np
    import torch
    from mica_r1 import soft, spec
    padded, lengths, L = _pad(recs)
    B = len(recs)
    bt = torch.as_tensor(np.maximum(padded, 0), device=dev)
    ln = torch.as_tensor(lengths, device=dev)
    out = np.zeros((L + 1, B, spec.N_SYMBOLS))
    soft.TRACE, soft.LEGACY_ARGMAX = [], legacy
    try:
        with torch.no_grad():
            st = sm.init_state(B, dev)
            rp = sm.rule_params()
            st = sm.ingest(st, torch.full((B,), spec.BOS, dtype=torch.long,
                                          device=dev), rp)
            pr = sm.probe_params()
            for t in range(L + 1):
                # logits() divides by the divisor, a power of two: exact
                out[t] = (sm.logits(st, pr) * spec.LOGIT_DIVISOR).double().cpu().numpy()
                if t >= L:
                    break
                feed = torch.where(t < ln, bt[:, t],
                                   torch.full_like(ln, spec.BOS))
                st = sm.ingest(st, feed, rp)
        tr = soft.TRACE
    finally:
        soft.TRACE, soft.LEGACY_ARGMAX = None, False
    cols = [torch.stack([x[k] for x in tr]).numpy() for k in range(4)]
    return out, cols[0], cols[1], cols[2], cols[3]


def int_run(m, recs, backend: str):
    import numpy as np
    from mica_r1 import batch, spec
    padded, lengths, L = _pad(recs)
    n = len(recs)
    ms = batch.ModelStack([m], score_backend=backend)
    st = batch.BatchSession(n, np.zeros(n, np.int32))
    out = np.zeros((L + 1, n, spec.N_SYMBOLS))
    batch.TRACE = []
    try:
        batch.ingest_batch(ms, st, np.full(n, spec.BOS, np.int32))
        for t in range(L + 1):
            out[t] = batch.probe_batch(ms, st)
            if t >= L:
                break
            feed = np.where(t < lengths, padded[:, t], spec.BOS)
            batch.ingest_batch(ms, st, feed.astype(np.int32))
        tr = batch.TRACE
    finally:
        batch.TRACE = None
    for e in tr:        # one window cell per session, sessions in order
        assert (e[1] == np.arange(n)).all(), "unexpected tick layout"
    win = np.stack([e[4] for e in tr])
    page = np.stack([e[3] for e in tr])
    top = np.stack([e[5] for e in tr]) if tr[0][5] is not None else None
    ntop = np.stack([e[6] for e in tr]) if tr[0][6] is not None else None
    return out, win, top, ntop, page


def engine_run(m, recs):
    import numpy as np
    from mica_r1 import engine, spec
    padded, lengths, L = _pad(recs)
    out = np.zeros((L + 1, len(recs), spec.N_SYMBOLS))
    for j, r in enumerate(recs):
        s = engine.new_session(m)            # zeroes, then ingests BOS once
        for t in range(L + 1):
            out[t, j] = engine.probe_scores(m, s)
            if t >= L:
                break
            engine.ingest(m, s, int(padded[j, t]) if t < lengths[j] else spec.BOS)
    return out


def fit_run(sm, recs, dev):
    import numpy as np
    from mica_r1 import fit, spec
    rows = []

    def on_tick(page, sc, win, ph):
        top = sc.max(-1).values
        rows.append((win.cpu().numpy(), top.cpu().numpy(),
                     (sc == top[:, None]).sum(-1).cpu().numpy(),
                     page.cpu().numpy()))
    fit._simulate(sm, recs, dev, spec.MAX_TICKS, batch=len(recs), on_tick=on_tick)
    return tuple(np.stack([r[k] for r in rows]) for k in range(4))


def nats(scores, recs):
    """Per-position nats of the true target from integer-unit scores."""
    import numpy as np
    from mica_r1 import spec
    padded, lengths, L = _pad(recs)
    elig = np.zeros(spec.N_SYMBOLS, bool)
    elig[:256] = True
    elig[spec.EOS] = True
    x = scores[:, :, elig] / spec.LOGIT_DIVISOR                  # (L+1,B,257)
    mx = x.max(-1, keepdims=True)
    lse = mx[..., 0] + np.log(np.exp(x - mx).sum(-1))
    tgt = np.where(np.arange(L + 1)[:, None] < lengths[None, :],
                   np.concatenate([padded.T, np.full((1, len(recs)), -1)])[:L + 1],
                   spec.EOS)
    tcol = np.where(tgt == spec.EOS, 256, tgt)                   # column in x
    tl = np.take_along_axis(x, tcol[..., None], -1)[..., 0]
    alive = np.arange(L + 1)[:, None] <= lengths[None, :]
    return np.where(alive, lse - tl, 0.0), alive


def first_mismatch(win, top, page, ref, ticks, chunk_start):
    """Where the first tick disagreeing with the numpy reference is, and what
    both sides had there."""
    import numpy as np
    rw, rt, rn, rp = ref
    bad = (win != rw) | (page != rp)
    if top is not None:
        bad |= top != rt
    if not bad.any():
        return None
    k, row = [int(v) for v in np.argwhere(bad)[0]]
    return {"record": chunk_start + row, "position": k // ticks,
            "tick": k % ticks,
            "this": {"page": int(page[k, row]), "winner": int(win[k, row]),
                     "top_score": None if top is None else float(top[k, row])},
            "numpy": {"page": int(rp[k, row]), "winner": int(rw[k, row]),
                      "top_score": float(rt[k, row]),
                      "candidates_at_top": int(rn[k, row])}}


def compare(scores, win, top, page, ref_scores, ref, recs, alive, ref_nats,
            ticks, chunk_start):
    import numpy as np
    d = np.abs(scores - ref_scores)
    rec = {"max_abs_score_diff_units": float(d.max()),
           "max_abs_score_diff_units_alive":
               float(d[alive].max()) if alive.any() else 0.0,
           "positions_with_any_score_diff": int((d.max(-1) > 0)[alive].sum())}
    n_, _ = nats(scores, recs)
    rec["max_abs_nats_diff_per_position"] = float(np.abs(n_ - ref_nats).max())
    rw, rt, rn, rp = ref
    tie = rn >= 2
    mism = win != rw
    rec.update({"tick_rows": int(win.size),
                "winner_mismatches": int(mism.sum()),
                "page_mismatches": int((page != rp).sum()),
                "tie_rows": int(tie.sum()),
                "winner_mismatches_on_ties": int((mism & tie).sum())})
    if top is not None:
        rec["top_score_mismatches"] = int((top != rt).sum())
        rec["max_abs_top_score_diff"] = float(np.abs(top - rt).max())
    fm = first_mismatch(win, top, page, ref, ticks, chunk_start)
    if fm:
        rec["first_mismatch"] = fm
    return rec, float(n_.sum())


def _merge(acc: dict, r: dict) -> None:
    """Fold one chunk's comparison into the running totals."""
    for k, v in r.items():
        if k == "first_mismatch":
            acc.setdefault(k, v)
        elif k.startswith("max_"):
            acc[k] = max(acc.get(k, 0.0), v)
        else:
            acc[k] = acc.get(k, 0) + v


def param_diff(sm, m) -> dict:
    """Every parameter as the training forward pass reads it (hard selector
    choices, rounded values) against the integer file, field by field."""
    import numpy as np
    import torch
    from mica_r1 import soft, spec
    out = {}

    def cmp(name, a, b):
        a = np.asarray(a).astype(np.int64)
        b = np.asarray(b).astype(np.int64)
        if a.shape != b.shape:
            out[name] = {"shape_soft": list(a.shape), "shape_file": list(b.shape)}
            return
        d = a != b
        r = {"entries": int(a.size), "differ": int(d.sum())}
        if d.any():
            i = tuple(int(x) for x in np.argwhere(d)[0])
            r["first"] = {"index": list(i), "soft": int(a[i]), "file": int(b[i])}
        out[name] = r

    with torch.no_grad():
        rp = sm.rule_params()
        idx = lambda t: t.argmax(-1).cpu().numpy()
        val = lambda t: t.double().cpu().numpy()
        cmp("sc_nb", idx(rp["nb"]), m.sc_nb)
        cmp("sc_ch", idx(rp["ch"]), m.sc_ch)
        cmp("sc_co", val(rp["co"]), m.sc_co)
        cmp("sc_bias", val(rp["bias"]), m.sc_bias)
        parts = torch.split(rp["ops"], soft.OPERAND_WIDTHS, dim=-1)
        cmp("op_code", idx(parts[0]), m.op_code)
        cmp("op_d", idx(parts[1]), m.op_d)
        cmp("op_n", idx(parts[2]), m.op_n)
        cmp("op_c", idx(parts[3]), m.op_c)
        cmp("op_u", idx(parts[4]), m.op_u)
        cmp("op_a", val(parts[5])[..., 0], m.op_a)
        cmp("op_b", val(parts[6])[..., 0], m.op_b)
        if len(parts) > 7:
            cmp("op_v", val(parts[7]), m.op_v)
        cmp("inj_cell", idx(rp["inj_cell"]), m.inj_cell)
        cmp("inj_chan", idx(rp["inj_chan"]), m.inj_chan)
        cmp("inj_delta", val(rp["inj_delta"]), m.inj_delta)
        del rp
        cell, chan, co, bias = sm.probe_params()
        lo = spec.N_CELLS - cell.shape[-1]
        cmp("pr_cell", idx(cell) + lo, m.pr_cell)
        cmp("pr_chan", idx(chan), m.pr_chan)
        cmp("pr_co", val(co), m.pr_co)
        cmp("pr_bias", val(bias), m.pr_bias)
        # values the forward pass cannot represent exactly
        bad = {}
        for name, p in sm.named_parameters():
            nf = int((~torch.isfinite(p)).sum())
            if nf:
                bad[name] = nf
        out["_non_finite_parameters"] = bad
    return out


def trainer_validation(sm, recs, dev, int_rec_nats, log, size: int = 16):
    """The number the trainer prints as "val": train_soft.validate() runs
    sequence_loss on length-sorted blocks of `size` records and averages the
    blocks' mean-of-record losses. Done here with the soft model (current and
    pre-fix argmax) and, on the same blocks, with the integer per-record
    losses."""
    import torch
    from mica_r1 import soft
    order = sorted(range(len(recs)), key=lambda i: len(recs[i]))
    blocks = [order[i:i + size] for i in range(0, len(order), size)]
    out = {"block_size": size,
           "integer": sum(sum(int_rec_nats[i] for i in bl) / len(bl)
                          for bl in blocks) / len(blocks) / math.log(2)}
    for legacy in (False, True):
        soft.LEGACY_ARGMAX = legacy
        try:
            losses = []
            for bl in blocks:
                rs = [recs[i] for i in bl]
                L = max(len(r) for r in rs)
                b = torch.zeros(len(rs), L, dtype=torch.long, device=dev)
                for j, r in enumerate(rs):
                    b[j, :len(r)] = torch.tensor(list(r), device=dev)
                ln = torch.tensor([len(r) for r in rs], device=dev)
                with torch.no_grad():
                    losses.append(float(sm.sequence_loss(b, ln)[0]))
        finally:
            soft.LEGACY_ARGMAX = False
        key = "soft_legacy_argmax" if legacy else "soft"
        out[key] = sum(losses) / len(losses) / math.log(2)
        log(f"trainer-style validation, {key}: {out[key]:.6f} "
            f"(integer {out['integer']:.6f})")
    return out


def forced_tie_model(m):
    """Candidate c + C/2 scores exactly like candidate c on every page (same
    terms, same bias) but keeps its own program, so whenever either would
    win, the two tie, and only the lowest-index rule picks c."""
    import copy
    from mica_r1 import spec
    mt = copy.deepcopy(m)
    h = spec.N_CANDIDATES // 2
    for f in ("sc_nb", "sc_ch", "sc_co", "sc_bias"):
        a = getattr(mt, f)
        a[:, h:2 * h] = a[:, :h]
    return mt


def worker(a) -> int:
    import numpy as np
    import torch
    from mica_r1 import spec, serialize, batch, engine, fit
    from mica_r1.soft import SoftMica
    from mica_r1.discretise import to_integer
    assert engine.ROUTING_MODE == "pairdiff", engine.ROUTING_MODE
    torch.set_num_threads(max(1, a.threads))
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                       if a.device == "auto" else a.device)
    batch.MAX_TICKS = spec.MAX_TICKS
    T = spec.MAX_TICKS
    m = serialize.load(a.mica)
    recs = record_set(a.set, a.data_root)[:a.n]
    rep = {"mica": a.mica, "device": str(dev), "set": a.set,
           "records": len(recs), "chunk": a.chunk, "ticks": T,
           "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "mica_md5": hashlib.md5(Path(a.mica).read_bytes()).hexdigest(),
           "checks": {}}
    t0 = time.time()

    def log(msg):
        beat(f"verify {msg}")
        print(f"[verify] {msg}  ({time.time() - t0:.0f}s)", flush=True)

    def save():
        rep["seconds"] = round(time.time() - t0, 1)
        Path(a.out).write_text(json.dumps(rep, indent=1))

    if a.pt and Path(a.pt).exists():
        sm = SoftMica(ticks=T, tau=0.5, hard=True, sel_init=0.05, sel_tau=1.0)
        sd = torch.load(a.pt, map_location="cpu", mmap=True)
        sm.load_state_dict(sd)
        del sd
        rep["soft_source"] = "trained parameters (best.pt)"
        with tempfile.TemporaryDirectory() as td:
            serialize.save(to_integer(sm), Path(td) / "x.mica")
            b = (Path(td) / "x.mica").read_bytes()
        ref = Path(a.mica).read_bytes()
        n_ = min(len(b), len(ref))
        rep["export_identical_bytes"] = b == ref
        rep["export_differing_bytes"] = int(
            np.count_nonzero(np.frombuffer(b, np.uint8)[:n_] !=
                             np.frombuffer(ref, np.uint8)[:n_])
            + abs(len(b) - len(ref)))
        log(f"export: best.pt -> to_integer -> bytes identical to best.mica: "
            f"{rep['export_identical_bytes']} ({rep['export_differing_bytes']} "
            f"differing bytes)")
        sm = sm.to(dev).eval()
        rep["param_diff"] = param_diff(sm, m)
        for k, v in rep["param_diff"].items():
            if k.startswith("_"):
                log(f"parameters: non-finite values: {v or 'none'}")
            elif "differ" not in v or v["differ"]:
                log(f"parameters: {k} DIFFERS: {v}")
        n_diff = sum(v.get("differ", 1) for k, v in rep["param_diff"].items()
                     if not k.startswith("_"))
        log(f"parameters: {n_diff} entries of the trained model differ from "
            f"the file, as the training forward pass reads them")
        save()
    else:
        sm = fit.from_integer(m, ticks=T).to(dev).eval()
        rep["soft_source"] = "rebuilt from the file (fit.from_integer)"

    for label in ("model", "forced_ties"):
        if label == "forced_ties":
            model_i = forced_tie_model(m)
            del sm
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            sm = fit.from_integer(model_i, ticks=T).to(dev).eval()
        else:
            model_i = m
        impls = ("c", "soft", "soft_legacy_argmax", "fit_simulate")
        res = {k: {} for k in ("numpy",) + impls}
        nats_tot = {k: 0.0 for k in res}
        n_pos = 0
        rec_nats = []
        for ci in range(0, len(recs), a.chunk):
            part = recs[ci:ci + a.chunk]
            ref_s, rw, rt, rn, rpg = int_run(model_i, part, "numpy")
            ref = (rw, rt, rn, rpg)
            ref_n, alive = nats(ref_s, part)
            n_pos += int(alive.sum())
            nats_tot["numpy"] += float(ref_n.sum())
            rec_nats += [float(ref_n[:, j].sum()) / (len(r) + 1)
                         for j, r in enumerate(part)]
            _merge(res["numpy"], {"tick_rows": int(rw.size),
                                  "tie_rows": int((rn >= 2).sum()),
                                  "max_candidates_tied": int(rn.max())})
            s_, w_, _, _, pg_ = int_run(model_i, part, "c")
            r, nt = compare(s_, w_, None, pg_, ref_s, ref, part, alive, ref_n,
                            T, ci)
            nats_tot["c"] += nt
            _merge(res["c"], r)
            for legacy in (False, True):
                key = "soft_legacy_argmax" if legacy else "soft"
                s_, w_, top_, _, pg_ = soft_run(sm, part, dev, legacy)
                r, nt = compare(s_, w_, top_, pg_, ref_s, ref, part, alive,
                                ref_n, T, ci)
                nats_tot[key] += nt
                _merge(res[key], r)
            fw, ftop, _, fpg = fit_run(sm, part, dev)
            r = {"tick_rows": int(fw.size),
                 "winner_mismatches": int((fw != rw).sum()),
                 "page_mismatches": int((fpg != rpg).sum()),
                 "winner_mismatches_on_ties": int(((fw != rw) & (rn >= 2)).sum()),
                 "max_abs_top_score_diff": float(np.abs(ftop - rt).max())}
            fm = first_mismatch(fw, ftop, fpg, ref, T, ci)
            if fm:
                r["first_mismatch"] = fm
            _merge(res["fit_simulate"], r)
            if label == "model" and ci == 0 and a.n_engine:
                k = min(a.n_engine, len(part))
                s_ = engine_run(model_i, part[:k])
                d = np.abs(s_ - ref_s[:, :k])
                res["engine"] = {"records": k,
                                 "max_abs_score_diff_units": float(d.max()),
                                 "max_abs_score_diff_units_alive":
                                     float(d[alive[:, :k]].max())}
            log(f"{label}: records {ci + len(part)}/{len(recs)}: soft winners "
                f"differing {res['soft'].get('winner_mismatches', 0)}, pages "
                f"{res['soft'].get('page_mismatches', 0)}, max score diff "
                f"{res['soft'].get('max_abs_score_diff_units', 0)}; legacy "
                f"winners {res['soft_legacy_argmax'].get('winner_mismatches', 0)}")
            if res["soft"].get("first_mismatch") and ci == 0:
                log(f"{label}: first soft mismatch: "
                    f"{json.dumps(res['soft']['first_mismatch'])}")
        for k in nats_tot:
            res[k]["positions"] = n_pos
            if k != "fit_simulate":
                res[k]["bits_per_target"] = nats_tot[k] / n_pos / math.log(2)
                res[k]["bits_diff_vs_numpy"] = \
                    (nats_tot[k] - nats_tot["numpy"]) / n_pos / math.log(2)
        rep["checks"][label] = res
        save()
        if label == "model":
            res["trainer_validation"] = trainer_validation(sm, recs, dev,
                                                           rec_nats, log)
            save()
    save()
    log("done")
    hard_exit(0)


def run_worker(cmd, env, name, timeout_s):
    """Run one worker, streaming its output; touch the heartbeat meanwhile;
    stop it after timeout_s. Returns (returncode or None on timeout)."""
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")

    def pump():
        for ln in p.stdout:
            print(f"  {name}: {ln.rstrip()}", flush=True)
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    t0 = time.time()
    while True:
        try:
            rc = p.wait(timeout=60)
            th.join(timeout=10)
            return rc
        except subprocess.TimeoutExpired:
            beat(f"verify {name} running {int(time.time() - t0)}s")
            if time.time() - t0 > timeout_s:
                p.kill()
                th.join(timeout=10)
                return None


def main() -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[],
                    help="name=run_dir (best.mica, run_info.json, best.pt)")
    ap.add_argument("--set", default="val64")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--n-engine", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=16,
                    help="records per pass (the soft readout's memory grows "
                         "with it)")
    ap.add_argument("--timeout", type=float, default=30,
                    help="minutes before a worker is stopped")
    ap.add_argument("--no-pt", action="store_true",
                    help="rebuild the soft model from the file even if best.pt exists")
    ap.add_argument("--out", required=False)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mica")
    ap.add_argument("--pt")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for item in a.run:
        name, run_dir = item.split("=", 1)
        run_dir = Path(run_dir)
        tmp = out_path.with_name(out_path.stem + f".{name}.part.json")
        cmd = [sys.executable, __file__, "--worker",
               "--mica", str(run_dir / "best.mica"), "--set", a.set,
               "--n", str(a.n), "--n-engine", str(a.n_engine),
               "--device", a.device, "--threads", str(a.threads),
               "--chunk", str(a.chunk), "--out", str(tmp)]
        if not a.no_pt:
            cmd += ["--pt", str(run_dir / "best.pt")]
        if a.data_root:
            cmd += ["--data-root", a.data_root]
        rc = run_worker(cmd, worker_env(run_geometry(run_dir)), name,
                        a.timeout * 60)
        r = json.loads(tmp.read_text()) if tmp.exists() else {}
        if rc != 0:
            r["worker_status"] = "timeout" if rc is None else f"exit {rc}"
            print(f"[verify] {name}: worker {r['worker_status']}; partial "
                  f"results kept", flush=True)
        if tmp.exists():
            tmp.unlink()
        results[name] = r
        out_path.write_text(json.dumps(results, indent=1))
        for label, res in r.get("checks", {}).items():
            for impl, v in res.items():
                if impl == "trainer_validation":
                    continue
                if impl == "numpy":
                    print(f"[verify] {name} {label:11s} numpy: "
                          f"{v['bits_per_target']:.9f} bits, "
                          f"{v['tie_rows']:,}/{v['tick_rows']:,} tick rows tied",
                          flush=True)
                    continue
                keys = ("max_abs_score_diff_units", "bits_diff_vs_numpy",
                        "winner_mismatches", "page_mismatches",
                        "winner_mismatches_on_ties", "max_abs_top_score_diff")
                print(f"[verify] {name} {label:11s} {impl:19s} " +
                      "  ".join(f"{k}={v[k]}" for k in keys if k in v),
                      flush=True)
        tv = r.get("checks", {}).get("model", {}).get("trainer_validation")
        if tv:
            print(f"[verify] {name} trainer-style val (blocks of "
                  f"{tv['block_size']}): soft {tv['soft']:.6f}, soft with the "
                  f"pre-fix argmax {tv['soft_legacy_argmax']:.6f}, integer "
                  f"{tv['integer']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
