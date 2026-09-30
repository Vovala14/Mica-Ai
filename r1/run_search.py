#!/usr/bin/env python3
"""MICA R1 discrete search — multi-restart, GPU-capable.

Section 13 specifies eight independent restarts. They share nothing, so this
runs them *concurrently* in one batch: 8 restarts x 17 models x 32 records =
4,352 sessions stepped together. On a GPU that is far better occupancy than
one restart's 544, which is where most of the speedup comes from.

The four initialisation and routing changes argued for in docs/r1-findings.md
are available as flags. **Every default is R1 exactly as written**, so a run
with no flags is a conformant run and a run with flags records which ones in
its result file.

    --routing pairdiff     page bits from F[c] >= F[c+4]      (finding 5.1)
    --bias-init spread     distinct candidate biases           (finding 5.2)
    --probe-init zero      probe coefficients start at zero    (finding 5.3)
    --probe-init unigram   ...and biases seeded from byte frequencies (5.5)
    --activity-headroom X  cap activity at X times the measured starting
                           activity (finding 5.4, without the wall)

MICA_LOGIT_DIVISOR rescales the probe readout (finding 5.5). It is read at
import, so set it in the environment, not after this module loads.
"""

from __future__ import annotations

import argparse, json, os, platform, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

import numpy as np

from make_records import load_records
from mica_r1 import spec, serialize
from mica_r1.engine import random_model, Model
from mica_r1.search import XorShift32, make_child, SearchConfig
from mica_r1.diagnose import instrument


# Bytes 0..255 and EOS are predictable; BOS is never a target (section 12).
_ELIGIBLE = np.zeros(spec.N_SYMBOLS, bool)
_ELIGIBLE[0:256] = True
_ELIGIBLE[spec.EOS] = True
_ELIG_IDX = np.flatnonzero(_ELIGIBLE)


def unigram_bias(records: list[bytes], alpha: float = 0.5) -> np.ndarray:
    """int16 probe biases equal to LOGIT_DIVISOR * log p(symbol).

    The target set is exactly the one the loss normalises over: every byte of
    every record, plus one EOS per record. Add-alpha smoothing keeps unseen
    bytes finite. Quantisation to int16 costs 0.0003 bits/byte at divisor 16
    and less as the divisor grows, so this is a free head start of
    (uniform - unigram) bits, which on the current corpus is 3.1 bits/byte.
    """
    cnt = np.zeros(spec.N_SYMBOLS, np.float64)
    for r in records:
        if r:
            cnt += np.bincount(np.frombuffer(r, np.uint8),
                               minlength=spec.N_SYMBOLS)
        cnt[spec.EOS] += 1
    w = cnt[_ELIG_IDX] + alpha
    logp = np.log(w / w.sum())
    b = np.zeros(spec.N_SYMBOLS, np.int16)
    # Ineligible symbols (BOS) keep bias 0; they are masked out of the
    # log-sum-exp, so their value never reaches the loss.
    b[_ELIG_IDX] = np.clip(np.rint(logp * spec.LOGIT_DIVISOR),
                           -32768, 32767).astype(np.int16)
    return b


def apply_init_fixes(m: Model, bias_init: str, probe_init: str,
                     uni: np.ndarray | None = None) -> Model:
    """The initialisation changes, applied after section 13's sampling."""
    if bias_init == "spread":
        # Section 13 sets every bias to zero, which makes two in five candidate
        # selections exact ties resolved by lowest index. Distinct biases break
        # every tie at no cost: the format already stores an int16 per candidate.
        ladder = (np.arange(spec.N_CANDIDATES) - spec.N_CANDIDATES // 2)
        m.sc_bias[:, :] = ladder[None, :].astype(np.int16)
    if probe_init in ("zero", "unigram"):
        # Random probe coefficients make the model start worse than uniform and
        # the gap grows with record length. With zeros every symbol scores its
        # bias alone -- and the field reaches the output through nothing else,
        # so this also makes two of section 13's three mutation kinds inert.
        # Measured: of six children with two mutations each, five changed the
        # loss by less than 1e-5 bits on 20 independent batches, and one
        # changed it by exactly zero on all of them. Prefer "unigram-live".
        m.pr_co[:, :] = 0
    if probe_init in ("unigram", "unigram-live"):
        # ...and with the bias set to the corpus log-frequencies, "its bias
        # alone" is the unigram model rather than the uniform one. Without this
        # the search starts 3.1 bits above a distribution it can reach by
        # copying a byte histogram, and every mutation that touches the field
        # is measured against noise.
        if uni is None:
            raise ValueError("probe-init unigram needs the training records")
        m.pr_bias[:] = uni
        # "unigram-live" differs from "unigram" only in what it does NOT do:
        # it leaves section 13's random coefficients alone, so the field
        # reaches the output from round one. That is only survivable at a
        # readout divisor large enough to make the field a perturbation of the
        # histogram rather than a replacement for it -- see MICA_LOGIT_DIVISOR.
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="r1/data/full/train.jsonl")
    ap.add_argument("--val-records", default="r1/data/full/val.jsonl")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=10000)
    ap.add_argument("--batch-records", type=int, default=32)
    ap.add_argument("--record-bytes", type=int, default=256)
    ap.add_argument("--children", type=int, default=16)
    ap.add_argument("--mutations", type=int, default=8)
    ap.add_argument("--validate-every", type=int, default=100)
    ap.add_argument("--val-records-n", type=int, default=1024)
    ap.add_argument("--backend", default="torch", choices=["torch", "numpy"])
    ap.add_argument("--score-backend", default="numpy", choices=["numpy", "c"],
                    help="'c' uses the compiled scorer: identical arithmetic, "
                         "measured ~5x faster on CPU")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--routing", default="r1",
                    choices=["r1", "positive", "pairdiff", "parity"])
    ap.add_argument("--bias-init", default="zero", choices=["zero", "spread"])
    ap.add_argument("--probe-init", default="random",
                    choices=["random", "zero", "unigram",
                             "unigram-live"])
    ap.add_argument("--logit-divisor", type=int, default=0,
                    help="override section 12's divisor of 16. Must be set "
                         "before import, so this flag re-execs; prefer the "
                         "MICA_LOGIT_DIVISOR environment variable.")
    ap.add_argument("--activity-cap", type=float, default=0.0,
                    help="reject a child whose updates/symbol exceeds this. "
                         "An ABSOLUTE ceiling: set it below the model's own "
                         "starting activity and nothing can ever be accepted. "
                         "Prefer --activity-headroom.")
    ap.add_argument("--activity-headroom", type=float, default=0.0,
                    help="cap activity at this multiple of the INCUMBENT's "
                         "measured activity at round 1, e.g. 1.25. Bounds "
                         "growth the way finding 5.4 asks without the risk of "
                         "a ceiling below the floor. Overrides --activity-cap.")
    ap.add_argument("--activity-weight", type=float, default=0.01)
    ap.add_argument("--accept-holdout", type=int, default=0,
                    help="re-test the winning child on this many FRESH records "
                         "before accepting it. 0 is R1 section 13 exactly, "
                         "which proposes and accepts on the same batch and is "
                         "therefore subject to the winner's curse.")
    ap.add_argument("--accept-sigma", type=float, default=0.0,
                    help="with --accept-holdout, require the held-out gain to "
                         "beat this many standard errors, paired across the "
                         "holdout records. The default of 0 is the bare "
                         "'better on the holdout' test. That test has no "
                         "margin, so a child with no real effect passes it "
                         "half the time -- but measured over 50 rounds it did "
                         "not misbehave: all 50 accepted mutations were "
                         "field-side and validation improved with z = 11.8. "
                         "Left off until a run shows it is needed.")
    ap.add_argument("--freeze-instructions", action="store_true",
                    help="section 16 ablation: rewrite programs stay at init")
    ap.add_argument("--randomize-rules", action="store_true",
                    help="section 16 ablation: re-randomise the rule section of "
                         "the final model, keeping injection and probes")
    ap.add_argument("--out", default="r1/runs/search")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    if args.logit_divisor and args.logit_divisor != spec.LOGIT_DIVISOR:
        # spec.LOGIT_DIVISOR is bound at import, and several modules have
        # already captured it. Re-exec with the environment set rather than
        # leave half the process on the old value.
        os.environ["MICA_LOGIT_DIVISOR"] = str(args.logit_divisor)
        os.execv(sys.executable, [sys.executable, *sys.argv])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.backend == "torch":
        import torch
        from mica_r1.torch_batch import TorchModelStack, evaluate, pick_device
        device = pick_device(args.device)
        stack = lambda ms: TorchModelStack(ms, device, routing=args.routing)
        backend_name = f"torch/{device}"
        if device.type == "cuda":
            backend_name += f" ({torch.cuda.get_device_name(0)})"
    else:
        from mica_r1 import engine
        from mica_r1.batch import ModelStack, evaluate
        engine.ROUTING_MODE = args.routing
        stack = lambda ms: ModelStack(ms, score_backend=args.score_backend)
        backend_name = f"numpy/cpu (scoring: {args.score_backend})"

    # ---- data ------------------------------------------------------------
    pool = load_records(args.records)
    if args.record_bytes and args.record_bytes < 256:
        trimmed = []
        for r in pool:
            k = min(len(r), args.record_bytes)
            while k > 0 and k < len(r) and (r[k] & 0xC0) == 0x80:
                k -= 1
            if k:
                trimmed.append(r[:k])
        pool = trimmed
    val_pool = load_records(args.val_records)[:args.val_records_n]

    print(f"[search] backend {backend_name}  routing {args.routing}  "
          f"bias-init {args.bias_init}  probe-init {args.probe_init}  "
          f"logit-divisor {spec.LOGIT_DIVISOR}")
    print(f"[search] {args.restarts} restarts x {args.rounds} rounds, "
          f"{args.batch_records} records/round, {args.children} children")
    print(f"[search] {len(pool):,} train records, {len(val_pool):,} validation records")

    R = args.restarts
    per = 1 + args.children
    rngs = [XorShift32(s + 1) for s in range(R)]
    uni = (unigram_bias(pool)
           if args.probe_init in ("unigram", "unigram-live") else None)
    if uni is not None:
        p = np.exp(uni[_ELIG_IDX].astype(np.float64) / spec.LOGIT_DIVISOR)
        p /= p.sum()
        print(f"[search] probe bias seeded from corpus unigram: "
              f"{-(p * np.log2(p)).sum():.4f} bits/byte at divisor "
              f"{spec.LOGIT_DIVISOR}")
    incumbents = [apply_init_fixes(random_model(s + 1), args.bias_init,
                                   args.probe_init, uni) for s in range(R)]

    # ---- liveness check --------------------------------------------------
    # This exact failure has now happened twice: an initialisation that leaves
    # every probe coefficient at zero disconnects the field from the output,
    # which makes two of section 13's three mutation kinds incapable of
    # changing the loss at all. The search then runs for hours and reports a
    # flat curve, which reads like "hard problem" rather than "broken setup".
    #
    # The test is exact rather than statistical: build children whose mutations
    # are drawn only from kinds 0 and 1 -- signature entries and candidate
    # records. Neither touches the readout, so both reach the loss ONLY through
    # the cell field. If none of them can move it, the field is disconnected,
    # and that is true at any record length and any batch size.
    probe_recs = [pool[i] for i in
                  np.random.default_rng(0).permutation(len(pool))[:16]]
    _rng = XorShift32(999)
    _kids = [make_child(incumbents[0], _rng, 4, kinds=(0, 1)) for _ in range(8)]
    _l, _u, _ = evaluate(stack([incumbents[0]] + _kids), probe_recs)
    # Loss alone, NOT J. A field-only mutation always shifts the activity term
    # slightly -- it changes how many rewrites fire -- so J moves even when the
    # readout is dead. The loss is the part that is provably invariant when
    # every probe coefficient is zero, so it is the part to test.
    _eff = np.abs(_l[1:] - _l[0]) / np.log(2)
    _live = int((_eff > 1e-9).sum())
    print(f"[search] liveness: {_live}/8 field-only children changed the loss "
          f"(max |effect on loss| {_eff.max():.6f} bits)")
    if _live < 2:
        print("[search] *** WARNING: the cell field is not reaching the "
              "output. Mutations to signatures, candidates and rewrite "
              "programs -- two of section 13's three kinds -- cannot change "
              "the loss at all, so this run will learn nothing from them.",
              flush=True)
        print("[search] *** Cause: every probe coefficient is zero, so each "
              "symbol scores its bias alone. Use --probe-init unigram-live "
              "(or random) and set MICA_LOGIT_DIVISOR high enough that the "
              "field perturbs the readout instead of swamping it.", flush=True)

    # ---- activity calibration --------------------------------------------
    # Finding 5.4 asks for a ceiling on rewrites per symbol so the search
    # cannot buy loss reduction with work. An ABSOLUTE ceiling is dangerous:
    # set below the model's own operating point it rejects every child before
    # scoring, and the run reports a flat curve with accept_rate 0.000 and
    # zero holdout rejections -- observed at cap 600 against a model that
    # naturally runs 733. The ceiling must be anchored to something measured.
    act0 = float(_u[0])
    cap = args.activity_cap
    if args.activity_headroom:
        cap = act0 * args.activity_headroom
        print(f"[search] activity: incumbent runs {act0:.0f} updates/symbol; "
              f"cap set to {cap:.0f} ({args.activity_headroom:g}x)")
    elif cap:
        print(f"[search] activity: incumbent runs {act0:.0f} updates/symbol; "
              f"cap {cap:.0f}")
        if cap <= act0:
            print(f"[search] *** WARNING: the activity cap ({cap:.0f}) is at "
                  f"or below the incumbent's own activity ({act0:.0f}). Every "
                  f"child will be rejected before it is scored and NOTHING "
                  f"can be accepted. Use --activity-headroom 1.25 instead, or "
                  f"raise the cap above {act0:.0f}.", flush=True)
    if cap and cap > spec.MAX_UPDATES_PER_SYMBOL:
        cap = 0.0        # a cap above section 10's bound constrains nothing

    orders = []
    for s in range(R):
        o = np.arange(len(pool))
        np.random.default_rng(s + 1).shuffle(o)
        orders.append(o)
    cursors = [0] * R

    history = [[] for _ in range(R)]
    accepted = [0] * R
    rejected_by_holdout = [0] * R
    best_val = [float("inf")] * R
    t0 = time.time()

    for rnd in range(1, args.rounds + 1):
        models, batches = [], []
        for s in range(R):
            take = [orders[s][(cursors[s] + i) % len(orders[s])]
                    for i in range(args.batch_records)]
            cursors[s] += args.batch_records
            batches.append([pool[i] for i in take])
            models.append(incumbents[s])
            for _ in range(args.children):
                models.append(make_child(incumbents[s], rngs[s], args.mutations,
                                         args.freeze_instructions))

        # every restart uses its own record batch, so evaluate restart by
        # restart but keep all of a restart's 17 models in one call
        losses, upds = [], []
        for s in range(R):
            sl = models[s * per:(s + 1) * per]
            l, u, _ = evaluate(stack(sl), batches[s])
            losses.append(l); upds.append(u)

        for s in range(R):
            l, u = losses[s], upds[s]
            J = l + args.activity_weight * u / spec.MAX_UPDATES_PER_SYMBOL
            parent = float(J[0])
            child = J[1:].copy()
            if cap:
                child[u[1:] > cap] = np.inf
            best = int(child.argmin())
            gain = parent - float(child[best])
            ok = gain >= 1e-6 and np.isfinite(child[best])

            if ok and args.accept_holdout:
                # Selecting the best of N children on the same records that
                # proposed them is a biased test: on a flat landscape the winner
                # is usually the luckiest child, not a better model, and
                # accepting it every round walks the incumbent steadily
                # downhill. Confirm on records neither model has been scored on.
                #
                # "Better on the holdout" is not enough. That comparison has no
                # margin, so a child with no real effect passes it half the
                # time, and at 16 children per round a winner is found nearly
                # every round -- measured accept_rate 0.47 with validation
                # flat. Require the held-out improvement to be significant
                # against its own sampling error instead, paired across
                # records because both models see the same ones.
                hold = [pool[orders[s][(cursors[s] + i) % len(orders[s])]]
                        for i in range(args.accept_holdout)]
                cursors[s] += args.accept_holdout
                cand = models[s * per + 1 + best]
                hl, hu, _, pr_l, pr_u = evaluate(
                    stack([incumbents[s], cand]), hold, per_record=True)
                hJ = hl + args.activity_weight * hu / spec.MAX_UPDATES_PER_SYMBOL
                if args.accept_sigma > 0:
                    pj = pr_l + args.activity_weight * pr_u / spec.MAX_UPDATES_PER_SYMBOL
                    d = pj[1] - pj[0]                  # per record, child minus parent
                    n = d.size
                    sem = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.inf
                    # Accept only if the whole confidence interval is below 0.
                    ok = bool(d.mean() + args.accept_sigma * sem < 0.0)
                else:
                    ok = float(hJ[1]) < float(hJ[0]) - 1e-9
                rejected_by_holdout[s] += (not ok)

            if ok:
                incumbents[s] = models[s * per + 1 + best]
                accepted[s] += 1
                cur = (float(child[best]), float(l[best + 1]), float(u[best + 1]))
            else:
                cur = (parent, float(l[0]), float(u[0]))
            rec = {"round": rnd, "J": round(cur[0], 6),
                   "bits_per_target": round(cur[1] / np.log(2), 4),
                   "updates_per_symbol": round(cur[2], 1),
                   "accept_rate": round(accepted[s] / rnd, 3),
                   "holdout_rejections": rejected_by_holdout[s],
                   "seconds": round(time.time() - t0, 1)}
            history[s].append(rec)

        if args.validate_every and rnd % args.validate_every == 0:
            for s in range(R):
                vl, _vu, _ = evaluate(stack([incumbents[s]]), val_pool)
                v = float(vl[0]) / np.log(2)
                history[s][-1]["val_bits_per_target"] = round(v, 4)
                if v < best_val[s]:
                    best_val[s] = v
                    serialize.save(incumbents[s], out / f"restart{s+1}_best.mica")

        if args.log_every and (rnd % args.log_every == 0 or rnd == 1):
            el = time.time() - t0
            bpt = [h[-1]["bits_per_target"] for h in history]
            acc = np.mean([h[-1]["accept_rate"] for h in history])
            vs = [h[-1].get("val_bits_per_target") for h in history]
            vtxt = (f"  val best {min(x for x in vs if x is not None):.4f}"
                    if any(v is not None for v in vs) else "")
            print(f"  round {rnd:6d}/{args.rounds}  train bpt "
                  f"min {min(bpt):.4f} med {np.median(bpt):.4f}{vtxt}  "
                  f"acc {acc:.2f}  {el:.0f}s  ({el/rnd:.2f}s/round)", flush=True)
            (out / "progress.json").write_text(json.dumps({
                "round": rnd, "seconds": round(el, 1),
                "seconds_per_round": round(el / rnd, 3),
                "restarts": [h[-1] for h in history]}, indent=2))

    if args.randomize_rules:
        # "If randomized rules plus trained readout perform equally well, the
        #  learned rewrite core has not earned the claimed contribution."
        from mica_r1.engine import random_model as _rm
        for s_ in range(R):
            fresh = _rm(1000 + s_)
            for f in ("sc_nb", "sc_ch", "sc_co", "sc_bias", "op_code", "op_d",
                      "op_n", "op_c", "op_a", "op_b", "op_u"):
                setattr(incumbents[s_], f, getattr(fresh, f))

    # ---- select and report ------------------------------------------------
    finals = []
    for s in range(R):
        vl, vu, _ = evaluate(stack([incumbents[s]]), val_pool)
        finals.append((float(vl[0]) / np.log(2), float(vu[0]), s))
    finals.sort(key=lambda x: (x[0], x[1], x[2]))     # loss, then fewer updates, then seed
    v_best, u_best, s_best = finals[0]
    serialize.save(incumbents[s_best], out / "selected.mica")

    result = {
        "spec": "MICA R1 section 13 search",
        "backend": backend_name,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "settings": {k: v for k, v in vars(args).items()},
        "activity": {"initial_updates_per_symbol": round(act0, 1),
                     "cap_applied": round(cap, 1)},
        # Not a field of the 86,820-byte file: a decoder needs this value the
        # same way it needs the tick cap and the neighbour offsets.
        "logit_divisor": spec.LOGIT_DIVISOR,
        "conformant": (args.routing == "r1" and args.bias_init == "zero"
                       and not args.accept_holdout
                       and args.probe_init == "random" and not args.activity_cap
                       and not args.activity_headroom
                       and spec.LOGIT_DIVISOR == 16
                       and not args.freeze_instructions
                       and not args.randomize_rules
                       and os.environ.get("MICA_NO_PHASE_ROUTING", "0") != "1"
                       and os.environ.get("MICA_FULL_FIELD", "0") != "1"
                       and not os.environ.get("MICA_OFFSETS")),
        "ablations": {
            "no_phase_routing": os.environ.get("MICA_NO_PHASE_ROUTING", "0") == "1",
            "full_field_updates": os.environ.get("MICA_FULL_FIELD", "0") == "1",
            "frozen_instructions": bool(args.freeze_instructions),
            "offsets_override": os.environ.get("MICA_OFFSETS", ""),
            "randomized_rules": bool(args.randomize_rules)},
        "wall_seconds": round(time.time() - t0, 1),
        "seconds_per_round": round((time.time() - t0) / args.rounds, 3),
        "selected": {"restart": s_best + 1, "val_bits_per_target": round(v_best, 4),
                     "updates_per_symbol": round(u_best, 1)},
        "all_restarts": [{"restart": s + 1, "val_bits_per_target": round(v, 4)}
                         for v, _u, s in finals],
        "instruments": instrument(incumbents[s_best], val_pool[:8]),
        "history": {f"restart{s+1}": history[s] for s in range(R)},
    }
    (out / "search_result.json").write_text(json.dumps(result, indent=2))
    print(f"\n[search] selected restart {s_best+1}: {v_best:.4f} bits/target")
    gate_file = Path("r1/runs/gate.json")
    if gate_file.exists():
        g = json.loads(gate_file.read_text())
        print(f"[search] section 15 first gate: {g['gate_bits_per_byte']} bits/byte "
              f"(5% below the storage-matched {g['strongest_baseline']['model']})")
    bl = Path("r1/runs/baselines.json")
    if bl.exists():
        rows = json.loads(bl.read_text())["baselines"]
        best = min(rows, key=lambda r: r["bits_per_byte"])
        print(f"[search] strongest learned baseline at the same cap: "
              f"{best['model']} {best['bits_per_byte']} bits/byte")
    print(f"[search] wrote {out}/search_result.json and selected.mica")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
