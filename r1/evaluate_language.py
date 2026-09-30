#!/usr/bin/env python3
"""Section 15 language evaluation, beyond bits per byte.

  "Measure raw text bits per byte, EOS accuracy, short completion preference,
   exact delayed copying at 16,64,256,1024-byte distances, and repetition in
   512-byte greedy continuations."

Bits per byte lives in gate.py. This covers the rest that can be automated —
short completion preference needs human raters and is deliberately absent
rather than faked with a proxy.

Delayed copying is measured two ways, because exact match is all-or-nothing and
tells you nothing about a model that is merely *better* on a repeat:

  exact   — greedy decoding reproduces the span byte for byte
  graded  — bits per byte on the second occurrence versus the first. A model
            that remembers anything at distance D scores lower on the repeat;
            the difference is the retained information, in bits.
"""

from __future__ import annotations

import argparse, json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

import numpy as np
from make_records import load_records
from mica_r1 import spec, serialize
from mica_r1.engine import Session, ingest, probe_scores
from mica_r1.decode import generate, Utf8State, advance, eligible_mask
from mica_r1.spec import BOS, EOS, N_SYMBOLS

ELIG = np.zeros(N_SYMBOLS, bool); ELIG[0:256] = True; ELIG[EOS] = True
EIDX = np.flatnonzero(ELIG)


def nats_for(model, s: Session, targets: bytes) -> list[float]:
    """Per-byte negative log probability, ingesting as it goes."""
    out = []
    for b in targets:
        sc = probe_scores(model, s).astype(np.float64) / spec.LOGIT_DIVISOR
        e = sc[EIDX]
        m = e.max()
        lse = m + np.log(np.exp(e - m).sum())
        out.append(float(lse - sc[b]))
        ingest(model, s, b)
    return out


# ---------------------------------------------------------------------------
def delayed_copy(model, filler_pool: list[bytes], distances=(16, 64, 256, 1024),
                 span=12, trials=8, seed=0) -> dict:
    """Show a span, wait D bytes, show it again, and measure what was retained."""
    rng = random.Random(seed)
    filler = b"".join(filler_pool)
    res = {}
    for D in distances:
        first_bits, second_bits, exact = [], [], 0
        for _ in range(trials):
            i = rng.randrange(0, max(1, len(filler) - span - D - 8))
            payload = filler[i:i + span]
            gap = filler[i + span:i + span + D]

            s = Session(); ingest(model, s, BOS)
            b1 = nats_for(model, s, payload)          # first occurrence
            nats_for(model, s, gap)                   # the wait
            b2 = nats_for(model, s, payload)          # the repeat
            first_bits.append(sum(b1) / span / math.log(2))
            second_bits.append(sum(b2) / span / math.log(2))

            # exact greedy reproduction of the span after the gap
            s2 = Session(); ingest(model, s2, BOS)
            for b in payload + gap:
                ingest(model, s2, b)
            got = bytearray()
            for _k in range(span):
                sc = probe_scores(model, s2)
                sym = int(np.where(ELIG, sc, np.iinfo(np.int64).min).argmax())
                if sym >= 256:
                    break
                got.append(sym); ingest(model, s2, sym)
            exact += int(bytes(got) == payload)

        res[D] = {"first_bits_per_byte": round(float(np.mean(first_bits)), 4),
                  "repeat_bits_per_byte": round(float(np.mean(second_bits)), 4),
                  "retained_bits": round(float(np.mean(first_bits) - np.mean(second_bits)), 4),
                  "exact_copy_rate": round(exact / trials, 3)}
    return res


def eos_accuracy(model, records: list[bytes], limit=40) -> dict:
    """Does the model put EOS at the end of a record, and only there?"""
    tp = fp = fn = 0
    for rec in records[:limit]:
        s = Session(); ingest(model, s, BOS)
        for k, b in enumerate(rec):
            sc = probe_scores(model, s)
            pred_eos = int(np.where(ELIG, sc, np.iinfo(np.int64).min).argmax()) == EOS
            if pred_eos:
                fp += 1                    # predicted EOS mid-record
            ingest(model, s, b)
        sc = probe_scores(model, s)
        if int(np.where(ELIG, sc, np.iinfo(np.int64).min).argmax()) == EOS:
            tp += 1
        else:
            fn += 1
    prec = tp / max(tp + fp, 1)
    rec_ = tp / max(tp + fn, 1)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": round(prec, 4), "recall": round(rec_, 4),
            "f1": round(2 * prec * rec_ / max(prec + rec_, 1e-9), 4)}


def repetition(model, prompts: list[bytes], n_bytes=512) -> dict:
    """Loop and collapse frequency in greedy continuations."""
    collapsed = 0
    rows = []
    for p in prompts:
        s = Session(); ingest(model, s, BOS)
        for b in p:
            ingest(model, s, b)
        out, status = generate(model, s, n_bytes, text_mode=False)
        n = len(out)
        if n == 0:
            collapsed += 1
            rows.append({"bytes": 0, "collapsed": True}); continue
        run = best = 1
        for i in range(1, n):
            run = run + 1 if out[i] == out[i - 1] else 1
            best = max(best, run)
        tail = out[n // 2:]
        period = 0
        for q in range(1, min(40, len(tail) // 2 + 1)):
            if all(tail[j] == tail[j % q] for j in range(len(tail))):
                period = q; break
        bad = (len(set(out)) <= 3) or best >= 20 or (0 < period <= 8)
        collapsed += bad
        rows.append({"bytes": n, "distinct": len(set(out)), "longest_run": best,
                     "tail_period": period, "status": status, "collapsed": bool(bad)})
    return {"prompts": len(prompts), "collapse_rate": round(collapsed / max(len(prompts), 1), 3),
            "detail": rows}


def utf8_validity(model, prompts: list[bytes], n_bytes=256) -> dict:
    ok = 0
    for p in prompts:
        s = Session(); ingest(model, s, BOS)
        for b in p:
            ingest(model, s, b)
        out, _ = generate(model, s, n_bytes, text_mode=False)
        try:
            out.decode("utf-8"); ok += 1
        except UnicodeDecodeError:
            pass
    return {"prompts": len(prompts), "valid_utf8_rate": round(ok / max(len(prompts), 1), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a .mica model file")
    ap.add_argument("--records", default="r1/data/full/test.jsonl")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--routing", default="pairdiff")
    ap.add_argument("--out", default="r1/runs/language_eval.json")
    args = ap.parse_args()

    from mica_r1 import engine
    engine.ROUTING_MODE = args.routing
    model = serialize.load(args.model)
    recs = load_records(args.records)
    prompts = [r[:48] for r in recs[:args.prompts]]

    print(f"[lang] model {args.model}, routing {args.routing}")
    print("[lang] delayed copying ...", flush=True)
    dc = delayed_copy(model, recs[:200], trials=args.trials)
    for D, r in dc.items():
        print(f"    D={D:>5}: first {r['first_bits_per_byte']:.3f} -> repeat "
              f"{r['repeat_bits_per_byte']:.3f} bits/byte "
              f"(retained {r['retained_bits']:+.3f}), exact {r['exact_copy_rate']:.0%}")

    print("[lang] EOS accuracy ...", flush=True)
    eos = eos_accuracy(model, recs)
    print(f"    precision {eos['precision']:.3f} recall {eos['recall']:.3f} "
          f"F1 {eos['f1']:.3f}")

    print("[lang] repetition in 512-byte continuations ...", flush=True)
    rep = repetition(model, prompts)
    print(f"    collapse rate {rep['collapse_rate']:.0%}")

    print("[lang] UTF-8 validity ...", flush=True)
    u8 = utf8_validity(model, prompts)
    print(f"    valid UTF-8 rate {u8['valid_utf8_rate']:.0%}")

    out = {"model": args.model, "routing": args.routing,
           "delayed_copying": dc, "eos": eos, "repetition": rep, "utf8": u8,
           "not_measured": ["short completion preference (needs human raters)"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[lang] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
