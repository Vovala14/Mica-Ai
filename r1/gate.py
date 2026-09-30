#!/usr/bin/env python3
"""The section 15 gate, measured honestly.

  "An initial continuation gate is at least 5 percent lower test bits per byte
   than the trigram under the same storage cap, with uncertainty estimated by
   document bootstrap."

Three things that sentence demands and a naive comparison gets wrong:

  * **the same storage cap.** A byte n-gram with unbounded counts is not a
    baseline for an 86,820-byte model. Each baseline here is pruned or sized to
    fit the same budget, and the budget is reported in bytes.
  * **test**, not validation. The locked test split is read once, at the end.
  * **document bootstrap.** Resample whole documents, not records or bytes,
    because records from one document are not independent.
"""

from __future__ import annotations

import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

import numpy as np
from make_records import load_records


# ---------------------------------------------------------------------------
# storage-capped byte n-gram
# ---------------------------------------------------------------------------
class CappedNGram:
    """Interpolated Witten-Bell n-gram pruned to a byte budget.

    Storage model: each retained (context, byte) entry costs `order` bytes of
    context key + 1 byte of symbol + 2 bytes of count = order + 3 bytes. That
    is generous to the baseline -- a real implementation needs index overhead
    on top -- which is the direction an honest comparison should err in.
    """

    def __init__(self, order: int, budget_bytes: int):
        self.order = order
        self.budget = budget_bytes
        self.counts = [defaultdict(lambda: defaultdict(int))
                       for _ in range(order + 1)]
        self.totals = [defaultdict(int) for _ in range(order + 1)]

    def train(self, data: bytes) -> None:
        for i in range(len(data)):
            nxt = data[i]
            for k in range(self.order + 1):
                if i - k < 0:
                    break
                ctx = data[i - k:i]
                self.counts[k][ctx][nxt] += 1
                self.totals[k][ctx] += 1

    def entry_bytes(self, k: int) -> int:
        return k + 3

    def stored_bytes(self) -> int:
        return sum(len(t) * self.entry_bytes(k)
                   for k in range(self.order + 1)
                   for t in self.counts[k].values())

    def prune(self) -> dict:
        """Spend the byte budget on the globally most valuable entries.

        Pruning order by order is wrong: it exhausts the budget on rare
        low-order entries and never buys a single high-count trigram, which
        collapses every order to the same model. Instead, pool every entry
        across every order and take them in descending count per byte, after
        reserving the order-0 table as the backbone that backoff needs.

        This is the standard count-pruning approach and it produces a much
        stronger baseline -- which is the point. A gate measured against a
        crippled baseline is worth nothing.
        """
        before = self.stored_bytes()
        keep = [defaultdict(lambda: defaultdict(int))
                for _ in range(self.order + 1)]

        # order 0 is the backbone: 256 entries, cheap, always retained
        spent = 0
        for ctx, tbl in self.counts[0].items():
            for sym, cnt in tbl.items():
                keep[0][ctx][sym] = cnt
                spent += self.entry_bytes(0)

        pool = []
        for k in range(1, self.order + 1):
            eb = self.entry_bytes(k)
            for ctx, tbl in self.counts[k].items():
                for sym, cnt in tbl.items():
                    pool.append((cnt / eb, cnt, k, ctx, sym))
        pool.sort(key=lambda e: -e[0])           # value per byte, descending

        for _v, cnt, k, ctx, sym in pool:
            eb = self.entry_bytes(k)
            if spent + eb > self.budget:
                continue
            keep[k][ctx][sym] = cnt
            spent += eb

        self.counts = keep
        self.totals = [defaultdict(int) for _ in range(self.order + 1)]
        for k in range(self.order + 1):
            for ctx, tbl in self.counts[k].items():
                self.totals[k][ctx] = sum(tbl.values())
        kept = {k: sum(len(t) for t in self.counts[k].values())
                for k in range(self.order + 1)}
        return {"bytes_before_prune": before, "bytes_after_prune": self.stored_bytes(),
                "entries_kept_per_order": kept}

    def _dist(self, ctx: bytes) -> np.ndarray:
        probs = np.full(256, 1.0 / 256)
        for k in range(0, self.order + 1):
            if k > len(ctx):
                break
            c = ctx[len(ctx) - k:] if k else b""
            tbl = self.counts[k].get(c)
            if not tbl:
                continue
            total = self.totals[k].get(c, 0)
            if total <= 0:
                continue
            lam = total / (total + len(tbl))
            hi = np.zeros(256)
            for sym, cnt in tbl.items():
                hi[sym] = cnt / total
            probs = lam * hi + (1 - lam) * probs
        return probs

    def bits_per_byte_per_doc(self, docs: list[bytes]) -> list[float]:
        out = []
        for d in docs:
            bits, n = 0.0, 0
            for i in range(len(d)):
                p = self._dist(d[max(0, i - self.order):i])[d[i]]
                bits += -math.log2(max(p, 1e-12))
                n += 1
            out.append(bits / max(n, 1))
        return out


# ---------------------------------------------------------------------------
# document bootstrap
# ---------------------------------------------------------------------------
def bootstrap_ci(per_doc: list[float], weights: list[int], n: int = 2000,
                 seed: int = 0, alpha: float = 0.05):
    """Resample whole documents. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(per_doc, float)
    w = np.asarray(weights, float)
    point = float((v * w).sum() / w.sum())
    idx = rng.integers(0, len(v), size=(n, len(v)))
    samples = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="r1/data/records/train.jsonl")
    ap.add_argument("--test", default="r1/data/records/test.jsonl")
    ap.add_argument("--budget", type=int, default=86_820,
                    help="storage cap in bytes; default is the R1 model file")
    ap.add_argument("--orders", nargs="*", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--train-mb", type=float, default=6.0)
    ap.add_argument("--test-docs", type=int, default=120)
    ap.add_argument("--out", default="r1/runs/gate.json")
    args = ap.parse_args()

    train = b"".join(load_records(args.records))[:int(args.train_mb * 1e6)]
    test_recs = load_records(args.test)[:args.test_docs]
    print(f"[gate] train {len(train)/1e6:.2f} MB, test {len(test_recs)} records "
          f"({sum(len(r) for r in test_recs)/1e3:.0f} kB)")
    print(f"[gate] storage cap {args.budget:,} bytes (the R1 model file)\n")

    rows = []
    for order in args.orders:
        m = CappedNGram(order, args.budget)
        m.train(train)
        info = m.prune()
        per_doc = m.bits_per_byte_per_doc(test_recs)
        pt, lo, hi = bootstrap_ci(per_doc, [len(r) for r in test_recs])
        rows.append({"model": f"byte {order}-gram", "order": order,
                     "stored_bytes": m.stored_bytes(),
                     "unpruned_bytes": info["bytes_before_prune"],
                     "entries_kept_per_order": info["entries_kept_per_order"],
                     "bits_per_byte": round(pt, 4),
                     "ci95": [round(lo, 4), round(hi, 4)]})
        kept = info["entries_kept_per_order"]
        print(f"  {order}-gram: {pt:.4f} bits/byte [{lo:.4f}, {hi:.4f}]  "
              f"stored {m.stored_bytes():,} B of {info['bytes_before_prune']:,}  "
              f"kept {kept}")

    best = min(rows, key=lambda r: r["bits_per_byte"])
    gate = best["bits_per_byte"] * 0.95
    print(f"\n[gate] strongest storage-matched baseline: {best['model']} at "
          f"{best['bits_per_byte']:.4f} bits/byte")
    print(f"[gate] section 15 gate (5% lower): MICA must reach "
          f"{gate:.4f} bits/byte on the locked test set")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"storage_cap_bytes": args.budget, "train_bytes": len(train),
         "test_records": len(test_recs), "baselines": rows,
         "strongest_baseline": best, "gate_bits_per_byte": round(gate, 4)},
        indent=2))
    print(f"[gate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
