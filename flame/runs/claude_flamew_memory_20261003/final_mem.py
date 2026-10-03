#!/usr/bin/env python3
"""Quantize a fitted memory readout to integers, check it against the float fit on dev,
then score val500 bits and Tiny Theory-of-Mind once.

    python final_mem.py mem_k16.pt OUT.json [--tom]
"""
import json, math, pickle, sys
import numpy as np, torch
import fit_mem as FM
from word_eval import W
import word_eval as E

SH = 30


def quantize(st, k):
    R = st["R"].numpy()
    out = {"R_int": np.round(R * 1024).astype(np.int64)}
    if k:
        A, Bm, lam = st["A"].numpy(), st["Bm"].numpy(), st["lam"].numpy()
        sa = np.abs(A).max() / 127.0
        sl = np.abs(lam).max() / 127.0
        sb = np.maximum(np.abs(Bm).max(1), 1e-12) / 127.0              # per word
        out["A_int"] = np.round(A / sa).astype(np.int64)
        out["lam_int"] = np.round(lam / sl).astype(np.int64)
        out["Bm_int"] = np.round(Bm / sb[:, None]).astype(np.int64)
        out["MUL"] = np.round(1024.0 * sa * sl * sb * 2.0 ** SH).astype(np.int64)
    return out


def int_bonus(q, Hrow_ids, Hrow_lags, cls):
    """Integer bonus for one position. Hrow_ids: ids (>=0) in the window, lags 1..64."""
    bonus = np.zeros(W.N_SYMBOLS, np.int64)
    if len(Hrow_ids) == 0:
        return bonus
    lb = FM.LAGB[Hrow_lags]
    np.add.at(bonus, Hrow_ids, q["R_int"][cls[Hrow_ids], lb])
    if "A_int" in q:
        M = (q["lam_int"][lb][:, None] * q["A_int"][Hrow_ids]).sum(0)          # int64 (k,)
        dot = q["Bm_int"] @ M                                                 # int64 (V,)
        bonus += (dot * q["MUL"]) >> SH
    return bonus


def logp_rows(F, Hs, T, q, co, bias, elig, cls):
    out = np.empty(len(T))
    for i in range(len(T)):
        h = Hs[i]; ok = h >= 0
        z = bias + co @ F[i].astype(np.int64) + int_bonus(q, h[ok], np.nonzero(ok)[0] + 1, cls)
        z = z.astype(np.float64) / 1024.0
        z = np.where(elig, z, -np.inf); mx = z.max()
        out[i] = z[T[i]] - mx - math.log(np.exp(z - mx).sum())
    return out


def main():
    ck = torch.load(sys.argv[1], weights_only=False)
    k = ck["k"]
    m0 = E.load_model("mica:/home/claude/mica/wordw/pkg_B")
    co, bias, elig = m0.co.astype(np.int64), m0.bias.astype(np.int64), m0.elig
    cls = FM.classes()
    q = quantize(ck["state"], k)
    res = {"k": k, "float_dev_bits": ck["dev_bits"], "dev_base": ck["dev_base"]}
    for name, files in (("dev", ["dev_chat.npz", "dev_every.npz", "dev_chatlong.npz", "dev_everylong.npz"]),
                        ("val500", ["test_val500.npz"])):
        F, H, T = FM.load_tokens_npz(files)
        lp = logp_rows(F, H, T, q, co, bias, elig, cls)
        res[f"{name}_bits_int"] = float(-lp.mean() / math.log(2))
        print(name, res[f"{name}_bits_int"], flush=True)
    if "--tom" in sys.argv:
        rows = pickle.load(open("tom.pkl", "rb"))
        correct, per = 0, []
        for r in rows:
            means = []
            for ei, Fe in r["ends"]:
                Hs, T = FM.histories([(r["ctx"], ei)])
                lp = logp_rows(Fe, Hs, T, q, co, bias, elig, cls)
                means.append(lp.mean() if len(lp) else -np.inf)
            pred = int(np.argmax(means)); c = pred == r["label"]; correct += c
            per.append({"ind": r["ind"], "label": r["label"], "pred": pred, "correct": bool(c), "topic": r["topic"]})
        res["tom_accuracy"] = correct / len(rows)
        res["tom_rows"] = per
        print("ToM", res["tom_accuracy"], flush=True)
    json.dump({kk: (v.tolist() if isinstance(v, np.ndarray) else v) for kk, v in res.items()},
              open(sys.argv[2], "w"))
    np.savez(sys.argv[2].replace(".json", "_int.npz"), **q)


if __name__ == "__main__":
    main()
