#!/usr/bin/env python3
"""Integerize the 0.3.1 memory fit (mask applied), write the release memory.npz, and score
val500 / dev bits with the exact integer readout.
    python quant031.py ../mem_v031.pt .
"""
import json, math, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/claude/mica/mem")
import fit_mem as FM, final_mem as FMm
import word_eval as E
from word_eval import W

ck = torch.load(sys.argv[1], weights_only=False)
out = Path(sys.argv[2]); out.mkdir(exist_ok=True)
st = {k: v.clone() for k, v in ck["state"].items()}
st["A"] = st["A"] * st["mask"]; st["Bm"] = st["Bm"] * st["mask"]
q = FMm.quantize(st, 16)
for k, lo, hi in (("A_int", -127, 127), ("Bm_int", -127, 127), ("lam_int", -32768, 32767), ("R_int", -2**31, 2**31 - 1)):
    assert q[k].min() >= lo and q[k].max() <= hi, k
np.savez(out / "final_v031_int.npz", **q)
rel = {"R_int": q["R_int"].astype(np.int32), "A_int": q["A_int"].astype(np.int8), "lam_int": q["lam_int"].astype(np.int16),
       "Bm_int": q["Bm_int"].astype(np.int8), "MUL": q["MUL"].astype(np.int64),
       "class_of_id": np.asarray(FM.classes()).astype(np.int8), "never_read": np.array([0, W.BOS, W.EOS], np.int32)}
np.savez(out / "memory.npz", **rel)
print("memory.npz written; R_int", rel["R_int"].tolist(), "lam_int", rel["lam_int"].tolist(), flush=True)
if "--no-bits" in sys.argv:
    sys.exit()
m0 = E.load_model("mica:/home/claude/mica/wordw/pkg_B")
co, bias, elig = m0.co.astype(np.int64), m0.bias.astype(np.int64), m0.elig
cls = FM.classes()
res = {"select_float": ck["dev_bits"]}
M = "/home/claude/mica/mem/"
for name, files in (("val500", ["test_val500.npz"]),
                    ("dev", ["dev_chat.npz", "dev_every.npz", "dev_chatlong.npz", "dev_everylong.npz"])):
    F, H, T = FM.load_tokens_npz([M + f for f in files])
    lp = FMm.logp_rows(F, H, T, q, co, bias, elig, cls)
    res[f"{name}_bits_int"] = float(-lp.mean() / math.log(2)); res[f"{name}_n"] = int(len(T))
    print(name, res[f"{name}_bits_int"], len(T), flush=True)
json.dump(res, open(out / "bits_v031.json", "w"), indent=1)
