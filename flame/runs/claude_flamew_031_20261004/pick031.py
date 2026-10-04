#!/usr/bin/env python3
"""Re-pick the answer-mode gain for the 0.3.1 memory on the synthetic practice items only
(same grid and rule as for v0.3: highest accuracy, ties to the smaller gain)."""
import json, pickle, sys
import numpy as np
sys.path.insert(0, "/home/claude/mica/ansmode")
from eval_ans import scores, acc
syn = pickle.load(open("cache_syn031.pkl", "rb")); lab = np.array([x["label"] for x in syn])
grid = [round(x, 4) for x in np.arange(0, 4.01, 0.25)]
res = {g: acc(scores(syn, g), lab) for g in grid}
g_star = max(grid, key=lambda g: (res[g], -g))
print("synthetic practice items:", {g: round(a, 4) for g, a in res.items()}); print("gain picked:", g_star)
json.dump({"grid_synthetic": res, "g_star": g_star, "n_items": len(syn)}, open("pick031.json", "w"), indent=1)
am = np.load("/home/claude/mica/final_tom/answer_mode.npz")
np.savez("answer_mode.npz", content=am["content"], gain=np.int64(round(g_star * 1024)))
