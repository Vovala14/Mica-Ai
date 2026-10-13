"""Second, frozen-rule coefficient calibration experiment (PROTOCOL_2.md)."""
import argparse
import json
from pathlib import Path

from experiment import (BASE, HERE, EmberWord, file_sha, logits, np, outcomes,
                        save_json, summary, minimize)

SLICES = (slice(0, 240), slice(240, 480), slice(480, 720), slice(720, 816))


def components(model, task, features):
    x = features.astype(np.float32)
    parts = [x[:, s] @ model.Wf[task][:, s].T / 1024 for s in SLICES]
    parts.append(np.broadcast_to(model.B[task] / 1024, parts[0].shape).astype(np.float32))
    return np.stack(parts)


def fit_gains(parts, labels, mask, penalty):
    n = len(labels)
    def objective(gains):
        z = np.einsum("gnv,g->nv", parts, gains)
        if mask is not None:
            z[~mask] = -np.inf
        mx = z.max(axis=1)
        probs = np.exp(z - mx[:, None])
        norms = probs.sum(axis=1)
        loss = (mx + np.log(norms) - z[np.arange(n), labels]).mean()
        probs /= norms[:, None]
        grad = np.einsum("gnv,nv->g", parts, probs) / n - parts[:, np.arange(n), labels].mean(axis=1)
        d = gains - 1
        return float(loss + penalty * np.dot(d, d) / 2), grad + penalty * d
    result = minimize(objective, np.ones(5), jac=True, method="L-BFGS-B",
                      bounds=[(0.5, 1.5)] * 5, options={"maxiter": 60, "ftol": 1e-9})
    return result.x, {"converged": bool(result.success), "iterations": result.nit,
                      "objective": float(result.fun), "message": result.message}


def apply(model, original_w, original_b, gains):
    for task in gains:
        weights = original_w[task].copy()
        for s, gain in zip(SLICES, gains[task][:4]):
            weights[:, s] = np.clip(np.rint(original_w[task][:, s].astype(np.float64) * gain), -127, 127).astype(np.int8)
        model.W[task] = weights
        model.Wf[task] = weights.astype(np.float32)
        model.B[task] = np.rint(original_b[task] * gains[task][4]).astype(np.int64)


def fit(work):
    model = EmberWord()
    original_w = {k: v.copy() for k, v in model.W.items()}
    original_b = {k: v.copy() for k, v in model.B.items()}
    rows = json.loads((work / "records.json").read_text())
    f = np.load(work / "features.npz")
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    select = np.array([i for i, r in enumerate(rows) if r["split"] == "selection"])
    selected_rows = [rows[i] for i in select]
    baseline_scores = {t: logits(model, f[t][select], t) for t in model.W}
    zero = {t: np.zeros(len(model.words), np.int32) for t in model.W}
    baseline = outcomes(model, baseline_scores, selected_rows, f["labels"][select], zero)
    parts = {t: components(model, t, f[t][train]) for t in model.W}
    mask = np.zeros((len(train), len(model.words)), bool)
    for n, i in enumerate(train):
        mask[n, model.candidates(rows[i]["target"][:2])] = True
    report = {"baseline_selection": summary(baseline), "candidates": []}
    save_json(work / "selection_baseline_rows.json", baseline)
    for penalty in (0.01, 0.1, 1.0):
        gains, fitting = {}, {}
        for task in model.W:
            gains[task], fitting[task] = fit_gains(parts[task], f["labels"][train],
                                                mask if task == "prefix2" else None, penalty)
        apply(model, original_w, original_b, gains)
        scores = {task: logits(model, f[task][select], task) for task in model.W}
        metrics = outcomes(model, scores, selected_rows, f["labels"][select], zero)
        candidate = {"penalty": penalty, "gains": {k: v.tolist() for k, v in gains.items()},
                     "fitting": fitting, "selection": summary(metrics)}
        report["candidates"].append(candidate)
        save_json(work / f"selection_rows_{penalty}.json", metrics)
        print(json.dumps(candidate), flush=True)
    selected = max(report["candidates"], key=lambda c: (c["selection"]["equal_domain"]["primary"], c["penalty"]))
    report["selected"] = selected
    save_json(work / "selection.json", report)


def verify(work):
    model = EmberWord()
    report = json.loads((work / "confirmation.json").read_text())
    assert file_sha(work / "candidate_word_heads.npz") == report["word_heads_sha256"]
    changed = {f"{task}_{part}" for task in model.W for part in ("W", "B")}
    with np.load(BASE / "word_heads.npz", allow_pickle=False) as original, np.load(work / "candidate_word_heads.npz", allow_pickle=False) as candidate:
        assert set(original.files) == set(candidate.files)
        for key in original.files:
            if key not in changed:
                assert np.array_equal(original[key], candidate[key]), key
        for task in model.W:
            model.W[task] = candidate[f"{task}_W"]
            model.Wf[task] = model.W[task].astype(np.float32)
            model.B[task] = candidate[f"{task}_B"].astype(np.int64)
            assert model.W[task].dtype == np.int8
    rows = json.loads((work / "records.json").read_text())
    features = np.load(work / "features.npz")
    outcomes_by_id = {r["id"]: r for r in json.loads((work / "confirmation_rows.json").read_text())["candidate"]}
    chosen = [(i, r) for i, r in enumerate(rows) if r["split"] == "confirmation"][::40]
    for task in model.W:
        weights = model.W[task].astype(np.int64)
        for i, _ in chosen:
            x = features[task][i].astype(np.int64)
            assert np.array_equal(model._scores(task, x), weights @ x + model.B[task])
    for _, row in chosen:
        expected = outcomes_by_id[row["id"]]
        assert model.suggest(row["context"]) == [model.words[i] for i in expected["next_ids"]]
        for length in (1, 2, 3):
            assert model.complete(row["context"] + row["target"][:length]) == [model.words[i] for i in expected[f"prefix{length}_ids"]]
    print(f"Verified: only existing readout W/B changed; integer scores and runtime top-5 match on {len(chosen)} prompts.")


def confirm(work):
    if (work / "confirmation.json").exists():
        raise RuntimeError("Confirmation already evaluated")
    model = EmberWord()
    rows = json.loads((work / "records.json").read_text())
    f = np.load(work / "features.npz")
    idx = np.array([i for i, row in enumerate(rows) if row["split"] == "confirmation"])
    rows = [rows[i] for i in idx]
    zero = {t: np.zeros(len(model.words), np.int32) for t in model.W}
    scores = {t: logits(model, f[t][idx], t) for t in model.W}
    baseline = outcomes(model, scores, rows, f["labels"][idx], zero)
    selected = json.loads((work / "selection.json").read_text())["selected"]
    apply(model, model.W, model.B, selected["gains"])
    scores = {t: logits(model, f[t][idx], t) for t in model.W}
    candidate = outcomes(model, scores, rows, f["labels"][idx], zero)
    before, after = summary(baseline), summary(candidate)
    rng = np.random.default_rng(38)
    samples = []
    for domain in ("conversation", "prose"):
        groups = sorted({r["group"] for r in rows if r["domain"] == domain})
        totals = np.array([[sum(c["primary"] - b["primary"] for b, c in zip(baseline, candidate) if b["group"] == group),
                            sum(r["group"] == group for r in rows)] for group in groups])
        picked = totals[rng.integers(len(totals), size=(10000, len(totals)))].sum(axis=1)
        samples.append(picked[:, 0] / picked[:, 1])
    ci = np.quantile(np.mean(samples, axis=0), [0.025, 0.975]).tolist()
    gain = after["equal_domain"]["primary"] - before["equal_domain"]["primary"]
    gates = {"gain_at_least_one_point": gain >= .01, "paired_ci_positive": ci[0] > 0,
             "domain_no_regression": all(after["domains"][d]["primary"] - before["domains"][d]["primary"] >= -.02 for d in before["domains"]),
             "task_no_regression": all(after["equal_domain"][t] - before["equal_domain"][t] >= -.02 for t in ("next", "prefix2"))}
    with np.load(BASE / "word_heads.npz", allow_pickle=False) as source:
        pack = {key: source[key] for key in source.files}
    for task in model.W:
        pack[f"{task}_W"] = model.W[task]
        pack[f"{task}_B"] = model.B[task].astype(np.int32)
    np.savez_compressed(work / "candidate_word_heads.npz", **pack)
    report = {"baseline": before, "candidate": after, "primary_gain": gain,
              "paired_cluster_bootstrap_ci95": ci, "gates": gates, "passed": all(gates.values()),
              "selected": selected, "word_heads_sha256": file_sha(work / "candidate_word_heads.npz")}
    save_json(work / "confirmation_rows.json", {"baseline": baseline, "candidate": candidate})
    save_json(work / "confirmation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "confirm", "verify"))
    parser.add_argument("--work", type=Path, default=Path("/workspace/scratch/mica038/run2"))
    args = parser.parse_args()
    globals()[args.stage](args.work)
