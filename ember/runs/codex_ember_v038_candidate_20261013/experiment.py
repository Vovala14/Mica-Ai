#!/usr/bin/env python3
"""Reproducible frozen-rule bias refit. See PROTOCOL.md before running."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "ember/runs/ember_v037_20261011"
sys.path.insert(0, str(BASE))
from word_model import EmberWord, word_starts


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def source_records(data):
    conversations = json.loads((data / "taskmaster3.json").read_text())
    for conversation in conversations:
        group = "taskmaster3:" + conversation["conversation_id"]
        for utterance in conversation["utterances"]:
            yield "conversation", group, utterance["text"]
    with zipfile.ZipFile(data / "brown.zip") as archive:
        for name in sorted(archive.namelist()):
            if re.fullmatch(r"brown/c[a-z]\d\d", name):
                for line in archive.read(name).decode("ascii").splitlines():
                    words = [token.rsplit("/", 1)[0] for token in line.split()]
                    if words:
                        yield "prose", "brown:" + name, " ".join(words)


def prepare(data, work, offset=0, reuse=None):
    model = EmberWord()
    vocabulary = {word: i for i, word in enumerate(model.words)}
    pools = {(domain, split): [] for domain in ("conversation", "prose")
             for split in ("train", "selection", "confirmation")}
    seen_text, seen_prompt = set(), set()
    for domain, group, text in source_records(data):
        text = re.sub(r"\s+", " ", text).strip()
        normalized = text.casefold()
        if normalized in seen_text:
            continue
        seen_text.add(normalized)
        positions = []
        for match in re.finditer(r"(?<!\S)([A-Za-z]{4,})(?=\s|[.,!?;:]|$)", text):
            target = match.group(1).lower()
            context = text[:match.start()]
            if target not in vocabulary or len(context.strip().split()) < 2:
                continue
            # Cut at word boundaries until the prefix fits the fixed budget.
            while len(context.encode()) > 160:
                context = context.split(" ", 1)[1]
            positions.append((digest(context.casefold() + target), context, target))
        if not positions:
            continue
        _, context, target = min(positions)
        identity = digest(context.casefold() + target)
        if identity in seen_prompt:
            continue
        seen_prompt.add(identity)
        bucket = int(digest(group), 16) % 10
        split = "train" if bucket < 6 else "selection" if bucket < 8 else "confirmation"
        pools[domain, split].append(dict(domain=domain, group=group, split=split,
                                        context=context, target=target, id=identity))
    rows = []
    for (domain, split), pool in sorted(pools.items()):
        count = 400 if split == "train" else 200
        start = 0 if split == "train" else offset
        rows.extend(sorted(pool, key=lambda r: r["id"])[start:start + count])
    work.mkdir(parents=True, exist_ok=True)
    save_json(work / "records.json", rows)
    manifest = {"sources": json.loads((data / "sources.json").read_text()),
                "source_hashes": {p.name: file_sha(p) for p in (data / "taskmaster3.json", data / "brown.zip")},
                "protocol_sha256": file_sha(HERE / "PROTOCOL.md"),
                "baseline_sha256": file_sha(BASE / "word_heads.npz"),
                "records_sha256": file_sha(work / "records.json"),
                "counts": {f"{d}/{s}": sum(r["domain"] == d and r["split"] == s for r in rows)
                           for d, s in pools}}
    save_json(work / "data_manifest.json", manifest)
    features = {task: [] for task in ("next", "prefix2")}
    labels = []
    old_rows = json.loads((reuse / "records.json").read_text()) if reuse else []
    old_features = np.load(reuse / "features.npz") if reuse else None
    cached = {row["id"]: i for i, row in enumerate(old_rows)}
    assert not any(row["id"] in cached for row in rows if row["split"] != "train") if reuse else True
    for n, row in enumerate(rows):
        if row["id"] in cached:
            i = cached[row["id"]]
            for task in features:
                features[task].append(old_features[task][i])
            labels.append(int(old_features["labels"][i]))
            continue
        context, target = row["context"], row["target"]
        data_bytes = (context + target[:2]).encode()
        pos = len(context.encode())
        probes = model.probes(data_bytes)
        starts = word_starts(data_bytes)
        for task, offset in (("next", 0), ("prefix2", 2)):
            features[task].append(model._input(task, data_bytes, probes, starts, pos + offset, pos).astype(np.int8))
        labels.append(vocabulary[target])
        if (n + 1) % 100 == 0:
            print(f"features {n + 1}/{len(rows)}", flush=True)
    np.savez_compressed(work / "features.npz", labels=np.asarray(labels), **{k: np.asarray(v) for k, v in features.items()})
    manifest["features_sha256"] = file_sha(work / "features.npz")
    manifest["experiment_sha256"] = file_sha(HERE / "experiment.py")
    save_json(work / "data_manifest.json", manifest)
    print(manifest["counts"], flush=True)


def logits(model, features, task):
    # Integer products are exact in float32 under the released range bound.
    out = np.empty((len(features), len(model.words)), np.float32)
    for start in range(0, len(features), 32):
        x = features[start:start + 32].astype(np.float32)
        out[start:start + 32] = x @ model.Wf[task].T + model.B[task]
    return out


def fit_bias(scores, labels, strength):
    z0 = scores.astype(np.float64) / 1024
    count, vocab = z0.shape
    target_count = np.bincount(labels, minlength=vocab) / count

    def objective(delta):
        z = z0 + delta
        max_z = z.max(axis=1)
        probs = np.exp(z - max_z[:, None])
        norm = probs.sum(axis=1)
        loss = (max_z + np.log(norm) - z[np.arange(count), labels]).mean()
        grad = (probs / norm[:, None]).mean(axis=0) - target_count + strength * delta
        return float(loss + strength * np.dot(delta, delta) / 2), grad

    result = minimize(objective, np.zeros(vocab), jac=True, method="L-BFGS-B",
                      options={"maxiter": 60, "ftol": 1e-9})
    return np.rint(result.x * 1024).astype(np.int32), {
        "strength": strength, "iterations": result.nit, "converged": bool(result.success),
        "message": result.message, "objective": float(result.fun)}


def outcomes(model, scores, rows, labels, deltas):
    metrics = []
    for n, row in enumerate(rows):
        result = {k: row[k] for k in ("id", "domain", "group", "split")}
        for length in (0, 1, 2, 3):
            task = "next" if length < 2 else "prefix2"
            idx = np.arange(len(model.words)) if not length else model.candidates(row["target"][:length])
            values = scores[task][n, idx].astype(np.int64) + deltas[task][idx]
            ranked = idx[np.argsort(values, kind="stable")[::-1][:5]]
            name = "next" if not length else f"prefix{length}"
            result[name] = int(len(ranked) > 0 and ranked[0] == labels[n])
            result[name + "_top5"] = int(labels[n] in ranked)
            result[name + "_ids"] = ranked.tolist()
        result["primary"] = (result["next"] + result["prefix2"]) / 2
        metrics.append(result)
    return metrics


def summary(rows):
    keys = ["primary", "next", "next_top5", "prefix1", "prefix1_top5", "prefix2", "prefix2_top5", "prefix3", "prefix3_top5"]
    by_domain = {domain: {key: float(np.mean([r[key] for r in rows if r["domain"] == domain])) for key in keys}
                 for domain in ("conversation", "prose")}
    return {"domains": by_domain,
            "equal_domain": {key: float(np.mean([v[key] for v in by_domain.values()])) for key in keys}}


def fit(work):
    model = EmberWord()
    rows = json.loads((work / "records.json").read_text())
    features = np.load(work / "features.npz")
    train_idx = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    select_idx = np.array([i for i, r in enumerate(rows) if r["split"] == "selection"])
    selected_rows = [rows[i] for i in select_idx]
    train_scores, selection_scores = {}, {}
    for task in ("next", "prefix2"):
        train_scores[task] = logits(model, features[task][train_idx], task)
        selection_scores[task] = logits(model, features[task][select_idx], task)
    for n, i in enumerate(train_idx):
        allowed = model.candidates(rows[i]["target"][:2])
        retained = train_scores["prefix2"][n, allowed].copy()
        train_scores["prefix2"][n] = -np.inf
        train_scores["prefix2"][n, allowed] = retained
    zero = {task: np.zeros(len(model.words), np.int32) for task in train_scores}
    baseline = outcomes(model, selection_scores, selected_rows, features["labels"][select_idx], zero)
    save_json(work / "selection_baseline_rows.json", baseline)
    report = {"baseline_selection": summary(baseline), "candidates": []}
    for strength in (0.001, 0.01, 0.1):
        deltas, fits = {}, {}
        for task in train_scores:
            deltas[task], fits[task] = fit_bias(train_scores[task], features["labels"][train_idx], strength)
        metrics = outcomes(model, selection_scores, selected_rows, features["labels"][select_idx], deltas)
        save_json(work / f"selection_rows_{strength}.json", metrics)
        candidate = {"strength": strength, "selection": summary(metrics), "fitting": fits}
        report["candidates"].append(candidate)
        np.savez_compressed(work / f"delta_{strength}.npz", **deltas)
        print(json.dumps(candidate), flush=True)
    chosen = max(report["candidates"], key=lambda c: (c["selection"]["equal_domain"]["primary"], c["strength"]))
    report["selected_strength"] = chosen["strength"]
    save_json(work / "selection.json", report)


def confirm(work):
    if (work / "confirmation.json").exists():
        raise RuntimeError("Confirmation already evaluated; do not tune on it")
    model = EmberWord()
    rows = json.loads((work / "records.json").read_text())
    f = np.load(work / "features.npz")
    idx = np.array([i for i, row in enumerate(rows) if row["split"] == "confirmation"])
    rows = [rows[i] for i in idx]
    selection = json.loads((work / "selection.json").read_text())
    delta_path = work / f"delta_{selection['selected_strength']}.npz"
    delta = dict(np.load(delta_path))
    scores = {task: logits(model, f[task][idx], task) for task in delta}
    zero = {task: np.zeros(len(model.words), np.int32) for task in delta}
    baseline = outcomes(model, scores, rows, f["labels"][idx], zero)
    candidate = outcomes(model, scores, rows, f["labels"][idx], delta)
    before, after = summary(baseline), summary(candidate)
    groups = []
    rng = np.random.default_rng(38)
    for domain in ("conversation", "prose"):
        group_names = sorted({r["group"] for r in rows if r["domain"] == domain})
        totals = np.array([[sum(c["primary"] - b["primary"] for b, c in zip(baseline, candidate) if b["group"] == group),
                            sum(r["group"] == group for r in rows)] for group in group_names])
        sample = rng.integers(len(totals), size=(10000, len(totals)))
        picked = totals[sample].sum(axis=1)
        groups.append(picked[:, 0] / picked[:, 1])
    ci = np.quantile(np.mean(groups, axis=0), [0.025, 0.975]).tolist()
    gain = after["equal_domain"]["primary"] - before["equal_domain"]["primary"]
    gates = {"gain_at_least_one_point": gain >= 0.01, "paired_ci_positive": ci[0] > 0,
             "domain_no_regression": all(after["domains"][d]["primary"] - before["domains"][d]["primary"] >= -0.02 for d in before["domains"]),
             "task_no_regression": all(after["equal_domain"][t] - before["equal_domain"][t] >= -0.02 for t in ("next", "prefix2"))}
    report = {"baseline": before, "candidate": after, "primary_gain": gain,
              "paired_cluster_bootstrap_ci95": ci, "gates": gates,
              "passed": all(gates.values()), "selected_strength": selection["selected_strength"],
              "delta_sha256": file_sha(delta_path)}
    save_json(work / "confirmation_rows.json", {"baseline": baseline, "candidate": candidate})
    save_json(work / "confirmation.json", report)
    # Export actual integer head biases. Every other packed array is unchanged.
    with np.load(BASE / "word_heads.npz", allow_pickle=False) as source:
        pack = {key: source[key] for key in source.files}
    for task in delta:
        values = pack[f"{task}_B"].astype(np.int64) + delta[task]
        assert np.all((values >= np.iinfo(np.int32).min) & (values <= np.iinfo(np.int32).max))
        pack[f"{task}_B"] = values.astype(np.int32)
    np.savez_compressed(work / "candidate_word_heads.npz", **pack)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "fit", "confirm"))
    parser.add_argument("--data", type=Path, default=Path("/workspace/scratch/mica038"))
    parser.add_argument("--work", type=Path, default=Path("/workspace/scratch/mica038/run"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.data, args.work, args.offset, args.reuse)
    else:
        globals()[args.stage](args.work)
