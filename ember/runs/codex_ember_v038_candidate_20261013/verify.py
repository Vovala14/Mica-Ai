"""Check fitted integer artifacts against the released runtime and saved rankings."""
import argparse
import json
from pathlib import Path
import numpy as np

from experiment import BASE, EmberWord, file_sha


def verify(work):
    model = EmberWord()
    report = json.loads((work / "confirmation.json").read_text())
    delta_path = work / f"delta_{report['selected_strength']}.npz"
    assert file_sha(delta_path) == report["delta_sha256"]
    with np.load(BASE / "word_heads.npz", allow_pickle=False) as baseline, np.load(work / "candidate_word_heads.npz", allow_pickle=False) as candidate:
        delta = dict(np.load(delta_path))
        assert set(baseline.files) == set(candidate.files)
        for key in baseline.files:
            if key in ("next_B", "prefix2_B"):
                task = key[:-2]
                assert candidate[key].dtype == np.int32
                assert np.array_equal(candidate[key].astype(np.int64), baseline[key].astype(np.int64) + delta[task])
            else:
                assert np.array_equal(candidate[key], baseline[key]), key
    rows = json.loads((work / "records.json").read_text())
    cached = np.load(work / "features.npz")
    saved = json.loads((work / "confirmation_rows.json").read_text())
    confirmation = {r["id"]: r for r in saved["candidate"]}
    chosen = [(i, r) for i, r in enumerate(rows) if r["split"] == "confirmation"][::40]
    original_bias = {task: bias.copy() for task, bias in model.B.items()}
    for task in model.B:
        model.B[task] = original_bias[task] + delta[task]
        weights = model.W[task].astype(np.int64)
        for i, _ in chosen:
            x = cached[task][i].astype(np.int64)
            assert np.array_equal(model._scores(task, x), weights @ x + model.B[task])
    for _, row in chosen:
        expected = confirmation[row["id"]]
        words = model.suggest(row["context"], top_k=5)
        assert words == [model.words[i] for i in expected["next_ids"]]
        for length in (1, 2, 3):
            actual = model.suggest(row["context"] + row["target"][:length], prefix2=True, top_k=5)
            assert actual == [model.words[i] for i in expected[f"prefix{length}_ids"]]
    print(f"Verified: only two integer biases changed; exact scalar scores and runtime top-5 match on {len(chosen)} prompts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("/workspace/scratch/mica038/run"))
    verify(parser.parse_args().work)
