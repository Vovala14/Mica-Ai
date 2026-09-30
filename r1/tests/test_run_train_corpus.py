"""The launcher must not mix a saved model with another corpus."""

import json
import os
from pathlib import Path
import sys


R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
_mica_env = {k: v for k, v in os.environ.items() if k.startswith("MICA_")}
import run_train  # noqa: E402
for _key in run_train.GEOM:
    if _key in _mica_env:
        os.environ[_key] = _mica_env[_key]
    else:
        os.environ.pop(_key, None)


def _saved_run(run_dir, records="r1/data/bulk/train.jsonl",
               val_records="r1/data/bulk/val.jsonl"):
    run_dir.mkdir(parents=True)
    (run_dir / "resume.pt").touch()
    (run_dir / "run_info.json").write_text(json.dumps({"args": {
        "records": records, "val_records": val_records}}))


def test_default_corpus_is_everyday_and_bulk_is_explicit():
    assert run_train.CORPUS_FILES["everyday"] == (
        "r1/data/everyday/train.jsonl", "r1/data/everyday/val.jsonl")
    assert run_train.CORPUS_FILES["bulk"] == (
        "r1/data/bulk/train.jsonl", "r1/data/bulk/val.jsonl")


def test_resume_rejects_different_corpus_and_accepts_equivalent_paths(tmp_path):
    run_dir = tmp_path / "runs" / "soft"
    _saved_run(run_dir)
    assert "was trained on" in run_train.resume_corpus_problem(
        run_dir, *run_train.CORPUS_FILES["everyday"], tmp_path)
    assert run_train.resume_corpus_problem(
        run_dir, str(tmp_path / "r1/data/bulk/train.jsonl"),
        "r1/data/bulk/val.jsonl", tmp_path) is None
    assert "was trained on" in run_train.resume_corpus_problem(
        run_dir, "r1/data/bulk/train.jsonl", "other/val.jsonl", tmp_path)


def test_resume_fails_closed_without_provenance(tmp_path):
    run_dir = tmp_path / "runs" / "soft"
    run_dir.mkdir(parents=True)
    (run_dir / "best.mica").touch()
    assert "cannot be verified" in run_train.resume_corpus_problem(
        run_dir, *run_train.CORPUS_FILES["everyday"], tmp_path)


def test_repeated_sweep_name_preserves_previous_checkpoint(tmp_path):
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    first = run_train.unused_sweep_dir(sweep, "pilot")
    first.mkdir()
    (first / "best.mica").write_bytes(b"saved")
    second = run_train.unused_sweep_dir(sweep, "pilot")
    assert second != first
    assert not second.exists()
    assert (first / "best.mica").read_bytes() == b"saved"


def test_main_blocks_old_bulk_run_before_training(monkeypatch, tmp_path):
    for path in run_train.CORPUS_FILES["everyday"]:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("test record\n")
    _saved_run(tmp_path / "r1/runs/soft")
    messages = []
    monkeypatch.setattr(run_train, "ROOT", tmp_path)
    monkeypatch.setattr(run_train, "take_lock", lambda: True)
    monkeypatch.setattr(run_train, "start_new_run_if_asked", lambda: None)
    monkeypatch.setattr(run_train, "say", messages.append)
    monkeypatch.setattr(run_train, "run", lambda *_a, **_kw: 1 / 0)
    assert run_train.main([]) == 2
    assert any("refusing to resume" in message for message in messages)
