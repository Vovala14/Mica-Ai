"""Selection checks for the separate prose checkpoint and audit records."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks.make_prose_eval import build


def _source(corpus, name, records):
    lines = [record.encode("utf-8").hex().encode("ascii") + b"\n"
             for record in records]
    # The prose builder hashes LF lines before Windows text mode may emit CRLF.
    (corpus / f"{name}.jsonl").write_bytes(b"".join(lines).replace(b"\n", b"\r\n"))
    return {"records": len(records),
            "jsonl_sha256": hashlib.sha256(b"".join(lines)).hexdigest()}


def _read(path):
    return [bytes.fromhex(line) for line in path.read_text().splitlines()]


def test_prose_eval_uses_whole_val_split_and_separate_reproducible_samples(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    val = [f"wiki validation {i:03d}" for i in range(15)] + [
        f"fineweb validation {i:03d}" for i in range(15)]
    test = [f"test prose {i:03d}" for i in range(25)]
    inputs = {"val": _source(corpus, "val", val),
              "test": _source(corpus, "test", test)}
    (corpus / "manifest.json").write_text(json.dumps({"splits": inputs}))

    first = build(corpus, tmp_path / "eval_a", seed=17, checkpoint_n=8,
                  val_audit_n=8, test_audit_n=8)
    second = build(corpus, tmp_path / "eval_b", seed=17, checkpoint_n=8,
                   val_audit_n=8, test_audit_n=8)
    assert first["inputs"] == second["inputs"]
    groups = {}
    for name in ("checkpoint_val", "audit_val", "audit_test"):
        left, right = first["outputs"][name], second["outputs"][name]
        assert left["source_indices"] == right["source_indices"]
        assert left["file_sha256"] == right["file_sha256"]
        rows = _read(tmp_path / "eval_a" / left["file"])
        assert len(rows) == 8
        assert left["record_sha256"] == [hashlib.sha256(row).hexdigest()
                                          for row in rows]
        groups[name] = set(rows)
    assert not groups["checkpoint_val"] & groups["audit_val"]
    assert not groups["checkpoint_val"] & groups["audit_test"]
    assert not groups["audit_val"] & groups["audit_test"]
    assert any(row.startswith(b"wiki") for row in groups["checkpoint_val"])
    assert any(row.startswith(b"fineweb") for row in groups["checkpoint_val"])

    # A partial or changed builder output must fail before writing eval files.
    test_file = corpus / "test.jsonl"
    test_file.write_bytes(test_file.read_bytes().replace(b"74657374", b"74657375", 1))
    with pytest.raises(ValueError, match="content hash"):
        build(corpus, tmp_path / "eval_changed", seed=17, checkpoint_n=8,
              val_audit_n=8, test_audit_n=8)
    assert not (tmp_path / "eval_changed").exists()
