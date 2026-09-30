"""Guard the clean evaluation split against repeated captions and drift."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from checks import make_clean_eval


def test_clean_eval_deduplicates_across_splits_and_selects_deterministically(
    tmp_path, monkeypatch,
):
    def source(name, texts):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(text.encode("utf-8").hex() + "\n"
                                for text in texts), encoding="ascii")
        return path

    train = source("train", ["Hello   WORLD", "Only train"])
    val = source("val", [
        "Reserved only", "HELLO world", "Alpha", "ALPHA", "Beta",
        "Shared Across", "Gamma",
    ])
    test = source("test", [
        "hello world", "Shared across", "Delta", "DELTA", "Epsilon",
        "reserved ONLY",
    ])
    out = tmp_path / "clean"
    monkeypatch.setattr(sys, "argv", [
        "make_clean_eval.py", "--train", str(train), "--val", str(val),
        "--test", str(test), "--out", str(out), "--n", "2",
        "--val-skip", "1",
    ])

    make_clean_eval.main()
    first = {name: (out / name).read_bytes() for name in
             ("val2.jsonl", "test2.jsonl", "manifest.json")}
    make_clean_eval.main()
    assert {name: (out / name).read_bytes() for name in first} == first

    def decoded(name):
        return [bytes.fromhex(line).decode("utf-8") for line in
                (out / name).read_text(encoding="ascii").splitlines()]

    assert decoded("val2.jsonl") == ["Alpha", "Shared Across"]
    assert decoded("test2.jsonl") == ["Delta", "Epsilon"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["val"] == {
        "excluded": 1, "within_split_duplicates": 1, "eligible": 4,
    }
    assert manifest["test"] == {
        "excluded": 3, "within_split_duplicates": 1, "eligible": 2,
    }
