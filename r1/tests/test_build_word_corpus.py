"""Tests for r1/data/build_word_corpus.py (word-symbol corpora for Flame-W)."""
from __future__ import annotations

import gzip
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import build_word_corpus as W  # noqa: E402


def _hex(lines):
    return "".join(t.encode("utf-8").hex() + "\n" for t in lines)


def test_alphabet_layout():
    assert (W.SEP, W.OOV0, W.FIRST_WORD) == (0, 1, 257)
    assert (W.BOS, W.EOS, W.N_SYMBOLS) == (16382, 16383, 16384)
    assert W.N_VOCAB == 16125 and W.MAX_TOKENS == 64


def test_tokenize():
    text = "I don’t know.\nOK, see you at 5:30! It's Bob's_car “fine”"
    assert W.tokenize(text) == [
        "i", "don't", "know", ".", "<sep>", "ok", ",", "see", "you", "at", "5", ":",
        "30", "!", "it's", "bob's", "_", "car", '"', "fine", '"']
    assert W.surface_tokens("Paris\r\nis") == ["Paris", "<sep>", "is"]
    assert W.tokenize("  \t ") == []


def test_pack_is_little_endian_and_round_trips():
    assert W.pack([1, 256, 16383]) == "01000001ff3f"
    ids = [0, 5, 257, 16382, 16383, 65535]
    assert W.unpack(W.pack(ids)) == ids


def test_oov_ids_are_stable_and_in_range():
    a = W.oov_id("zyzzyva")
    assert a == W.oov_id("zyzzyva") and W.OOV0 <= a < W.FIRST_WORD
    assert {W.oov_id(f"x{i}") for i in range(2000)} <= set(range(1, 257))


def test_vocab_order_surface_and_encoding():
    cnt = {"the": 5, "paris": 3, "i": 3, "a": 3, ".": 9}
    mid = {"the": 4, "paris": 3, "i": 3, "a": 2, ".": 9}
    caps = {("paris", "Paris"): 3, ("i", "I"): 3, ("the", "The"): 2, ("a", "A"): 1}
    v = W.build_vocab(cnt, mid, caps)
    assert v.tokens == [".", "the", "a", "i", "paris"]   # count, then text
    assert v.surface == [".", "the", "a", "I", "Paris"]
    ids = v.encode(["the", "<sep>", "paris", "unseen"])
    assert ids[:3] == [W.FIRST_WORD + 1, W.SEP, W.FIRST_WORD + 4]
    assert W.OOV0 <= ids[3] < W.FIRST_WORD
    assert v.decode(ids + [W.BOS, W.EOS])[:3] == ["the", "<sep>", "paris"]
    assert v.decode([W.BOS, W.EOS]) == ["<bos>", "<eos>"]
    assert len(v.encode(["a"] * 100)) == W.MAX_TOKENS


def _tree(tmp: Path):
    mix = tmp / "mix" / "v02a"
    mix.mkdir(parents=True)
    train = ["Hello there, how are you?\nI am fine, thanks.",
             "The dog runs in Paris.", "I like the dog.", "   ",
             " ".join(["word"] * 80), "Zyzzyva appears once."] * 3
    (mix / "train.jsonl").write_text(_hex(train))
    (mix / "val.jsonl").write_text(_hex(["How are you?", "The cat sleeps."]))
    (mix / "manifest.json").write_text("{}")
    ev = tmp / "eval_src"
    ev.mkdir()
    (ev / "dev.jsonl").write_text(_hex(["See you in Paris!", "Qwerty uiop."]))
    # the sealed test holds non-hex text: opening it as records would fail
    (ev / "test1000.jsonl").write_text("NOT HEX\n")
    return mix, tmp / "word", {"dev": ev / "dev.jsonl", "gone": ev / "missing.jsonl"}


def test_build_is_deterministic_and_complete(tmp_path):
    mix, out, sets = _tree(tmp_path)
    m1 = W.build(mix, out, sets, export_parts=2, log=lambda s: None)
    first = {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    m2 = W.build(mix, out, sets, export_parts=2, log=lambda s: None)
    second = {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    assert first == second and m1 == m2
    assert m1["missing_eval_sets"] == ["gone"]
    tr = m1["files"]["v02a/train.jsonl"]
    assert tr["records"] == 15 and tr["records_empty"] == 3 and tr["records_cut"] == 3
    v = W.Vocab.load(out / "vocab.json")
    assert v.surface[v.tokens.index("paris")] == "Paris"
    assert v.surface[v.tokens.index("i")] == "i"      # only ever sentence-initial here
    lines = (out / "v02a" / "train.jsonl").read_text().split()
    assert all(len(W.unpack(l)) <= W.MAX_TOKENS for l in lines)
    assert v.decode(W.unpack(lines[0]))[:6] == ["hello", "there", ",", "how", "are", "you"]
    assert "<sep>" in v.decode(W.unpack(lines[0]))
    # the shards hold exactly the train records, in order
    recs = []
    for k in range(2):
        with gzip.open(out / "v02a" / f"train.part{k}.u16.gz", "rb") as f:
            data = f.read()
        i = 0
        while i < len(data):
            n = struct.unpack_from("<H", data, i)[0]
            recs.append(W.pack(list(struct.unpack_from(f"<{n // 2}H", data, i + 2))))
            i += 2 + n
    assert recs == lines
    dev = (out / "eval" / "dev.jsonl").read_text().split()
    d0 = v.decode(W.unpack(dev[0]))            # "see" and "!" are not in train
    assert d0[1:4] == ["you", "in", "paris"] and all(d0[i].startswith("<oov:") for i in (0, 4))
    assert m1["files"]["eval/dev.jsonl"]["oov_rate"] > 0     # qwerty, uiop
    json.loads((out / "manifest.json").read_text())


def test_sealed_test_is_refused(tmp_path):
    p = tmp_path / "test1000.jsonl"
    p.write_text("NOT HEX\n")
    with pytest.raises(ValueError, match="sealed"):
        list(W.hex_lines(p))
    assert "test1000" not in json.dumps({k: str(v) for k, v in W.EVAL_SETS.items()})


def test_check_only_writes_nothing(tmp_path):
    mix, out, sets = _tree(tmp_path)
    rep = W.build(mix, out, sets, check_only=True, log=lambda s: None)
    assert rep["train_records"] == 18 and not out.exists()
