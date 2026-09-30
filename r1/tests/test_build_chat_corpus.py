"""The conversational corpus build is deterministic, keeps dialogues and
stories inside one split, and never lets an evaluation record into training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import build_chat_corpus as bc  # noqa: E402


def _dialogue(i: int, n: int = 6) -> list[str]:
    return [f"Turn {k} of dialogue {i}, about plan number {i * 7 + k}." for k in range(n)]


def _fixture(root: Path) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data = root / "data"
    ext = data / "external" / "chat"
    tm1 = ext / "taskmaster" / "TM-1-2019"
    tm2 = ext / "taskmaster" / "TM-2-2020"
    tm1.mkdir(parents=True)
    tm2.mkdir(parents=True)
    conv = [{"conversation_id": f"dlg-{i}", "utterances":
             [{"speaker": "USER", "text": t} for t in _dialogue(i)]} for i in range(400)]
    (tm1 / "self-dialogs.json").write_text(json.dumps(conv[:200]))
    (tm1 / "woz-dialogs.json").write_text(json.dumps(conv[200:]))
    for d in bc.TM2_DOMAINS:
        (tm2 / f"{d}.json").write_text("[]")
    for split, off in (("train", 1000), ("dev", 1100), ("test", 1200)):
        (ext / "sgd" / split).mkdir(parents=True)
        dl = [{"dialogue_id": f"{split}-{i}", "turns":
               [{"speaker": "USER", "utterance": t} for t in _dialogue(off + i)]}
              for i in range(60)]
        dl[0]["turns"].append({"speaker": "USER", "utterance": "Everyday sentence 3."})
        (ext / "sgd" / split / "dialogues_001.json").write_text(json.dumps(dl))
    (ext / "topicalchat").mkdir(parents=True)
    for k, f in enumerate(("train.json", "valid_freq.json", "valid_rare.json",
                           "test_freq.json", "test_rare.json")):
        convs = {f"{f}-{i}": {"content": [{"message": t}
                                          for t in _dialogue(2000 + 100 * k + i)]}
                 for i in range(40)}
        (ext / "topicalchat" / f).write_text(json.dumps(convs))
    (ext / "soda").mkdir(parents=True)
    for f, off in (("train", 3000), ("valid", 4000), ("test", 5000)):
        pq.write_table(pa.table({"dialogue": [_dialogue(off + i) for i in range(80)]}),
                       ext / "soda" / f"{f}.parquet")
    (ext / "tinystories").mkdir(parents=True)
    story = ("Lily had a red ball. She threw it very high! " * 12).strip()
    for f in ("TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"):
        body = "".join(f"{story} Story {i} ends.\n<|endoftext|>\n" for i in range(50))
        (ext / "tinystories" / f).write_text(body)
    (data / "external" / "coco").mkdir(parents=True)
    caps = [{"image_id": i, "caption": f"a dog number {i} on the grass"} for i in range(40)]
    (data / "external" / "coco" / "captions_train2017.json").write_text(
        json.dumps({"annotations": caps}))
    (data / "external" / "coco" / "captions_val2017.json").write_text(
        json.dumps({"annotations": []}))
    (data / "everyday").mkdir(parents=True)
    every = [f"A dog number {i} on the grass.".encode() for i in range(40)] + \
        [f"Everyday sentence {i}.".encode() for i in range(100)] + [b"Hello."] * 9 + \
        [b"everyday   SENTENCE 3.", b"Plain everyday line."]
    (data / "everyday" / "train.jsonl").write_text("".join(r.hex() + "\n" for r in every))
    val = [f"Everyday val sentence {i}.".encode() for i in range(300)] + [b"Everyday sentence 3."]
    (data / "everyday" / "val.jsonl").write_text("".join(r.hex() + "\n" for r in val))
    (data / "eval_clean").mkdir(parents=True)
    (data / "eval_clean" / "val1000.jsonl").write_text(b"Everyday sentence 3.".hex() + "\n")
    # the sealed test must never be opened: reading this would raise
    (data / "eval_clean" / "test1000.jsonl").write_text("sealed, not hex\n")
    (root / "runs").mkdir()
    (root / "runs" / "dev_fresh1000.jsonl").write_text(b"everyday val sentence 5.".hex() + "\n")
    return data


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    data = _fixture(tmp_path)
    monkeypatch.setattr(bc, "HERE", data)
    monkeypatch.setattr(bc, "EXT", data / "external" / "chat")
    monkeypatch.setattr(bc, "OUT", data / "chat")
    monkeypatch.setattr(bc, "CLEAN", data / "eval_clean")
    monkeypatch.setattr(bc, "SGD_FILES", {"train": 1, "dev": 1, "test": 1})
    monkeypatch.setattr(bc, "EVAL_MIX", (("soda", 10), ("taskmaster", 5),
                                         ("sgd", 5), ("topicalchat", 5)))
    monkeypatch.setattr(bc, "MAX_COPIES", 2)
    monkeypatch.setattr(bc, "SODA_TRAIN_KEEP", 100)
    monkeypatch.setattr(bc, "TINY_TRAIN_KEEP", 100)
    monkeypatch.setattr(bc, "COCO_KEEP", 25)
    monkeypatch.setattr(bc, "EXTRA_EVAL", (tmp_path / "runs" / "dev_fresh1000.jsonl",))
    monkeypatch.setattr(bc, "MIX_VAL_BYTES", 3000)
    return data


def _read(path: Path) -> list[bytes]:
    return [bytes.fromhex(l) for l in path.read_text().split()]


def test_build_is_deterministic_and_clean(corpus):
    first = bc.build()["files_sha256"]
    second = bc.build()["files_sha256"]
    assert first == second

    train = _read(corpus / "chat" / "train.jsonl")
    dev = _read(corpus / "chat" / "dev1000.jsonl")
    clean = _read(corpus / "eval_clean" / "chat_val1000.jsonl")
    assert len(dev) == len(clean) == 25
    assert not set(dev) & set(clean)
    assert not (set(dev) | set(clean) | {b"Everyday sentence 3."}) & set(train)
    assert all(len(r) <= bc.LIMIT for r in train + dev + clean)
    assert train.count(b"Hello.") == 2                    # identical records capped
    kept = sum(r.startswith(b"A dog number") for r in train)
    assert 0 < kept < 40                                  # COCO captions cut back
    assert any(b"\n" in r for r in train)                 # adjacent turn pairs
    assert any(r.startswith(b"Lily had a red ball.") for r in train)

    # version 2 hygiene: normalised copies of evaluation records, the sealed
    # test and extra development sets are gone from every split
    val = _read(corpus / "chat" / "val.jsonl")
    evals = {b"Everyday sentence 3."} | set(dev) | set(clean)
    keys = {bc.norm_key(r) for r in evals}
    for split in (train, val, _read(corpus / "chat" / "test.jsonl")):
        assert not {bc.norm_key(r) for r in split} & keys
    assert b"Plain everyday line." in train
    assert b"Everyday sentence 7." in train      # sealed-test records are not looked up
    man = json.loads((corpus / "chat" / "manifest.json").read_text())
    assert man["counts"]["train/removed_eval_normalised"] >= 1   # "everyday   SENTENCE 3."
    assert man["counts"]["val/removed_eval_normalised"] >= 1
    # one source label per record, and they add up in the manifest
    for s in ("train", "val", "test"):
        codes = (corpus / "chat" / f"{s}.src").read_bytes()
        assert len(codes) == len(_read(corpus / "chat" / f"{s}.jsonl"))
        assert max(codes) < len(bc.SOURCES)
        assert sum(v["records"] for v in man["by_source"][s].values()) == len(codes)
    codes = (corpus / "chat" / "train.src").read_bytes()
    for r, c in zip(train, codes):
        if r.startswith(b"Lily had"):
            assert bc.SOURCES[c] == "tinystories"
        if r.startswith(b"A dog number"):
            assert bc.SOURCES[c] == "coco"
        if r.startswith(b"Turn ") and b"dialogue 3" in r[:40]:
            assert bc.SOURCES[c] in ("soda", "taskmaster", "sgd", "topicalchat")


def test_story_chunks_end_on_sentences():
    text = "One two three. " * 40
    pieces = bc.chunks(text.strip())
    assert all(len(p) <= bc.LIMIT for p in pieces)
    assert all(p.endswith(b".") for p in pieces)
    assert b" ".join(pieces) == text.strip().encode()


def test_export_packs_every_record(corpus, monkeypatch):
    import gzip
    import struct
    bc.build()
    monkeypatch.setattr(bc, "EXPORT", corpus / "export")
    man = bc.export()
    got = []
    for k in range(bc.EXPORT_PARTS):
        with gzip.open(corpus / "export" / f"chat_train.part{k}.bin.gz", "rb") as f:
            while True:
                h = f.read(2)
                if not h:
                    break
                got.append(f.read(struct.unpack("<H", h)[0]))
    train = _read(corpus / "chat" / "train.jsonl")
    assert sorted(got) == sorted(train)
    assert man["chat_dev1000.bin.gz"]["records"] == 25


def test_mix_has_the_byte_shares_and_no_eval_records(corpus):
    bc.build()
    shares = {"conversation": 0.5, "everyday": 0.3, "tinystories": 0.2}
    first = bc.mix("t", shares)
    second = bc.mix("t", shares)
    assert first["files_sha256"] == second["files_sha256"]
    out = corpus / "mix" / "t"
    train = _read(out / "train.jsonl")
    codes = (out / "train.src").read_bytes()
    assert len(codes) == len(train) > 0
    got = first["splits"]["train"]["total"]["shares"]
    for g, v in shares.items():
        assert abs(got[g] - v) < 0.08, (g, got)
    # the scarcest group is used completely: every everyday training record
    # that is not an evaluation match (capped at MAX_COPIES)
    every = [r for r, c in zip(train, codes) if bc.SOURCES[c] == "everyday"]
    assert b"Plain everyday line." in every
    assert every.count(b"Hello.") == bc.MAX_COPIES
    evals = {b"Everyday sentence 3.", b"everyday val sentence 5."} | \
        set(_read(corpus / "chat" / "dev1000.jsonl")) | \
        set(_read(corpus / "eval_clean" / "chat_val1000.jsonl"))
    keys = {bc.norm_key(r) for r in evals}
    val = _read(out / "val.jsonl")
    assert val and not {bc.norm_key(r) for r in train + val} & keys
    assert sum(len(r) for r in val) <= bc.MIX_VAL_BYTES + bc.LIMIT * 3


def test_check_reports_normalised_overlaps(corpus):
    bc.build()
    path = corpus / "chat" / "train.jsonl"
    path.write_text(path.read_text() + b"EVERYDAY sentence 3.".hex() + "\n" +
                    b"everyday val sentence 5.".hex() + "\n")
    rep = bc.check()
    tr = rep["train"]
    assert tr["everyday_clean_val1000"]["eval_records_with_normalised_copy"] == 1
    assert tr["everyday_clean_val1000"]["eval_records_with_exact_copy"] == 0
    assert tr["dev_fresh1000"]["eval_records_with_exact_copy"] == 1
    assert tr["chat_dev1000"]["eval_records_with_normalised_copy"] == 0
    assert "everyday_clean_test1000" not in rep["sets"]


def test_rebuild_leaves_unchanged_eval_files_untouched(corpus):
    bc.build()
    dev, clean = corpus / "chat" / "dev1000.jsonl", corpus / "eval_clean" / "chat_val1000.jsonl"
    stamp = (dev.stat().st_mtime_ns, clean.stat().st_mtime_ns)
    import time
    time.sleep(0.05)
    bc.build()
    assert (dev.stat().st_mtime_ns, clean.stat().st_mtime_ns) == stamp
