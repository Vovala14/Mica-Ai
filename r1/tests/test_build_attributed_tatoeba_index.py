"""Source linkage and train/held-out boundary checks for Tatoeba candidates."""

import bz2
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from build_attributed_tatoeba_index import build, split_of_tatoeba_id
from mica_r1.retrieval_suggest import SentenceIndex


def _id_for(split):
    return next(str(i) for i in range(1, 1000)
                if split_of_tatoeba_id(str(i)) == split)


def test_tatoeba_index_requires_actual_train_text_and_excludes_heldout(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    accepted = "The quiet room filled with music as friends arrived."
    val_leak = "The family walked together through the nearby park."
    test_leak = "A young bird sang beside the open kitchen window."
    wrong_id_split = "The dog waited patiently beside the old front door."
    not_in_train = "The people visited a new market after work today."
    train = corpus / "train.txt"
    train.write_text("\n".join((accepted, val_leak, test_leak,
                                wrong_id_split)) + "\n", encoding="utf-8")
    (corpus / "val.txt").write_text(val_leak + "\n", encoding="utf-8")
    (corpus / "test.txt").write_text(test_leak + "\n", encoding="utf-8")
    source = tmp_path / "eng_sentences.tsv.bz2"
    rows = [(_id_for("train"), accepted),
            (str(int(_id_for("train")) + 1000), val_leak),
            (str(int(_id_for("train")) + 2000), test_leak),
            (_id_for("val"), wrong_id_split),
            (str(int(_id_for("train")) + 3000), not_in_train)]
    # Ensure all of the irrelevant source rows would otherwise be train IDs.
    rows[1] = (next(str(i) for i in range(1000, 2000)
                    if split_of_tatoeba_id(str(i)) == "train"), val_leak)
    rows[2] = (next(str(i) for i in range(2000, 3000)
                    if split_of_tatoeba_id(str(i)) == "train"), test_leak)
    rows[4] = (next(str(i) for i in range(3000, 4000)
                    if split_of_tatoeba_id(str(i)) == "train"), not_in_train)
    with bz2.open(source, "wt", encoding="utf-8") as stream:
        for sid, text in rows:
            stream.write(f"{sid}\teng\t{text}\n")

    out = tmp_path / "index"
    manifest = build(train, source, out, 10)
    loaded = SentenceIndex.load(out)
    assert manifest["publication_ready"] is False
    assert manifest["selected_sentences"] == 1
    assert loaded.sentences[0].text == accepted
    assert loaded.sentences[0].train_line == 1
    assert loaded.sentences[0].source_url.endswith("/" + rows[0][0])
    assert loaded.sentences[0].source_license == "CC BY 2.0 FR"
    attribution = json.loads((out / "attribution.jsonl").read_text())
    assert attribution["source_id"] == int(rows[0][0])
    assert attribution["source_author"] is None
    assert manifest["counts"]["heldout_rejected"] == 2
    assert manifest["counts"]["not_in_train_text"] == 1


def test_builder_refuses_missing_eval_files_and_non_train_source(tmp_path):
    train = tmp_path / "train.txt"
    train.write_text("The quiet room filled with music as friends arrived.\n")
    source = tmp_path / "eng.tsv.bz2"
    with bz2.open(source, "wt") as stream:
        stream.write("")
    with pytest.raises(FileNotFoundError):
        build(train, source, tmp_path / "out", 10)
    (tmp_path / "val.txt").write_text("")
    (tmp_path / "test.txt").write_text("")
    with pytest.raises(ValueError, match="train.txt"):
        build(tmp_path / "val.txt", source, tmp_path / "out", 10)


def test_detailed_export_joins_owner_and_rejects_revised_text(tmp_path):
    accepted = "The quiet room filled with music as friends arrived."
    revised = "The people visited a new market after work today."
    (tmp_path / "train.txt").write_text(accepted + "\n" + revised + "\n")
    (tmp_path / "val.txt").write_text("")
    (tmp_path / "test.txt").write_text("")
    sid1 = _id_for("train")
    sid2 = next(str(i) for i in range(int(sid1) + 1, 1000)
                if split_of_tatoeba_id(str(i)) == "train")
    basic = tmp_path / "basic.tsv.bz2"
    detailed = tmp_path / "detailed.tsv.bz2"
    with bz2.open(basic, "wt", encoding="utf-8") as stream:
        stream.write(f"{sid1}\teng\t{accepted}\n{sid2}\teng\t{revised}\n")
    with bz2.open(detailed, "wt", encoding="utf-8") as stream:
        stream.write(f"{sid1}\teng\t{accepted}\tAlice\t2025-01-01\t2025-01-02\n")
        stream.write(f"{sid2}\teng\tThe text was revised by its owner.\tBob\t2025-01-01\t2025-01-02\n")
    out = tmp_path / "index"
    manifest = build(tmp_path / "train.txt", basic, out, 10, detailed)
    assert manifest["selected_sentences"] == 1
    assert manifest["counts"]["detailed_text_mismatch"] == 1
    assert manifest["counts"]["detailed_exact_matches"] == 1
    assert manifest["source_detailed_sha256"]
    assert SentenceIndex.load(out).sentences[0].text == accepted
    attr = json.loads((out / "attribution.jsonl").read_text())
    assert attr["source_owner"] == "Alice"
    assert attr["source_author"] is None
