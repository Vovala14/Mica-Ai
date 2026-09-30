"""Small Parquet fixture checks for the bounded prose builder."""

from argparse import Namespace
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from build_prose import (build, digest, normalized, paragraph_status, records_of,
                         sampled, split_of, without_headings)


def _args(tmp_path, wiki, fineweb, **changes):
    values = dict(simple_wiki=[str(wiki)], fineweb=[str(fineweb)],
                  out=str(tmp_path / "out"), max_source_rows=20,
                  max_source_documents=20, max_records=100, max_output_mb=1,
                  max_document_bytes=10_000, max_document_records=10,
                  record_limit=256, batch_size=2, min_language_score=0.9,
                  simple_wiki_sample_rate=1.0, fineweb_sample_rate=1.0)
    values.update(changes)
    return Namespace(**values)


def _records(path):
    return [bytes.fromhex(line).decode("utf-8") for line in path.read_text().splitlines()]


def test_records_pack_sentences_and_keep_utf8_boundaries():
    paragraph = ("The café opens every morning. "
                 + "Neighbors drink tea and talk about the weather. " * 12)
    chunks = list(records_of(paragraph))
    assert len(chunks) > 1
    assert all(len(chunk.encode("utf-8")) <= 256 for chunk in chunks)
    assert " ".join(chunks) == normalized(paragraph)
    long_sentence = "The neighbor told us " + "a very detailed story " * 20 + "at dinner."
    assert list(records_of(long_sentence)) == []


def test_math_and_boilerplate_are_filtered():
    assert paragraph_status("Compute 3 + 4 = 7. The sum of sums is now clear.") == "math"
    assert paragraph_status("Please accept our cookie policy to continue reading this article.") == "boilerplate"
    assert paragraph_status("My friend stopped at the market after work. She bought fruit for dinner.") is None
    assert without_headings("Education\nTuring went to school when he was young.") == (
        "Turing went to school when he was young.")


def test_parquet_streaming_split_and_global_record_dedup(tmp_path):
    shared = ("People in the neighborhood visit the small market on Saturday morning. "
              "They talk to friends and buy fresh food.")
    first = shared + "\n\n" + "A friend brought some apples home for her family."
    # Find two different document buckets to verify record dedup across splits.
    second = None
    for n in range(1, 100):
        candidate = shared.upper() + "\n\n" + f"The next family brought some oranges home after visit number {n}."
        if split_of(digest(candidate)) != split_of(digest(first)):
            second = candidate
            break
    assert second is not None
    wiki = tmp_path / "wiki.parquet"
    pq.write_table(pa.table({"id": ["1", "2", "3", "4"],
                             "text": [first, first, second,
                                      "Compute 7 + 8 = 15. The sum of sums is 15 in this calculation."]}),
                   wiki, row_group_size=2)
    fineweb = tmp_path / "fineweb.parquet"
    pq.write_table(pa.table({"id": ["a", "b"],
                             "text": ["My sister found a little shop near the station. "
                                      "We went there together after lunch.",
                                      "The child played outside in the garden. "
                                      "Her father watched from the window."],
                             "language": ["en", "en"],
                             "language_score": [0.98, 0.70]}),
                   fineweb, row_group_size=1)
    report = build(_args(tmp_path, wiki, fineweb))
    actual = []
    for split in ("train", "val", "test"):
        actual.extend(_records(tmp_path / "out" / f"{split}.jsonl"))
    keys = [digest(record) for record in actual]
    assert len(keys) == len(set(keys))
    assert all(len(record.encode("utf-8")) <= 256 for record in actual)
    assert report["sources"][0]["counts"]["row_reject_duplicate_document"] == 1
    assert report["sources"][0]["counts"]["duplicate_records_rejected"] >= 1
    assert report["sources"][0]["counts"]["paragraph_reject_math"] == 1
    assert report["sources"][1]["counts"]["row_reject_language"] == 1
    assert report["totals"]["records"] == len(actual)
    assert (tmp_path / "out" / "manifest.json").is_file()


def test_source_row_cap_stops_before_later_rows(tmp_path):
    prose = "My neighbor opened the shop at dawn. She made fresh bread every day."
    wiki = tmp_path / "wiki.parquet"
    pq.write_table(pa.table({"id": ["1", "2", "3"],
                             "text": [prose, prose + " Thank you.", prose + " Goodbye."]}),
                   wiki, row_group_size=1)
    fineweb = tmp_path / "fineweb.parquet"
    pq.write_table(pa.table({"id": ["1"], "text": [prose],
                             "language": ["en"], "language_score": [0.99]}), fineweb)
    report = build(_args(tmp_path, wiki, fineweb, max_source_rows=1))
    assert report["sources"][0]["counts"]["rows_seen"] == 1
    assert report["sources"][0]["stop_reason"] == "row_limit"


def test_content_sampling_is_stable_and_independent_of_split():
    hashes = [digest(f"The family visited the market on day {day}.")
              for day in range(100)]
    assert [sampled(h, 0.3) for h in hashes] == [sampled(h, 0.3) for h in hashes]
    assert 15 < sum(sampled(h, 0.3) for h in hashes) < 45
    same_split_a = bytes(32)
    same_split_b = bytes(8) + bytes([255]) * 8 + bytes(16)
    assert split_of(same_split_a) == split_of(same_split_b)
    assert sampled(same_split_a, 0.3) and not sampled(same_split_b, 0.3)
