"""The native bucket initializer must bound score rows on long records."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_r1.fit import _bounded_bucket_sample


def test_bucket_sample_preserves_short_input():
    records = [b"a" * 10, b"b" * 20]
    assert _bounded_bucket_sample(records, 32) is records


def test_bucket_sample_caps_long_records_and_spreads_selection():
    records = [bytes([i]) * (75 + i) for i in range(100)]
    chosen = _bounded_bucket_sample(records, 1_500)
    assert 1 <= len(chosen) < len(records)
    assert sum(len(row) + 1 for row in chosen) <= 1_500
    assert chosen[0] is records[0]
    assert chosen[-1] in records[-15:]
    assert _bounded_bucket_sample(records, 1_500) == chosen
