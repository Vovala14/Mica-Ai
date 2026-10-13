import json

import pytest

from r1.data.mix_site_corrections import mix, sha, targets
from r1.data.ingest_site_logs import completed_word


def test_long_completion_corrections():
    assert completed_word("Thank you for bei", "ng") == "Thank you for being"
    assert completed_word("Thank you for bei", "being") == "Thank you for being"
    assert completed_word("Thank you for BEI", "being") == "Thank you for BEIng"
    assert completed_word("Thank you for b", "being") == "Thank you for being"
    assert completed_word("Thank you for b", "eing") == "Thank you for being"


@pytest.mark.parametrize("model", ["ember", "flamew"])
def test_bounded_deduplicated_deterministic_mix(tmp_path, model):
    site = tmp_path / "site"
    folder = site / model
    folder.mkdir(parents=True)
    base = tmp_path / "base.jsonl"
    base.write_text("0000\n" * 100)
    # Base duplicate, validation overlap, held-out overlap, repeated correction,
    # and a record too long for the target budget.
    train = folder / "train.jsonl"
    train.write_text("0000\n0100\n0200\n0300\n0300\n0400\n0500\n0600\n" + "ff" * 1000 + "\n")
    val = folder / "val.jsonl"
    val.write_text("0100\n")
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text("0200\n")
    manifest = {"corrections_only": True, "files": {
        f"{model}/{path.name}": {"sha256": sha(path)} for path in (train, val)}}
    (site / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "mixed.jsonl"
    report = mix(base, site, model, out, [heldout])
    lines = out.read_text().splitlines()
    corrections = [line for line in lines if line != "0000"]
    assert lines.count("0000") == 100
    assert set(corrections) == {"0300", "0400", "0500", "0600"}
    assert len(corrections) == len(set(corrections))
    assert report["correction_records"] / len(lines) <= 0.05
    total_targets = sum(targets(line, model) for line in lines)
    assert report["correction_targets"] / total_targets <= 0.05
    other = tmp_path / "other.jsonl"
    mix(base, site, model, other, [heldout])
    assert out.read_bytes() == other.read_bytes()
    with pytest.raises(ValueError, match="at most"):
        mix(base, site, model, out, fraction="0.5")
    with pytest.raises(ValueError, match="overwrite"):
        mix(base, site, model, base)
    manifest["corrections_only"] = False
    (site / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="corrections-only"):
        mix(base, site, model, out)
