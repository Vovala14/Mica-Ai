"""S1 changes only two hard score terms in four local MICA phases."""
from __future__ import annotations

from types import SimpleNamespace

import torch

import mica_r1.fit as fit


def test_s1_forces_spread_state_trits_and_preserves_other_rules(monkeypatch) -> None:
    # A one-page-per-phase synthetic book tests the selector mutation without
    # allocating the full Flame soft wiring tensor.
    monkeypatch.setattr(fit, "N_PHASE", 16)
    monkeypatch.setattr(fit, "PAGE_STRIDE", 1)
    monkeypatch.setattr(fit, "N_CANDIDATES", 243)
    monkeypatch.setattr(fit, "N_SCORE_TERMS", 6)
    monkeypatch.setattr(fit, "OFFSETS", (0, -1, -2, -3, -4, -6, -8))
    monkeypatch.setattr(fit.spec, "VSET_WIDTH", 6)
    monkeypatch.setattr(fit.spec, "TAPE_CHANNELS", 16)
    torch.manual_seed(12)
    model = SimpleNamespace(
        sc_nb=torch.randn(16, 243, 6, 7),
        sc_ch=torch.randn(16, 243, 6, 32),
    )
    old_nb, old_ch = model.sc_nb.clone(), model.sc_ch.clone()
    info = fit.condition_local_buckets_on_previous_word(model)
    assert info["state_trit_pairs"] == {
        7: (0, 1), 11: (2, 3), 14: (4, 0), 15: (1, 3)}
    for phase in range(16):
        for term in range(6):
            if phase in info["state_trit_pairs"] and term in (4, 5):
                trit = info["state_trit_pairs"][phase][term - 4]
                assert torch.all(model.sc_nb[phase, :, term].argmax(-1) == 0)
                assert torch.all(model.sc_ch[phase, :, term].argmax(-1) == 16 + trit)
            else:
                assert torch.equal(model.sc_nb[phase, :, term], old_nb[phase, :, term])
                assert torch.equal(model.sc_ch[phase, :, term], old_ch[phase, :, term])


def test_track_only_initialization_preserves_local_phases(monkeypatch) -> None:
    monkeypatch.setattr(fit, "N_PHASE", 16)
    monkeypatch.setattr(fit, "PAGE_STRIDE", 2)
    monkeypatch.setattr(fit, "N_CANDIDATES", 243)
    monkeypatch.setattr(fit, "N_SCORE_TERMS", 6)
    monkeypatch.setattr(fit, "N_SYMBOLS", 258)
    monkeypatch.setattr(fit, "OFFSETS", (0, -1, -2, -3, -4, -6, -8))
    monkeypatch.setattr(fit.spec, "VSET_WIDTH", 6)
    monkeypatch.setattr(fit.spec, "TAPE_CHANNELS", 16)
    patterns = torch.zeros(258, dtype=torch.long)
    patterns[list(fit.WORD_BYTES)] = 1
    monkeypatch.setattr(fit, "byte_patterns", lambda model: patterns)

    def model():
        return SimpleNamespace(
            op_code=torch.zeros(32, 243),
            op_b=torch.zeros(32, 243),
            op_v=torch.zeros(32, 243, 5),
            sc_nb=torch.randn(32, 243, 6, 7),
            sc_ch=torch.randn(32, 243, 6, 32),
            sc_co=torch.randn(32, 243, 6, 3),
            sc_bias=torch.zeros(32, 243),
        )

    only = model()
    old_nb, old_ch = only.sc_nb.clone(), only.sc_ch.clone()
    info = fit.lexical_state_rules(only, .05, decode_state_features=False)
    assert info["feature_phases"] == 0
    assert not torch.equal(only.sc_nb[:4], old_nb[:4])
    assert torch.equal(only.sc_nb[4:], old_nb[4:])
    assert torch.equal(only.sc_ch[4:], old_ch[4:])

    decoded = model()
    old_nb = decoded.sc_nb.clone()
    info = fit.lexical_state_rules(decoded, .05)
    assert info["feature_phases"] == 2
    assert not torch.equal(decoded.sc_nb[4:8], old_nb[4:8])
    assert torch.equal(decoded.sc_nb[8:], old_nb[8:])
