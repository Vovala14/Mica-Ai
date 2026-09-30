"""Neighbour coverage keeps distant byte context available to MICA rules."""
from __future__ import annotations

import torch

from mica_r1.fit import _covered_neighbour_choice


def test_coverage_uses_each_allowed_lag_before_repeating() -> None:
    allowed = torch.tensor([1, 3, 6])
    choice = _covered_neighbour_choice(
        allowed, candidates=128, terms=6, self_terms=1,
        generator=torch.Generator().manual_seed(7))
    assert choice.shape == (3, 128, 6)
    assert (choice[0, :, 1:] == 0).all()
    for row in choice[1, :, 1:]:
        assert set(row.tolist()) == {0, 1, 2}
    for row in choice[2, :, 1:]:
        assert len(set(row.tolist())) == 5
        assert set(row.tolist()) <= set(range(6))
    # The two distant offsets are represented across the complete page.
    assert (choice[2, :, 1:] == 4).any()
    assert (choice[2, :, 1:] == 5).any()
