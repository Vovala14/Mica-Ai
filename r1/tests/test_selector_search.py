from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mica_r1 import selector_search, spec


def test_projection_changes_hard_choices_within_existing_state_layout(monkeypatch):
    for name, value in {"N_PHASE": 3, "PAGE_STRIDE": 1,
                        "N_CANDIDATES": 2, "N_SCORE_TERMS": 2,
                        "TAPE_CHANNELS": 2, "VSET_WIDTH": 2}.items():
        monkeypatch.setattr(spec, name, value)
    tensors = {}
    for name, options in (("sc_nb", 3), ("sc_ch", 4), ("sc_co", 3)):
        data = torch.zeros(3, 2, 2, options)
        current = (2 if name == "sc_ch" else 0)
        if name == "sc_ch":
            data[2, :, :, 2] = 0
            data[2, :, :, 0] = 2
        data[..., current] = 2
        param = torch.nn.Parameter(data)
        grad = torch.zeros_like(data)
        preferred = (3 if name == "sc_ch" else 1)
        grad[..., preferred] = -1
        if name == "sc_ch":
            grad[2, :, :, 1] = -2  # phase 2 may use tape only
        param.grad = grad
        tensors[name] = param
    model = SimpleNamespace(**tensors)
    moved = selector_search.project_score_selectors(model, per_phase=1,
                                                     margin=1)
    assert moved == {"sc_nb": 1, "sc_ch": 3, "sc_co": 3}
    ch = model.sc_ch.argmax(-1)
    assert torch.all((ch[:2] >= 2) & (ch[:2] <= 3))
    assert torch.all(ch[2] < 2)
    assert (model.sc_nb.argmax(-1)[:2] == 0).all()


def test_projection_rejects_invalid_budget():
    with pytest.raises(ValueError):
        selector_search.project_score_selectors(None, per_phase=-1)
