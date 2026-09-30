"""Discrete, gradient-guided search over existing MICA scoring selectors.

Training only: the exported integer model and inference engine are unchanged.
For a small number of actively used score terms, move the hard argmax to the
alternative with the best first-order loss estimate. Exact integer validation
still decides which checkpoint is kept.
"""
from __future__ import annotations

import torch

from . import spec


@torch.no_grad()
def project_score_selectors(model, *, per_phase: int = 2,
                            margin: float = 1.0) -> dict[str, int]:
    """Force up to ``per_phase`` hard changes for each scoring selector.

    Phase 0 and 1 retain the recurrent state's structure: phase 0 reads the
    previous cell, phase 1 reads its own cell, and both choose among the six
    existing state channels. Other phases may select any existing neighbour
    offset, but read only the protected byte tape. Coefficients can change in
    every phase. A group is changed only when the current gradient predicts a
    positive first-order gain. No operation or geometry is added.
    """
    if per_phase < 0 or margin <= 0:
        raise ValueError("per_phase must be nonnegative and margin positive")
    counts = {name: 0 for name in ("sc_nb", "sc_ch", "sc_co")}
    if per_phase == 0:
        return counts
    terms_per_phase = spec.PAGE_STRIDE * spec.N_CANDIDATES * spec.N_SCORE_TERMS
    device = model.sc_nb.device
    for name in counts:
        param = getattr(model, name)
        if param.grad is None:
            continue
        options = param.shape[-1]
        logits = param.reshape(spec.N_PHASE, terms_per_phase, options)
        grads = param.grad.reshape_as(logits)
        for phase in range(spec.N_PHASE):
            if name == "sc_nb" and phase < 2:
                continue
            if name == "sc_ch":
                choices = (range(spec.TAPE_CHANNELS,
                                 spec.TAPE_CHANNELS + spec.VSET_WIDTH)
                           if phase < 2 else range(spec.TAPE_CHANNELS))
            else:
                choices = range(options)
            allowed = torch.tensor(list(choices), device=device)
            if allowed.numel() < 2:
                continue
            weights = logits[phase]
            gradient = grads[phase]
            current = weights.argmax(dim=1)
            alt = allowed[gradient[:, allowed].argmin(dim=1)]
            row = torch.arange(terms_per_phase, device=device)
            gain = gradient[row, current] - gradient[row, alt]
            valid = (alt != current) & torch.isfinite(gain) & (gain > 0)
            if not bool(valid.any()):
                continue
            eligible = torch.nonzero(valid, as_tuple=True)[0]
            take = min(per_phase, eligible.numel())
            chosen = eligible[gain[eligible].topk(take).indices]
            weights[chosen, alt[chosen]] = weights[chosen, current[chosen]] + margin
            counts[name] += take
    return counts
