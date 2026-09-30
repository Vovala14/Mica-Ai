"""Collapse a trained SoftMica into the integer MICA file, and measure the cost.

This is the step the whole gradient-training route depends on. The relaxation
trains ~48 million real-valued parameters; the model that ships is 4.25 million
bytes of integers. Every softmax collapses to its argmax, every real number
rounds, and the real-valued field becomes int8. None of that is free.

The number this module exists to produce is the DISCRETISATION GAP: the loss of
the integer model minus the loss of the soft model it came from, on the same
held-out records. A small gap means gradient training works for MICA. A large
one means the relaxation trains a machine that the integer machine cannot
imitate, and no amount of further training helps.

It also reports the ACTIVATION GAP separately, because the two are often
confused. The relaxation updates every cell every tick; the integer machine
updates only cells the change-trigger woke. Training dense and running sparse
is an assumption, and this splits its cost out from the rounding cost so the
two can be judged on their own.
"""

from __future__ import annotations

import numpy as np
import torch

from . import spec
from .engine import Model
from .spec import SAT_MIN, SAT_MAX


def _argmax(t: torch.Tensor) -> np.ndarray:
    return t.detach().argmax(-1).cpu().numpy()


def _round_clip(t: torch.Tensor, lo: int, hi: int, dtype) -> np.ndarray:
    return np.clip(np.rint(t.detach().cpu().numpy()), lo, hi).astype(dtype)


def to_integer(sm) -> Model:
    """Argmax every selector, round every scalar. No tuning, no calibration."""
    tern = lambda x: (_argmax(x).astype(np.int8) - 1)        # {0,1,2} -> {-1,0,1}
    cell_dtype = np.uint16 if spec.WIDE_CELLS else np.uint8
    # selectors go through the same masks training used (tape extensions)
    lg = sm.logits_of
    def expand_symbol_rows(a: np.ndarray) -> np.ndarray:
        if getattr(sm, "compact_selectors", False):
            if a.shape[0] != 1:
                raise ValueError("compact symbol selectors must have one shared row")
            return np.broadcast_to(a, (spec.N_SYMBOLS,) + a.shape[1:]).copy()
        return a

    def symbols(t: torch.Tensor) -> np.ndarray:
        return expand_symbol_rows(_argmax(t))
    pr_co = (_round_clip(sm.pr_w, -127, 127, np.int8) if spec.WIDE_PROBE_COEF
             else expand_symbol_rows(tern(sm.pr_co)))
    op_v = (_round_clip(sm.op_v, SAT_MIN, SAT_MAX, np.int8)
            if spec.VSET_WIDTH > 1 else None)
    return Model(
        op_v=op_v,
        inj_cell=symbols(lg("inj_cell")).astype(cell_dtype),
        inj_chan=symbols(lg("inj_chan")).astype(np.uint8),
        inj_delta=_round_clip(sm.inj_delta, SAT_MIN, SAT_MAX, np.int8),
        sc_nb=_argmax(sm.sc_nb).astype(np.uint8),
        sc_ch=_argmax(sm.sc_ch).astype(np.uint8),
        sc_co=tern(sm.sc_co),
        sc_bias=_round_clip(sm.sc_bias, -32768, 32767, np.int16),
        op_code=_argmax(sm.op_code).astype(np.uint8),
        op_d=_argmax(lg("op_d")).astype(np.uint8),
        op_n=_argmax(sm.op_n).astype(np.uint8),
        op_c=_argmax(sm.op_c).astype(np.uint8),
        op_a=tern(sm.op_a),
        op_b=_round_clip(sm.op_b, SAT_MIN, SAT_MAX, np.int8),
        op_u=_argmax(lg("op_u")).astype(np.uint8),
        pr_cell=symbols(lg("pr_cell")).astype(cell_dtype),
        pr_chan=symbols(sm.pr_chan).astype(np.uint8),
        pr_co=pr_co,
        pr_bias=_round_clip(sm.pr_bias, -32768, 32767, np.int16),
    )


def selector_confidence(sm) -> dict:
    """How peaked is each selector? A selector still spread across its choices
    is one the argmax will misrepresent, so this predicts the gap before you
    pay to measure it."""
    import torch.nn.functional as Fn
    out = {}
    for name in ("sc_nb", "sc_ch", "sc_co", "op_code", "op_d", "op_n", "op_c",
                 "op_a", "op_u", "inj_cell", "inj_chan", "pr_cell", "pr_chan",
                 "pr_co"):
        if name == "pr_co" and spec.WIDE_PROBE_COEF:
            continue                      # int8 coefficients, not a selector
        p = Fn.softmax(sm.logits_of(name).detach() / sm.tau, dim=-1)
        out[name] = float(p.max(-1).values.mean())
    return out


def measure_gap(sm, records, device="cpu", full_field_too: bool = True,
                match_ticks: bool = True, chunk: int = 4) -> dict:
    """Soft loss, integer loss, and the two gaps, on the same records."""
    from .batch import ModelStack, evaluate
    from . import engine, batch as _batch

    # The relaxation runs a FIXED number of ticks per symbol; the integer
    # machine runs up to MAX_TICKS and stops when activity dies. Comparing a
    # 2-tick model against a 12-tick one measures the tick count, not the
    # rounding, so match them unless asked not to.

    B = float(np.log(2))
    # In chunks, never all at once: on the GPU every record in a batch costs
    # its own copy of each tick's candidate tensors.
    tot_nats, tot_n = 0.0, 0
    for i in range(0, len(records), max(1, chunk)):
        part = records[i:i + chunk]
        L = max(len(r) for r in part)
        batch = torch.zeros(len(part), L, dtype=torch.long, device=device)
        lengths = torch.tensor([len(r) for r in part], device=device)
        for j, r in enumerate(part):
            batch[j, :len(r)] = torch.tensor(list(r), device=device)
        with torch.no_grad():
            loss, _ = sm.sequence_loss(batch, lengths)
        # sequence_loss returns the mean over records; weight it back up
        tot_nats += float(loss) * len(part)
        tot_n += len(part)
    soft = tot_nats / tot_n / B

    m = to_integer(sm)
    saved = engine.FULL_FIELD
    # batch.py binds MAX_TICKS at import, so the batched evaluator reads its
    # own module-level name, not the engine's.
    saved_ticks = (engine.TICKS, _batch.MAX_TICKS)
    if match_ticks:
        engine.TICKS = _batch.MAX_TICKS = sm.ticks
    out = {"soft_bits": round(soft, 4),
           "ticks": sm.ticks,
           "selector_confidence": {k: round(v, 3)
                                   for k, v in selector_confidence(sm).items()}}
    try:
        if full_field_too:
            # Same activation rule as training: every cell, every tick. This
            # isolates the cost of ROUNDING from the cost of going sparse.
            engine.FULL_FIELD = True
            l, _u, _ = evaluate(ModelStack([m], score_backend="c"), records)
            out["integer_dense_bits"] = round(float(l[0]) / B, 4)
        engine.FULL_FIELD = False
        l, u, _ = evaluate(ModelStack([m], score_backend="c"), records)
        out["integer_bits"] = round(float(l[0]) / B, 4)
        out["updates_per_symbol"] = round(float(u[0]), 1)
    finally:
        engine.FULL_FIELD = saved
        engine.TICKS, _batch.MAX_TICKS = saved_ticks

    out["discretisation_gap"] = round(
        out.get("integer_dense_bits", out["integer_bits"]) - soft, 4)
    if "integer_dense_bits" in out:
        out["activation_gap"] = round(
            out["integer_bits"] - out["integer_dense_bits"], 4)
    return out
