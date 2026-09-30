"""Direct discrete mutation search — section 13, exactly as specified.

"R1 uses direct discrete mutation search so the first training method executes
 exactly the same hard transitions used at deployment."

The RNG, the mutation kinds, the acceptance rule and the tie-breaks all follow
the text. Nothing here is differentiable and nothing is relaxed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import spec
from .engine import Model, random_model
from .batch import ModelStack, evaluate, objective

M32 = 0xFFFFFFFF


class XorShift32:
    """Section 13: x ^= x<<13; x ^= x>>17; x ^= x<<5, 32-bit after each step."""

    def __init__(self, seed: int):
        if seed == 0:
            raise ValueError("zero seeds are forbidden")
        self.x = seed & M32

    def next_u32(self) -> int:
        x = self.x
        x ^= (x << 13) & M32
        x ^= (x >> 17)
        x ^= (x << 5) & M32
        self.x = x & M32
        return self.x

    def below(self, n: int) -> int:
        """Unbiased index in [0, n), by the rejection rule in section 13."""
        if n <= 0:
            raise ValueError(n)
        limit = M32 - (M32 % n)
        while True:
            y = self.next_u32() - 1
            if y < limit:
                return y % n

    def state(self) -> int:
        return self.x


# ---------------------------------------------------------------------------
# mutation
# ---------------------------------------------------------------------------
# "For a categorical field, resample uniformly from legal alternatives
#  excluding its current value."
CAT = "categorical"
STEP = "step"

SIGNATURE_FIELDS = [("inj_cell", CAT, spec.N_CELLS),
                    ("inj_chan", CAT, spec.N_CHANNELS),
                    ("inj_delta", STEP, 127)]

CANDIDATE_FIELDS = (
    [("sc_nb", CAT, spec.N_SELECTORS, t) for t in range(spec.N_SCORE_TERMS)]
    + [("sc_ch", CAT, spec.N_CHANNELS, t) for t in range(spec.N_SCORE_TERMS)]
    + [("sc_co", CAT, 3, t) for t in range(spec.N_SCORE_TERMS)]
    + [("sc_bias", STEP, 32767, None),
       ("op_code", CAT, spec.N_OPCODES, None),
       ("op_d", CAT, spec.N_CHANNELS, None),
       ("op_n", CAT, spec.N_SELECTORS, None),
       ("op_c", CAT, spec.N_CHANNELS, None),
       ("op_a", CAT, 3, None),
       ("op_b", STEP, 127, None),
       ("op_u", CAT, spec.N_CHANNELS, None)]
)

PROBE_FIELDS = ([("pr_cell", CAT, spec.N_CELLS, t) for t in range(spec.N_PROBE)]
                + [("pr_chan", CAT, spec.N_CHANNELS, t) for t in range(spec.N_PROBE)]
                + [("pr_co", CAT, 3, t) for t in range(spec.N_PROBE)]
                + [("pr_bias", STEP, 32767, None)])


def _resample_excluding(rng: XorShift32, current: int, n: int, offset: int = 0) -> int:
    """Uniform over the n legal values excluding the current one."""
    j = rng.below(n - 1)
    cur = current - offset
    return offset + (j if j < cur else j + 1)


def _step(rng: XorShift32, current: int, limit: int) -> int:
    """+1 or -1 equally; clamp; reverse the step if clamping is a no-op."""
    d = 1 if rng.below(2) else -1
    v = current + d
    if v > limit or v < -limit:
        v = current - d
        if v > limit or v < -limit:
            return current
    return v


FROZEN_INSTRUCTION_FIELDS = {"op_code", "op_d", "op_n", "op_c", "op_a",
                             "op_b", "op_u"}


def mutate_once(m: Model, rng: XorShift32, freeze_instructions: bool = False,
                kinds: tuple = (0, 1, 2)) -> str:
    """One mutation, chosen exactly as section 13 describes.

    ``freeze_instructions`` is the section 16 ablation: scoring, injection and
    probes still learn, but the rewrite programs stay at initialisation.

    ``kinds`` restricts which of the three mutation kinds may be drawn. The
    default consumes the random stream exactly as section 13 does, because
    kinds[j] == j for the full tuple. Restricting it to (0, 1) is the liveness
    probe: those two kinds reach the output only through the cell field, so if
    none of them can change the loss, the readout is disconnected.
    """
    kind = kinds[rng.below(len(kinds))]   # signature, candidate, probe

    if kind == 0:
        sym = rng.below(spec.N_SYMBOLS)
        ent = rng.below(spec.N_INJECT)
        name, mode, lim = SIGNATURE_FIELDS[rng.below(len(SIGNATURE_FIELDS))]
        arr = getattr(m, name)
        cur = int(arr[sym, ent])
        arr[sym, ent] = (_step(rng, cur, lim) if mode == STEP
                         else _resample_excluding(rng, cur, lim))
        return name

    if kind == 1:
        page = rng.below(spec.N_PAGES)
        cand = rng.below(spec.N_CANDIDATES)
        name, mode, n, term = CANDIDATE_FIELDS[rng.below(len(CANDIDATE_FIELDS))]
        if freeze_instructions and name in FROZEN_INSTRUCTION_FIELDS:
            return name + "(frozen)"
        arr = getattr(m, name)
        idx = (page, cand) if term is None else (page, cand, term)
        cur = int(arr[idx])
        if mode == STEP:
            arr[idx] = _step(rng, cur, n)
        elif n == 3:                       # a coefficient in {-1, 0, +1}
            arr[idx] = _resample_excluding(rng, cur, 3, offset=-1)
        else:
            arr[idx] = _resample_excluding(rng, cur, n)
        return name

    sym = rng.below(spec.N_SYMBOLS)
    name, mode, n, term = PROBE_FIELDS[rng.below(len(PROBE_FIELDS))]
    arr = getattr(m, name)
    idx = (sym,) if term is None else (sym, term)
    cur = int(arr[idx])
    if mode == STEP:
        arr[idx] = _step(rng, cur, n)
    elif n == 3:
        arr[idx] = _resample_excluding(rng, cur, 3, offset=-1)
    else:
        arr[idx] = _resample_excluding(rng, cur, n)
    return name


def make_child(parent: Model, rng: XorShift32, n_mutations: int = 8,
               freeze_instructions: bool = False,
               kinds: tuple = (0, 1, 2)) -> Model:
    child = parent.copy()
    for _ in range(n_mutations):
        mutate_once(child, rng, freeze_instructions, kinds)
    return child


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------
@dataclass
class SearchConfig:
    seed: int = 1
    rounds: int = 100
    batch_records: int = 32
    children: int = 16
    mutations: int = 8
    accept_margin: float = 1e-6      # "beats the incumbent by at least 0.000001"
    activity_weight: float = 0.01


def search(records: list[bytes], cfg: SearchConfig, log_every: int = 10,
           on_round=None, validation: list[bytes] | None = None,
           validate_every: int = 0):
    """Run `cfg.rounds` rounds and return (incumbent, history)."""
    rng = XorShift32(cfg.seed)
    incumbent = random_model(cfg.seed)

    order = np.arange(len(records))
    perm_rng = np.random.default_rng(cfg.seed)
    perm_rng.shuffle(order)
    cursor = 0

    history = []
    accepted = 0
    t0 = time.time()

    for rnd in range(1, cfg.rounds + 1):
        # "Choose the next 32 training records from a seeded shuffled cyclic order."
        take = []
        for _ in range(min(cfg.batch_records, len(records))):
            take.append(order[cursor % len(order)])
            cursor += 1
        batch = [records[i] for i in take]

        children = [make_child(incumbent, rng, cfg.mutations)
                    for _ in range(cfg.children)]
        stack = ModelStack([incumbent] + children)
        loss, upd, _ = evaluate(stack, batch)
        J = objective(loss, upd, cfg.activity_weight)

        parent_J = float(J[0])
        child_J = J[1:]
        best = int(child_J.argmin())          # ties -> earliest generation index
        improved = parent_J - float(child_J[best])
        if improved >= cfg.accept_margin:
            incumbent = children[best]
            accepted += 1
            cur_J, cur_loss, cur_upd = float(child_J[best]), float(loss[best + 1]), float(upd[best + 1])
        else:
            cur_J, cur_loss, cur_upd = parent_J, float(loss[0]), float(upd[0])

        rec = {"round": rnd, "J": round(cur_J, 6),
               "loss_nats": round(cur_loss, 6),
               "bits_per_target": round(cur_loss / np.log(2), 4),
               "updates_per_symbol": round(cur_upd, 1),
               "accepted": bool(improved >= cfg.accept_margin),
               "accept_rate": round(accepted / rnd, 3),
               "seconds": round(time.time() - t0, 1)}

        if validation and validate_every and (rnd % validate_every == 0 or rnd == cfg.rounds):
            vloss, vupd, _ = evaluate(ModelStack([incumbent]), validation)
            rec["val_bits_per_target"] = round(float(vloss[0]) / np.log(2), 4)

        history.append(rec)
        if on_round:
            on_round(rec)
        if log_every and (rnd % log_every == 0 or rnd == 1):
            v = f" val {rec['val_bits_per_target']:.4f}" if "val_bits_per_target" in rec else ""
            print(f"  round {rnd:5d}/{cfg.rounds}  J {cur_J:.5f}  "
                  f"train {rec['bits_per_target']:.4f} bpt{v}  "
                  f"acc {rec['accept_rate']:.2f}  upd/sym {cur_upd:6.0f}  "
                  f"{rec['seconds']:.0f}s", flush=True)

    return incumbent, history
