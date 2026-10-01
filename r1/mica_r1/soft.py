"""A differentiable MICA, matching the R1 transition where it can.

Direct mutation search cannot train a multi-megabyte model: measured on the
reference geometry, one accepted change costs ~44 s, and a 4.25 MB model has
3.4 million tunable slots. Gradient training is the only route to that size,
and this module is the relaxation that makes it possible.

WHAT IS RELAXED, AND WHY EACH CHOICE

  candidate selection   Fully soft. A page holds N candidates; the integer
                        machine takes the argmax of their scores. Here the
                        scores go through a softmax, and the tick's outcome is
                        the weighted mixture. N is 128 at the scaled geometry,
                        which is affordable to mix.

  opcode and operands   Fully soft. Eight opcodes and a handful of channel and
                        neighbour choices per candidate, so mixing all of them
                        costs little and gives every operand a gradient.

  page routing          NOT softened. Routing picks 1 page of 1,024 from the
                        signs of eight channels. Mixing over 1,024 pages x 128
                        candidates is 131,072 parameter sets per cell per tick,
                        which is not affordable at any batch size. Instead the
                        page is chosen hard, exactly as the integer machine
                        does, and the update is scaled by the router's own
                        confidence -- the standard trick from sparse
                        mixture-of-experts. The chosen page's parameters get a
                        true gradient; the routing decision gets one through
                        the confidence factor rather than through the index.

  activation            NOT modelled during training: every cell updates every
                        tick. R1's change-triggered activation is what makes
                        inference cheap, and it is a hard boolean that would
                        stop gradients dead. Training dense and running sparse
                        is a real assumption, not a free one -- verify it with
                        the activation-gap check in discretise.py before
                        believing any trained model's inference cost.

  saturation, DECAY     Left as they are. clamp and sign are differentiable
                        almost everywhere, which is enough.

The field itself stays a real-valued tensor in [-127, 127] rather than int8.
Rounding it back to integers is the discretisation gap, and measuring that gap
is the whole point of discretise.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import spec
from .spec import (N_CELLS, N_CHANNELS, N_PAGES, N_CANDIDATES, N_SCORE_TERMS,
                   N_INJECT, N_PROBE, N_SYMBOLS, N_SELECTORS, OFFSETS,
                   N_PHASE, PAGE_STRIDE, ROUTING_CHANNELS,
                   ROUTING_PAIR_OFFSET, BOS, EOS, SAT_MIN, SAT_MAX)

TERNARY = torch.tensor([-1.0, 0.0, 1.0])
# opcode, d, n, c, u, a, b -- the column blocks of rule_params()["ops"]
N_OPS = 9 if spec.VSET_WIDTH else 8          # VSET is opcode 8
VEXTRA = max(0, spec.VSET_WIDTH - 1)           # VSET's extra immediates
OPERAND_WIDTHS = (N_OPS, N_CHANNELS, N_SELECTORS, N_CHANNELS, N_CHANNELS, 1,
                  1) + ((VEXTRA,) if VEXTRA else ())


class _TakeAccum(torch.autograd.Function):
    """rows = src[idx], whose backward ADDS into a caller-owned buffer.

    The plain src[idx] backward allocates a zero tensor the size of all of src
    and scatters into it -- per tick. src here is a finished rule-book
    selector, 25 million floats for the channel selector alone, so every tick
    of every byte zero-filled and then added ~100 MB to reach a gradient that
    touches a few hundred rows: 70% of a training step's time, measured.
    This index_adds the rows into one buffer per step instead."""

    @staticmethod
    def forward(ctx, src, idx, buf):
        ctx.save_for_backward(idx)
        ctx.buf = buf
        return src[idx]

    @staticmethod
    def backward(ctx, g):
        idx, = ctx.saved_tensors
        ctx.buf.index_add_(0, idx, g)
        return None, None, None


def _take(rp: dict, name: str, idx: torch.Tensor) -> torch.Tensor:
    """rp[name][idx], through the accumulating gather when rp carries
    gradient buffers (SoftMica.backward_tbptt)."""
    bufs = rp.get("_bufs")
    if bufs is not None and name in bufs:
        return _TakeAccum.apply(rp[name], idx, bufs[name])
    return rp[name][idx]


def _logits(*shape, scale: float = 1.0) -> nn.Parameter:
    """Selector logits. A near-uniform selector mixes every choice equally and
    injects a near-zero, structureless field, so start them peaked."""
    return nn.Parameter(torch.randn(*shape) * scale)


# Verification hooks (r1/checks/verify_exact.py); both off in training.
#   TRACE: a list -> every hard tick appends (winner, top score, number of
#          candidates sharing the top score, page) for each updated cell.
#   LEGACY_ARGMAX: pick winners with a plain argmax, as this module did before
#          2026-09-25, to measure what the explicit tie rule changed.
TRACE = None
LEGACY_ARGMAX = False


class SoftMica(nn.Module):
    def __init__(self, ticks: int = 4, tau: float = 1.0,
                 tau_route: float = 4.0, hard: bool = True,
                 sel_init: float = 2.0, sel_tau: float | None = None,
                 compact_selectors: bool = False):
        super().__init__()
        if compact_selectors and N_SYMBOLS == 258:
            raise ValueError("compact selectors are for word-symbol fit mode")
        self.compact_selectors = compact_selectors
        self.ticks = ticks
        self.tau = tau
        self.tau_route = tau_route
        # Selector logits (which cell, channel, opcode, neighbour...). In the
        # HARD forward the argmax picks one option whatever the scale, so the
        # scale only shapes the backward pass: the softmax the gradient flows
        # through. Initialised at 2.0 and read at tau 0.5 (the original
        # choice, made for the soft forward), a 768-way softmax gives the top
        # few cells almost all the weight and every other cell a gradient of
        # about e^-10 -- a probe can never move to the cell that would help.
        # sel_init near 0 and sel_tau ~1 start every selector near-uniform in
        # the backward pass, so every option gets a gradient.
        self.sel_tau = tau if sel_tau is None else sel_tau
        # ``hard`` makes every forward pass run the INTEGER machine: argmax
        # selectors, rounded parameters, int8 field. Gradients still flow
        # through the soft version (straight-through), so training works, but
        # the thing being trained is the thing that ships.
        #
        # Without it the relaxation trains a strictly more powerful machine.
        # A soft read of a distribution over 768 cells is a weighted average
        # of 768 values; the integer machine can only read one cell. Gradient
        # descent will happily build solutions that need the average, and
        # those do not survive the argmax. Measured on a four-record overfit:
        # soft 3.01 bits, integer 9.52 bits -- worse than uniform.
        self.hard = hard

        P, C, T = N_PAGES, N_CANDIDATES, N_SCORE_TERMS
        S = N_SELECTORS
        # ---- candidate scoring ------------------------------------------
        self.sc_nb = _logits(P, C, T, S, scale=sel_init)
        self.sc_ch = _logits(P, C, T, N_CHANNELS, scale=sel_init)
        self.sc_co = _logits(P, C, T, 3, scale=sel_init)
        self.sc_bias = nn.Parameter(torch.zeros(P, C))
        # ---- rewrite program --------------------------------------------
        self.op_code = _logits(P, C, N_OPS, scale=sel_init)
        self.op_d = _logits(P, C, N_CHANNELS, scale=sel_init)
        self.op_n = _logits(P, C, S, scale=sel_init)
        self.op_c = _logits(P, C, N_CHANNELS, scale=sel_init)
        self.op_a = _logits(P, C, 3, scale=sel_init)
        self.op_b = nn.Parameter(torch.zeros(P, C))
        self.op_u = _logits(P, C, N_CHANNELS, scale=sel_init)
        if VEXTRA:
            # VSET's immediates beyond op_b (spec.VSET_WIDTH)
            self.op_v = nn.Parameter(torch.zeros(P, C, VEXTRA))
        # ---- injection ---------------------------------------------------
        selector_rows = 1 if compact_selectors else N_SYMBOLS
        self.inj_cell = _logits(selector_rows, N_INJECT, N_CELLS, scale=sel_init)
        self.inj_chan = _logits(selector_rows, N_INJECT, N_CHANNELS, scale=sel_init)
        self.inj_delta = nn.Parameter(torch.randn(N_SYMBOLS, N_INJECT) * 8.0)
        # ---- readout -------------------------------------------------------
        self.pr_cell = _logits(selector_rows, N_PROBE, N_CELLS, scale=sel_init)
        self.pr_chan = _logits(selector_rows, N_PROBE, N_CHANNELS, scale=sel_init)
        self.pr_co = _logits(selector_rows, N_PROBE, 3, scale=sel_init)
        self.pr_bias = nn.Parameter(torch.zeros(N_SYMBOLS))
        if spec.WIDE_PROBE_COEF:
            # spec.WIDE_PROBE_COEF: an int8 coefficient per probe, trained as
            # a rounded real number like the biases. It starts at zero, so a
            # probe contributes nothing until the gradient says it should;
            # pr_co is then unused (kept so the parameter set stays uniform).
            self.pr_w = nn.Parameter(torch.zeros(N_SYMBOLS, N_PROBE))

        self.register_buffer("off", torch.tensor(OFFSETS, dtype=torch.long))
        self.register_buffer("tern", TERNARY.clone())
        elig = torch.zeros(N_SYMBOLS, dtype=torch.bool)
        elig[:BOS] = True
        elig[EOS] = True
        self.register_buffer("elig", elig)
        # Routing in one shot instead of a Python loop over the eight bits.
        # Not persistent: they are constants, and leaving them out of the
        # state_dict keeps every saved checkpoint loadable.
        rc = torch.tensor(ROUTING_CHANNELS, dtype=torch.long)
        self.register_buffer("rc_a", rc, persistent=False)
        self.register_buffer("rc_b", rc + ROUTING_PAIR_OFFSET, persistent=False)
        self.register_buffer("rc_w", 1 << torch.arange(len(ROUTING_CHANNELS)),
                             persistent=False)
        # Address rules of the tape extensions, as "forbidden" masks applied to
        # the selector logits before softmax and argmax: a forbidden choice gets
        # probability 0, no gradient, and is never the argmax. Not persistent.
        K, N = spec.TAPE_CHANNELS, N_CELLS
        masks = {}
        if K:
            no_tape = torch.arange(N_CHANNELS) < K
            masks["op_d"] = no_tape
            masks["op_u"] = no_tape
            # tape entries: exactly (head, e); work entries: any work channel
            ic = no_tape[None, :].repeat(N_INJECT, 1)
            ic[:K] = True
            ic[torch.arange(K), torch.arange(K)] = False
            masks["inj_chan"] = ic
            icell = torch.zeros(N_INJECT, N, dtype=torch.bool)
            icell[:K] = True
            icell[:K, 0] = False
            if spec.WINDOW:
                lag = (N - torch.arange(N)) % N          # cell k is `lag` behind
                icell[K:] = lag[None, :] >= spec.WINDOW
            masks["inj_cell"] = icell
        T = spec.TOPIC_CHANNELS
        if T:
            # spec.TOPIC_CHANNELS: no rule or work injection writes the
            # register; a VSET writes VSET_WIDTH channels from d
            top = spec.TOPIC_AT
            ch = torch.arange(N_CHANNELS)
            masks["op_d"] = masks["op_d"] | (ch + max(1, spec.VSET_WIDTH) - 1 >= top)
            masks["op_u"] = masks["op_u"] | (ch >= top)
            ic = masks["inj_chan"]
            ic[K + T:] |= (ch >= top)[None, :]
            ic[K:K + T] = True
            ic[K + torch.arange(T), top + torch.arange(T)] = False
            masks["inj_cell"][K:K + T] = True
            masks["inj_cell"][K:K + T, 0] = False
        if spec.PROBE_WINDOW:
            masks["pr_cell"] = torch.arange(N) < N - spec.PROBE_WINDOW
        self._mask_names = tuple(masks)
        for name, m in masks.items():
            self.register_buffer("mask_" + name, m, persistent=False)
        tape = torch.zeros(N_CHANNELS)
        tape[:K] = 1.0
        self.register_buffer("tape_ch", tape, persistent=False)

    # -- selector helpers -------------------------------------------------
    def logits_of(self, name: str) -> torch.Tensor:
        """A selector's logits with the extensions' forbidden choices masked.
        Everything that reads a selector -- training, discretisation, the
        confidence report -- goes through here, so they cannot disagree."""
        x = getattr(self, name)
        if name in self._mask_names:
            x = x.masked_fill(getattr(self, "mask_" + name), -1e9)
        return x

    def _st(self, soft: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
        """Straight-through: ``hard`` forward, ``soft``'s gradient backward.

        The brackets matter. ``hard + soft - soft.detach()`` rounds hard+soft
        to float first, so the forward value comes out as -7.0000005 instead
        of -7; the error then decides sign tests in routing (0 vs -5e-7) and
        the trained machine drifts away from the integer one. soft - soft is
        exactly zero in IEEE arithmetic, so this form is exact."""
        return hard + (soft - soft.detach())

    def _sel(self, logit: torch.Tensor) -> torch.Tensor:
        soft = Fn.softmax(logit / self.sel_tau, dim=-1)
        if not self.hard:
            return soft
        # scatter into a float tensor rather than Fn.one_hot(...).to(float):
        # one_hot builds an int64 tensor first, which at (cells x 128
        # candidates x 6 terms x 32 channels) is a 600 MB transient per call.
        hard = torch.zeros_like(soft).scatter_(
            -1, logit.argmax(-1, keepdim=True), 1.0)
        return self._st(soft, hard)

    def _ternary(self, logit: torch.Tensor) -> torch.Tensor:
        """A coefficient in {-1, 0, +1}: exactly one of them when hard."""
        return (self._sel(logit) * self.tern).sum(-1)

    def _round(self, x: torch.Tensor, lo: float = SAT_MIN,
               hi: float = SAT_MAX) -> torch.Tensor:
        """Straight-through rounding, so integer-valued parameters and the
        int8 field are integer-valued in the forward pass too."""
        if not self.hard:
            return x
        return self._st(x, x.detach().round().clamp(lo, hi))

    # -- state ------------------------------------------------------------
    def init_state(self, B: int, device) -> dict:
        return {
            "F": torch.zeros(B, N_CELLS, N_CHANNELS, device=device),
            "phase": torch.zeros(B, N_CELLS, dtype=torch.long, device=device),
            "pos": torch.zeros(B, dtype=torch.long, device=device),
        }

    # -- routing ----------------------------------------------------------
    def _page_ids(self, F: torch.Tensor, phase: torch.Tensor):
        """Hard page index, plus the router's confidence in it.

        The confidence is the product of the per-bit sigmoid margins. It is a
        scalar in (0, 1] per cell that multiplies the tick's update, so a
        routing decision the field only just supports contributes a smaller
        update than a decisive one -- and the field gets a gradient telling it
        which way to move to make the decision cleaner.
        """
        diff = F[:, :, self.rc_a] - F[:, :, self.rc_b]      # (B,N,K)
        bits = ((diff >= 0).long() * self.rc_w).sum(-1)     # (B,N)
        conf = torch.sigmoid(diff.abs() / self.tau_route).prod(-1)
        if self.hard:
            # The integer machine scales nothing by routing confidence. Keep
            # the gradient path to the router, but make the forward pass 1.
            conf = self._st(conf, torch.ones_like(conf))
        return PAGE_STRIDE * phase + bits, conf

    def _neighbour_index(self, B: int, device) -> torch.Tensor:
        """Row of every cell's neighbour at each selector offset, in the
        flattened (B*N) field. Constant for a batch size, so it is built once
        and reused on every tick instead of costing eight operations each."""
        cache = self.__dict__.setdefault("_nbr_cache", {})
        key = (B, str(device))
        if key not in cache:
            idx = torch.arange(N_CELLS, device=device)
            src = (idx[:, None] + self.off.to(device)[None, :]) % N_CELLS
            base = torch.arange(B, device=device)[:, None, None] * N_CELLS
            cache[key] = (base + src[None]).reshape(-1, N_SELECTORS)
        return cache[key]

    # -- one tick ---------------------------------------------------------
    def _window_index(self, head: torch.Tensor):
        """spec.WINDOW: the W cells behind each head, newest first (B,W), and
        each one's neighbour rows in the flattened (B*N) field (B*W,S)."""
        B = head.shape[0]
        j = torch.arange(spec.WINDOW, device=head.device)
        widx = (head[:, None] - j[None, :]) % N_CELLS
        nbr = (widx[:, :, None] + self.off[None, None, :]) % N_CELLS
        base = torch.arange(B, device=head.device)[:, None, None] * N_CELLS
        return widx, (base + nbr).reshape(-1, N_SELECTORS)

    def tick(self, st: dict, rp: dict = None) -> dict:
        rp = self.rule_params() if rp is None else rp
        F, phase = st["F"], st["phase"]
        B = F.shape[0]
        Fv = F.reshape(-1, N_CHANNELS)
        if spec.WINDOW:
            # only the W cells behind the head update; the rest is frozen.
            # ingest has already advanced pos, so the head is pos - 1.
            widx, nbr_rows = self._window_index((st["pos"] - 1) % N_CELLS)
            gidx = widx[:, :, None].expand(-1, -1, N_CHANNELS)
            Fr = torch.gather(F, 1, gidx)                  # (B,W,Ch)
            phr = torch.gather(phase, 1, widx)             # (B,W)
            own = Fr.reshape(-1, N_CHANNELS)
        else:
            nbr_rows = self._neighbour_index(B, F.device)
            Fr, phr, own = F, phase, Fv
        pid, conf = self._page_ids(Fr, phr)                # (B,R), (B,R)
        flat = pid.reshape(-1)

        # this cell's page: finished selector rows, built once per step
        nb = _take(rp, "nb", flat)                                # (BN,C,T,S)
        ch = _take(rp, "ch", flat)                                # (BN,C,T,Ch)
        co = _take(rp, "co", flat)                                # (BN,C,T)

        # neighbour values for every selector and channel: (BN, S, Ch)
        neigh = Fv[nbr_rows]                               # (BN,S,Ch)

        # score = bias + sum_t co_t * <nb_t, <ch_t, neigh>>
        # contract channels first, then selectors: (BN,C,T)
        val = torch.einsum("bsc,bktc->bkts", neigh, ch)    # (BN,C,T,S)
        val = (val * nb).sum(-1)                           # (BN,C,T)
        scores = _take(rp, "bias", flat) + (co * val).sum(-1)     # (BN,C)

        soft_w = Fn.softmax(scores / self.tau, -1)
        if self.hard:
            # argmax, ties to the lowest index -- the integer machine's rule
            # scores are integers here; scaling by C+1 and subtracting the
            # index makes the lowest index win every tie explicitly, instead
            # of trusting the GPU argmax to return the first maximum (the
            # engines and fit.provenance do the same)
            C = scores.shape[-1]
            if LEGACY_ARGMAX:
                # verification only: the code before 2026-09-25 (plain argmax)
                win = scores.detach().argmax(-1, keepdim=True)
            else:
                key = scores.detach() * (C + 1) - torch.arange(
                    C, device=scores.device, dtype=scores.dtype)
                win = key.argmax(-1, keepdim=True)
            w = self._st(soft_w, torch.zeros_like(soft_w).scatter_(
                -1, win, 1.0))
            if TRACE is not None:
                s_ = scores.detach()
                top = s_.max(-1).values
                TRACE.append((win[:, 0].cpu(), top.double().cpu(),
                              (s_ == top[:, None]).sum(-1).cpu(),
                              pid.reshape(-1).cpu()))
        else:
            w = soft_w

        # ---- operands, mixed over candidates ----------------------------
        def mix(sel):                                       # (BN,C,K) -> (BN,K)
            return (w[:, :, None] * sel).sum(1)

        # every operand selector in one tensor: one gather, one mix
        parts = torch.split(mix(_take(rp, "ops", flat)), OPERAND_WIDTHS,
                            dim=1)
        opc, d, n, c_, u, a, b = parts[:7]
        a, b = a[:, 0], b[:, 0]                             # (BN,)

        own_d = (own * d).sum(-1, keepdim=True)             # (BN,1)
        own_u = (own * u).sum(-1, keepdim=True)
        v = ((neigh * n[:, :, None]).sum(1) * c_).sum(-1, keepdim=True)

        # ---- the eight opcodes, each as a full-width channel delta -------
        # Every opcode writes through the soft channel selector d (or u), so
        # the write lands as a weighted spread over channels rather than a
        # single index. At tau -> 0 the spread collapses to one channel and
        # this matches the integer machine.
        zero = torch.zeros_like(own_d)
        tgt = {
            spec.HOLD: (zero, zero),
            spec.ADD: (own_d + a[:, None] * v + b[:, None] - own_d, zero),
            spec.SET: (a[:, None] * v + b[:, None] - own_d, zero),
            spec.SWAP: (own_u - own_d, own_d - own_u),
            # "one step toward zero" is exactly -sign(x). tanh is its smooth
            # stand-in and the two differ by up to 1 per tick, which is a real
            # semantic gap rather than a rounding one -- so take sign in the
            # forward pass and tanh's gradient in the backward.
            spec.DECAY: (self._st(-torch.tanh(own_d * 4.0),
                                  -torch.sign(own_d)) if self.hard
                         else -torch.tanh(own_d * 4.0), zero),
            spec.TURN: (-2.0 * own_d, zero),
            spec.PULSE: (zero, zero),
            spec.QUIET: (zero, zero),
        }
        # One stacked product per side instead of sixteen small ones: the
        # step is paced by the NUMBER of GPU operations, not their size.
        tgt_d = torch.stack([tgt[k][0] for k in range(8)], 1)  # (BN,8,1)
        tgt_u = torch.stack([tgt[k][1] for k in range(8)], 1)
        delta_d = (opc[:, :8, None] * tgt_d).sum(1)            # (BN,1)
        delta_u = (opc[:, :8, None] * tgt_u).sum(1)

        upd = delta_d * d + delta_u * u
        if spec.VSET_WIDTH:
            # VSET: channel d+j is SET to its j-th immediate; the one-hot d
            # shifted by j (dropping off the end: those channels do not exist)
            vals = [b] + ([parts[7][:, j] for j in range(VEXTRA)] if VEXTRA
                          else [])
            vs = torch.zeros_like(own)
            for j, val in enumerate(vals):
                sj = Fn.pad(d[:, :N_CHANNELS - j], (j, 0)) if j else d
                vs = vs + (val[:, None] - (own * sj).sum(-1, keepdim=True)) * sj
            upd = upd + opc[:, spec.VSET:spec.VSET + 1] * vs
        upd = upd * conf.reshape(-1, 1)
        G = self._round((own + upd).clamp(SAT_MIN, SAT_MAX))
        if spec.WINDOW:
            G = F.scatter(1, gidx, G.reshape(B, -1, N_CHANNELS))
            phase = phase.scatter(1, widx, (phr + 1) % N_PHASE)
            return {"F": G, "phase": phase, "pos": st["pos"]}
        G = G.reshape(B, N_CELLS, N_CHANNELS)
        return {"F": G, "phase": (phase + 1) % N_PHASE, "pos": st["pos"]}

    # -- ingest and readout ----------------------------------------------
    def ingest(self, st: dict, sym: torch.Tensor, rp: dict = None) -> dict:
        if spec.TOPIC_CHANNELS:
            raise NotImplementedError(
                "the straight-through soft machine has no topic register; "
                "topic models train in fit mode (fit._simulate) only")
        rp = self.rule_params() if rp is None else rp
        F, phase = st["F"], st["phase"]
        B = F.shape[0]
        K = spec.TAPE_CHANNELS
        shift = st["pos"]
        if K:
            # spec.TAPE_CHANNELS: the head cell starts a new life -- cleared,
            # phase 0 -- and the symbol's code is SET on the tape channels.
            # The code is the first K injection deltas, so it gets its
            # gradient from every later probe or rule that reads it.
            hm = torch.zeros(B, N_CELLS, device=F.device)
            hm.scatter_(1, shift[:, None], 1.0)
            code = _take(rp, "inj_delta", sym)[:, :K]                # (B,K)
            head_val = Fn.pad(code, (0, N_CHANNELS - K))       # (B,Ch)
            F = F * (1.0 - hm[:, :, None]) + hm[:, :, None] * head_val[:, None, :]
            phase = phase.masked_fill(hm.bool(), 0)
        if N_INJECT > K:
            cell = _take(rp, "inj_cell", sym)[:, K:]                 # (B,I',N)
            chan = _take(rp, "inj_chan", sym)[:, K:]                 # (B,I',Ch)
            delta = _take(rp, "inj_delta", sym)[:, K:]               # (B,I')
            # roll each symbol's cell distribution by the rolling position
            ar = torch.arange(N_CELLS, device=F.device)
            rolled = (ar[None, :] - shift[:, None]) % N_CELLS  # (B,N)
            cell = torch.gather(cell, 2, rolled[:, None, :].expand(
                -1, N_INJECT - K, -1))
            write = torch.einsum("bin,bic,bi->bnc", cell, chan, delta)
            F = self._round((F + write).clamp(SAT_MIN, SAT_MAX))
        st = {"F": F, "phase": phase, "pos": (shift + 1) % N_CELLS}
        for _ in range(self.ticks):
            st = self.tick(st, rp)
        return st

    def rule_params(self) -> dict:
        """The rule book's and the injection's selectors, finished.

        They depend only on parameters, so they are built once per training
        step and each tick just gathers the rows its cells need. Before, every
        cell on every tick gathered raw logits and ran softmax, argmax,
        one-hot and the straight-through sum on its own copy: about seventy
        small GPU operations per tick and, at 768 cells x 128 candidates x 6
        terms x 32 channels, four ~300 MB tensors per tick at batch 4. The
        same fix probe_params made for the readout. The result is identical;
        only where it is computed moves.
        """
        return {
            "nb": self._sel(self.sc_nb), "ch": self._sel(self.sc_ch),
            "co": self._ternary(self.sc_co),
            "bias": self._round(self.sc_bias, -32768, 32767),
            "ops": torch.cat([self._sel(self.op_code),
                              self._sel(self.logits_of("op_d")),
                              self._sel(self.op_n), self._sel(self.op_c),
                              self._sel(self.logits_of("op_u")),
                              self._ternary(self.op_a)[..., None],
                              self._round(self.op_b)[..., None]] +
                             ([self._round(self.op_v)] if VEXTRA else []), -1),
            "inj_cell": self._sel(self.logits_of("inj_cell")),
            "inj_chan": self._sel(self.logits_of("inj_chan")),
            "inj_delta": self._round(self.inj_delta),
        }

    def probe_params(self) -> tuple:
        """The readout's selectors, which depend only on parameters.

        Hoisted out of the per-symbol loop deliberately. Computing them inside
        it was the whole memory problem: pr_cell alone is (258, 32, 768), so a
        softmax plus a one-hot plus the result is about 75 MB, and autograd
        retained a fresh copy for each of the 1,025 symbols in a record. That
        is tens of gigabytes of identical tensors -- which is why a 16 GB card
        reported 29.47 GiB allocated even at batch size 1.
        """
        co = (self._round(self.pr_w, -127, 127) if spec.WIDE_PROBE_COEF
              else self._ternary(self.pr_co))                # (V,P)
        cell = self._sel(self.logits_of("pr_cell"))          # (V,P,N)
        if spec.PROBE_WINDOW and spec.ROLLING_READOUT:
            # only the window's R cells can be chosen; slicing here, once,
            # saves a full-size gradient buffer per byte in the backward
            cell = cell[:, :, N_CELLS - spec.PROBE_WINDOW:]  # (V,P,R)
        return (cell,
                self._sel(self.pr_chan),                     # (V,P,Ch)
                co,
                self._round(self.pr_bias, -32768, 32767))    # (V,)

    def logits(self, st: dict, pr: tuple = None) -> torch.Tensor:
        F = st["F"]
        cell, chan, co, bias = pr if pr is not None else self.probe_params()
        if spec.ROLLING_READOUT:
            # Read in the write head's frame: probe cell k means field cell
            # (k + pos). Rolling F (B,N,C) is cheap; rolling the selectors
            # would be (B,V,P,N), a hundred megabytes per byte. With a probe
            # window only its R cells can be read, so only they are rolled
            # and contracted -- the rest have probability exactly 0.
            lo = N_CELLS - cell.shape[-1]       # probe_params sliced it
            idx = (torch.arange(lo, N_CELLS, device=F.device)[None, :] +
                   st["pos"][:, None]) % N_CELLS
            F = torch.gather(F, 1, idx[:, :, None].expand(-1, -1, F.shape[2]))
        if self.hard and not torch.is_grad_enabled():
            # Forward-only (validation, checks): the hard selectors are
            # one-hot, so read each probe's one cell and channel directly.
            # Exact, and it avoids the dense contraction below, which on the
            # PC's GPU (ROCm 7.2.1, RX 9070 XT) returned output scores off by
            # up to ~42,000 units for the 32-phase geometry (432 probes x 208
            # channels) while the rule machine agreed exactly; exact for the
            # main 16-phase geometry -- measured by r1/checks/verify_exact.py
            # on 2026-09-26.
            ci = cell.argmax(-1)                             # (V,P)
            hi = chan.argmax(-1)                             # (V,P)
            reads = F[:, ci, hi]                             # (B,V,P)
            sc = bias[None, :] + (reads * co[None]).sum(-1)
            return sc / spec.LOGIT_DIVISOR
        # Two explicit contractions, never one three-operand einsum. The
        # single-einsum form materialises a (B, V, P, N, C) intermediate --
        # 203 million floats, 812 MB per symbol at batch 1. Contracting cells
        # first gives (V, P, B, C), which is 264 thousand.
        tmp = torch.einsum("vpn,bnc->vpbc", cell, F)
        reads = torch.einsum("vpbc,vpc->bvp", tmp, chan)
        sc = bias[None, :] + (reads * co[None]).sum(-1)
        return sc / spec.LOGIT_DIVISOR

    # -- training step: forward and backward one segment at a time ----------
    def backward_tbptt(self, batch: torch.Tensor, lengths: torch.Tensor,
                       tbptt: int) -> tuple:
        """sequence_loss(...).backward() with the same gradient, in a fraction
        of the memory and without recomputation.

        With truncated backpropagation the gradient of the loss at byte t can
        only reach the last `tbptt` bytes, so each segment's graph can be
        differentiated and freed as soon as the segment ends -- instead of
        keeping (or, with checkpointing, recomputing) the whole record's graph
        until one backward at the end. The rule book's and the readout's
        finished selectors are shared by every segment: they are detached
        into leaves here, collect their gradient from every segment, and are
        pushed back into the parameters with one backward at the end.

        Returns (mean loss as a detached scalar, number of targets)."""
        B, L = batch.shape
        dev = batch.device
        rp_full = self.rule_params()
        pr_full = self.probe_params()
        # leaves that "require grad" only so autograd records the gathers;
        # their gradient goes to the buffers, never to .grad
        rp = {k: v.detach().requires_grad_(v.requires_grad)
              for k, v in rp_full.items()}
        rp["_bufs"] = {k: torch.zeros_like(v) for k, v in rp_full.items()
                       if v.requires_grad}
        pr = tuple(v.detach().requires_grad_(v.requires_grad) for v in pr_full)
        mask = torch.full((N_SYMBOLS,), float("-inf"), device=dev)
        mask[self.elig] = 0.0
        # the loss is the mean over records of each record's mean nats, and a
        # record has lengths+1 targets (its bytes, then EOS) -- known up front,
        # so every target's weight in the final mean is known up front too
        weight = 1.0 / ((lengths.clamp(max=L) + 1).float() * B)
        st = self.init_state(B, dev)
        st = self.ingest(st, torch.full((B,), BOS, dtype=torch.long,
                                        device=dev), rp)
        seg = None
        total = torch.zeros((), device=dev)
        for t in range(L + 1):
            sc = self.logits(st, pr) + mask
            tgt = torch.where(t < lengths, batch[:, min(t, L - 1)],
                              torch.full_like(lengths, EOS))
            alive = (t <= lengths).float()
            term = (alive * weight *
                    Fn.cross_entropy(sc, tgt, reduction="none")).sum()
            seg = term if seg is None else seg + term
            if t >= L:
                break
            if tbptt and t and t % tbptt == 0:
                seg.backward()
                total = total + seg.detach()
                seg = None
                st = {"F": st["F"].detach(), "phase": st["phase"],
                      "pos": st["pos"]}
            feed = torch.where(t < lengths, batch[:, t],
                               torch.full_like(lengths, BOS))
            st = self.ingest(st, feed, rp)
        if seg is not None:
            seg.backward()
            total = total + seg.detach()
        outs, grads = [], []
        for k, full in rp_full.items():
            if k in rp["_bufs"]:
                outs.append(full)
                grads.append(rp["_bufs"][k])
        for full, leaf in zip(pr_full, pr):
            if full.requires_grad and leaf.grad is not None:
                outs.append(full)
                grads.append(leaf.grad)
        if outs:
            torch.autograd.backward(outs, grads)
        return total, (lengths.clamp(max=L) + 1).sum()

    # -- sequence loss ----------------------------------------------------
    def sequence_loss(self, batch: torch.Tensor, lengths: torch.Tensor,
                      checkpoint_every: int = 0, tbptt: int = 0) -> tuple:
        """Mean nat loss over predicted targets, section 12's protocol.

        ``checkpoint_every`` (any non-zero value) recomputes each symbol's
        field instead of storing its activations, which is what makes
        1,024-symbol records trainable at all: full storage would be twelve
        thousand sequential states per record.
        """
        B, L = batch.shape
        dev = batch.device
        st = self.init_state(B, dev)
        rp = self.rule_params()           # once per step, not once per tick
        st = self.ingest(st, torch.full((B,), BOS, dtype=torch.long,
                                        device=dev), rp)
        pr = self.probe_params()          # once per record, not once per byte
        nats = torch.zeros(B, device=dev)
        count = torch.zeros(B, device=dev)
        mask = torch.full((N_SYMBOLS,), float("-inf"), device=dev)
        mask[self.elig] = 0.0

        for t in range(L + 1):
            # The readout is also recomputed rather than stored: its
            # intermediate is a megabyte per symbol, which over a 1,024-byte
            # record is another gigabyte of retained graph for no reason.
            sc = (torch.utils.checkpoint.checkpoint(
                      self.logits, st, pr, use_reentrant=False)
                  if checkpoint_every else self.logits(st, pr)) + mask
            tgt = torch.where(t < lengths, batch[:, min(t, L - 1)],
                              torch.full_like(lengths, EOS))
            alive = (t <= lengths).float()
            nats = nats + alive * Fn.cross_entropy(sc, tgt, reduction="none")
            count = count + alive
            if t >= L:
                break
            feed = torch.where(t < lengths, batch[:, t],
                               torch.full_like(lengths, BOS))
            if tbptt and t and t % tbptt == 0:
                # Truncated backpropagation through time. The field still
                # carries everything forward -- the model's memory is
                # untouched -- but the gradient may only flow back `tbptt`
                # symbols. Without this it explodes: measured on the
                # reference geometry, the gradient norm is ~1e3 at 16 bytes,
                # 6.5e14 at 64 and infinite by 128. Records are 1,024 bytes.
                st = {"F": st["F"].detach(), "phase": st["phase"],
                      "pos": st["pos"]}
            if checkpoint_every:
                # EVERY step, not "every n-th". The earlier `t % n` form left
                # one step in n un-checkpointed, and an un-checkpointed step
                # retains the whole tick's activations: at 768 cells x 128
                # candidates x 6 terms x 32 channels a single tick holds about
                # 600 MB per tensor. With 128 such steps in a 1,024-byte
                # record that is 28.7 GiB, which is exactly what a 15.9 GiB
                # card reported before this was fixed.
                st = torch.utils.checkpoint.checkpoint(
                    self.ingest, st, feed, rp, use_reentrant=False)
            else:
                st = self.ingest(st, feed, rp)
        return (nats / count).mean(), count.sum()
