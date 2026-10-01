"""GPU port of the batched evaluator. Exact integer semantics, torch tensors.

Same machine as `batch.py`, same results, different backend. Everything is
int32/int64; nothing here is a float except the final log-sum-exp, which
section 12 already specifies as float64 in the trainer.

Two details that matter for conformance:

  * `torch.argmax` does not promise the lowest index on ties, and section 5
    requires exactly that. So the comparison is done on `score * 64 - index`,
    which is injective and makes the lowest index win every tie. Scores are
    bounded by |bias| + 6*127 < 33_530, so this cannot overflow int64.
  * The dense path evaluates every cell each tick and masks the writes by the
    activity flags, rather than gathering the active subset. On a GPU the
    dense form is faster despite doing more work, and the masked result is
    identical to the sparse one because inactive cells write nothing.
"""

from __future__ import annotations

import numpy as np
import torch

from . import spec
from .engine import Model
from .spec import (N_CELLS, N_CHANNELS, N_PAGES, N_CANDIDATES, N_SCORE_TERMS,
                   N_INJECT, N_PROBE, MAX_TICKS, N_SYMBOLS, OFFSETS,
                   ROUTING_CHANNELS, N_PHASE, PAGE_STRIDE,
                   ROUTING_PAIR_OFFSET, BOS, EOS)

FIELDS = ("inj_cell", "inj_chan", "inj_delta", "sc_nb", "sc_ch", "sc_co",
          "sc_bias", "op_code", "op_d", "op_n", "op_c", "op_a", "op_b",
          "op_u", "pr_cell", "pr_chan", "pr_co", "pr_bias")


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():          # covers ROCm builds too
        return torch.device("cuda")
    return torch.device("cpu")


class TorchModelStack:
    """K models as int32 tensors on one device."""

    def __init__(self, models: list[Model], device, routing: str = "r1"):
        if spec.TOPIC_CHANNELS:
            raise NotImplementedError("this batch machine has no topic register; "
                                      "use engine.py for topic models")
        if spec.EXTENDED:
            raise NotImplementedError(
                "torch_batch runs R1 only; the tape extensions (MICA_TAPE, "
                "WINDOW, PROBE_*) are in engine.py and batch.py")
        self.K = len(models)
        self.device = device
        self.routing = routing
        for f in FIELDS:
            arr = np.stack([getattr(m, f) for m in models]).astype(np.int64)
            setattr(self, f, torch.from_numpy(arr).to(device))
        self.offsets = torch.tensor(OFFSETS, dtype=torch.long, device=device)


class TorchSession:
    def __init__(self, n: int, model_of: torch.Tensor, device):
        self.n = n
        self.device = device
        self.mi = model_of.to(device)
        self.F = torch.zeros((n, N_CELLS, N_CHANNELS), dtype=torch.int64, device=device)
        self.phase = torch.zeros((n, N_CELLS), dtype=torch.int64, device=device)
        self.position = torch.zeros(n, dtype=torch.long, device=device)
        self.updates = torch.zeros(n, dtype=torch.long, device=device)


def _routing_bits(F: torch.Tensor, mode: str) -> torch.Tensor:
    """(B,N,C) -> (B,N) four-bit routing pattern."""
    bits = torch.zeros(F.shape[:2], dtype=torch.long, device=F.device)
    if mode == "r1":
        for c in ROUTING_CHANNELS:
            bits |= (F[:, :, c] >= 0).long() << c
    elif mode == "positive":
        for c in ROUTING_CHANNELS:
            bits |= (F[:, :, c] > 0).long() << c
    elif mode == "pairdiff":
        for k, c in enumerate(ROUTING_CHANNELS):
            bits |= (F[:, :, c] >= F[:, :, c + ROUTING_PAIR_OFFSET]).long() << k
    elif mode == "parity":
        for k, c in enumerate(ROUTING_CHANNELS):
            bits |= (F[:, :, c] & 1) << k
    else:
        raise ValueError(mode)
    return bits


def _run_ticks(ms: TorchModelStack, st: TorchSession, active: torch.Tensor) -> None:
    B, N, C = st.F.shape
    cell_ix = torch.arange(N, device=st.device)
    batch_ix = torch.arange(B, device=st.device)

    for _ in range(MAX_TICKS):
        if not bool(active.any()):
            return
        st.updates += active.sum(1)

        F = st.F
        pid = PAGE_STRIDE * st.phase + _routing_bits(F, ms.routing)          # (B,N)
        k = st.mi[:, None].expand(B, N)                              # (B,N)

        # scoring: six triples per candidate
        nb = ms.sc_nb[k, pid]                                        # (B,N,32,6)
        ch = ms.sc_ch[k, pid]
        co = ms.sc_co[k, pid]
        src = (cell_ix[None, :, None, None] + ms.offsets[nb]) % N    # (B,N,32,6)
        # Address the (cell, channel) pair directly on a flattened field. The
        # obvious two-step form — gather whole rows, then pick the channel —
        # materialises a tensor C times larger for no reason.
        flat = (src * C + ch).reshape(B, -1)                         # (B,N*32*6)
        vals = torch.gather(F.reshape(B, N * C), 1, flat).reshape(
            B, N, N_CANDIDATES, N_SCORE_TERMS)
        scores = ms.sc_bias[k, pid] + (co * vals).sum(-1)            # (B,N,32)

        # ties resolve to the lowest candidate index (section 5)
        ranked = scores * 64 - torch.arange(N_CANDIDATES, device=st.device)
        win = ranked.argmax(-1)                                      # (B,N)

        opc = ms.op_code[k, pid, win]
        d = ms.op_d[k, pid, win]
        n_ = ms.op_n[k, pid, win]
        c_ = ms.op_c[k, pid, win]
        a = ms.op_a[k, pid, win]
        imm = ms.op_b[k, pid, win]
        u = ms.op_u[k, pid, win]

        src_cell = (cell_ix[None, :] + ms.offsets[n_]) % N            # (B,N)
        v = F[batch_ix[:, None], src_cell, c_]
        own_d = F[batch_ix[:, None], cell_ix[None, :], d]
        own_u = F[batch_ix[:, None], cell_ix[None, :], u]

        lo, hi = spec.SAT_MIN, spec.SAT_MAX
        new_d = torch.where(opc == spec.ADD, (own_d + a * v + imm).clamp(lo, hi),
                 torch.where(opc == spec.SET, (a * v + imm).clamp(lo, hi),
                 torch.where(opc == spec.SWAP, own_u,
                 torch.where(opc == spec.DECAY, own_d - own_d.sign(),
                 torch.where(opc == spec.TURN, -own_d, own_d)))))
        writes_d = (opc == spec.ADD) | (opc == spec.SET) | (opc == spec.DECAY) | \
                   (opc == spec.TURN) | ((opc == spec.SWAP) & (d != u))
        writes_u = (opc == spec.SWAP) & (d != u)

        G = F.clone()
        m = writes_d & active
        if bool(m.any()):
            G[batch_ix[:, None].expand(B, N)[m], cell_ix[None, :].expand(B, N)[m], d[m]] = new_d[m]
        m = writes_u & active
        if bool(m.any()):
            G[batch_ix[:, None].expand(B, N)[m], cell_ix[None, :].expand(B, N)[m], u[m]] = own_d[m]

        phase_next = torch.where(active, (st.phase + 1) % N_PHASE, st.phase)

        changed = (G != F).any(-1) & active                          # (B,N)
        next_active = changed.clone()
        for off in OFFSETS[1:]:
            next_active |= changed.roll(shifts=off, dims=1)
        pulse = (opc == spec.PULSE) & active
        if bool(pulse.any()):
            tgt = (cell_ix[None, :] + ms.offsets[n_]) % N
            next_active[batch_ix[:, None].expand(B, N)[pulse], tgt[pulse]] = True

        st.F, st.phase = G, phase_next
        active = next_active


def ingest_batch(ms: TorchModelStack, st: TorchSession, symbols: torch.Tensor) -> None:
    B = st.n
    rows = torch.arange(B, device=st.device)
    k = st.mi
    active = torch.zeros((B, N_CELLS), dtype=torch.bool, device=st.device)
    for e in range(N_INJECT):
        cell = (ms.inj_cell[k, symbols, e] + st.position) % N_CELLS
        chan = ms.inj_chan[k, symbols, e]
        delta = ms.inj_delta[k, symbols, e]
        cur = st.F[rows, cell, chan]
        st.F[rows, cell, chan] = (cur + delta).clamp(spec.SAT_MIN, spec.SAT_MAX)
        active[rows, cell] = True
    _run_ticks(ms, st, active)
    st.position = (st.position + 1) % N_CELLS


def probe_batch(ms: TorchModelStack, st: TorchSession) -> torch.Tensor:
    k = st.mi
    cell = ms.pr_cell[k]                                   # (B,258,8)
    if spec.ROLLING_READOUT:
        cell = (cell + st.position[:, None, None]) % N_CELLS
    chan = ms.pr_chan[k]
    co = ms.pr_co[k]
    B = st.n
    flat = (cell * N_CHANNELS + chan).reshape(B, -1)       # (B, 258*8)
    vals = torch.gather(st.F.reshape(B, N_CELLS * N_CHANNELS), 1, flat)
    vals = vals.reshape(B, N_SYMBOLS, N_PROBE)
    return ms.pr_bias[k] + (co * vals).sum(-1)


_ELIG = np.zeros(N_SYMBOLS, bool)
_ELIG[0:spec.BOS] = True
_ELIG[EOS] = True
ELIG_IDX = np.flatnonzero(_ELIG)


def evaluate(ms: TorchModelStack, records: list[bytes], chunk: int = 0,
             per_record: bool = False):
    """Same contract as batch.evaluate: (loss, updates_per_symbol, targets)."""
    K, R = ms.K, len(records)
    dev = ms.device
    mi = torch.arange(K, device=dev).repeat_interleave(R)
    st = TorchSession(K * R, mi, dev)

    ingest_batch(ms, st, torch.full((st.n,), BOS, dtype=torch.long, device=dev))

    maxlen = max(len(r) for r in records)
    padded = np.full((R, maxlen), 0, np.int64)
    for j, r in enumerate(records):
        padded[j, :len(r)] = (np.frombuffer(r, np.uint8) if N_SYMBOLS == 258
                              else np.asarray(r, dtype=np.int64))
    lengths = torch.tensor([len(r) for r in records], device=dev).repeat(K)
    toks = torch.from_numpy(np.tile(padded, (K, 1))).to(dev)

    elig = torch.from_numpy(ELIG_IDX).to(dev)
    nats = torch.zeros(st.n, dtype=torch.float64, device=dev)
    counts = torch.zeros(st.n, dtype=torch.long, device=dev)

    for t in range(maxlen + 1):
        scores = probe_batch(ms, st).to(torch.float64) / spec.LOGIT_DIVISOR
        alive = (t <= lengths)
        target = torch.where(t < lengths, toks[:, min(t, maxlen - 1)],
                             torch.full_like(lengths, EOS))
        sub = scores[:, elig]
        lse = torch.logsumexp(sub, dim=1)
        step = lse - scores.gather(1, target.unsqueeze(1)).squeeze(1)
        nats += torch.where(alive, step, torch.zeros_like(step))
        counts += alive.long()
        if t >= maxlen:
            break
        feed = torch.where(t < lengths, toks[:, t],
                           torch.full_like(lengths, BOS))
        ingest_batch(ms, st, feed)

    n_kr = nats.reshape(K, R); c_kr = counts.reshape(K, R)
    u_kr = st.updates.reshape(K, R).to(torch.float64)
    tgt = c_kr.sum(1)
    loss = n_kr.sum(1) / tgt
    upd = u_kr.sum(1) / tgt
    if per_record:
        return (loss.cpu().numpy(), upd.cpu().numpy(), tgt.cpu().numpy(),
                (n_kr / c_kr).cpu().numpy(), (u_kr / c_kr).cpu().numpy())
    return loss.cpu().numpy(), upd.cpu().numpy(), tgt.cpu().numpy()
