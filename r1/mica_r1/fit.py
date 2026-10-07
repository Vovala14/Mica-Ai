"""Direct fits for the tape machine (spec.TAPE_CHANNELS and friends).

Straight-through training has to discover every integer of the model one
noisy, hard step at a time. Two parts of the tape machine do not need that,
because with the right starting structure their effect on the output is an
ordinary differentiable function that can be fitted to the data in seconds:

  * the readout of the tape: logit(v) = bias[v] + sum_p w[v,p] * code[byte
    at lag_p][chan_p] / divisor  (train_soft.fit_tape_readout);

  * rule-computed features, when the rule book starts STRUCTURED: every
    candidate's program is "SET work channel d to the immediate b", and every
    candidate is scored and routed on tape channels only. Then which candidate
    wins at a cell -- its page and its argmax -- is a function of the byte
    tape alone (a hash of the byte there and the bytes before it), and the
    value a work channel holds is exactly the winner's immediate. The readout
    is then linear in the immediates, so the immediates and the work probes'
    coefficients can be fitted jointly, exactly, with the file's rounding.

Everything afterwards is ordinary straight-through training of the whole
machine, which can change any of this (reroute, rescore, change opcodes) if
the gradient says so.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from . import spec
from .spec import (N_CELLS, N_CHANNELS, N_PAGES, N_CANDIDATES, N_SCORE_TERMS,
                   N_SYMBOLS, N_PHASE, PAGE_STRIDE, ROUTING_CHANNELS,
                   ROUTING_PAIR_OFFSET, OFFSETS, BOS, EOS,
                   SAT_MIN, SAT_MAX)


def _covered_neighbour_choice(nallow: torch.Tensor, candidates: int,
                              terms: int, self_terms: int,
                              generator: torch.Generator) -> torch.Tensor:
    """Assign distinct allowed tape lags to each rule before repeating any.

    ``nallow`` counts the available offsets in nearest-first order. When a
    rule has fewer external score terms than offsets, it samples a subset;
    across candidates each offset can still be used. This only initializes
    integer neighbour selectors; the cellular inference engine is unchanged.
    """
    slots = terms - self_terms
    if not 0 <= self_terms < terms or torch.any(nallow < 1):
        raise ValueError("lag coverage needs a scoring term and a neighbour")
    choice = torch.zeros(len(nallow), candidates, terms, dtype=torch.long)
    for page, allowed in enumerate(nallow.tolist()):
        order = torch.rand(candidates, allowed, generator=generator).argsort(-1)
        choice[page, :, self_terms:] = order[:, torch.arange(slots) % allowed]
    return choice


def structured_rules(model, sel_init: float, margin: float = 3.0,
                     seed: int = 2, max_back: tuple = (1, 2, 2, 2),
                     self_terms: int = 0,
                     lag_coverage: bool = False) -> dict:
    """Rule book start for the tape machine.

    Every candidate: SET d = 0 * v + b, with a random immediate b. The channel
    d depends on the page's phase: phase p writes the p-th quarter of the work
    channels, one of them per candidate (candidate k takes k mod the quarter's
    size), so the head cell's two ticks (phases 0 and 1) fill the first half
    and its two ticks one byte later (phases 2 and 3) the second half.

    Scoring: each of a candidate's terms reads a TAPE channel of an older
    neighbour, selected from the first max_back[phase] available offsets,
    with coefficient +-1. Optional lag coverage uses distinct allowed offsets
    before reusing one, so long offsets are not lost to repeated short reads.
    The page
    already encodes the byte at the cell itself (routing reads tape channels),
    so at the head the winner is a hash of (this byte, the byte before) for
    phase 0 and of (this byte, the two before) for phase 1: 128 buckets per
    page, each with its own immediate.

    Margins fix the argmax; the soft selectors stay near-uniform around it so
    straight-through training can still move anything."""
    K = spec.TAPE_CHANNELS
    assert K and N_CHANNELS > K, "structured rules need tape and work channels"
    g = torch.Generator().manual_seed(seed)

    def noise(t):
        return torch.randn(t.shape, generator=g) * sel_init

    P, C, T = N_PAGES, N_CANDIDATES, N_SCORE_TERMS
    nwork = N_CHANNELS - K
    quarter = max(1, nwork // N_PHASE)
    phase = torch.arange(P) // PAGE_STRIDE                          # (P,)
    k = torch.arange(C)
    VW = spec.VSET_WIDTH
    if VW:
        # VSET: every candidate writes its phase's whole group of VW channels
        assert nwork >= N_PHASE * VW, (
            f"VSET width {VW} needs {N_PHASE * VW} work channels")
        quarter = VW
        d = (K + phase[:, None] * VW).expand(P, C).clone()
    else:
        d = K + (phase[:, None] * quarter + k[None, :] % quarter) % nwork
    # neighbour selectors whose offset is -1, -2, ... (spec.OFFSETS order);
    # phase p's terms read one of the max_back[p] nearest older neighbours
    back = {-o: s for s, o in enumerate(OFFSETS) if o < 0}          # dist -> sel
    assert 1 in back, "structured rules need the -1 neighbour"
    dists = sorted(back)
    nallow = torch.tensor([min(max_back[p % len(max_back)], len(dists))
                           for p in range(N_PHASE)])[phase]          # (P,)
    # Consume the same random draws in both arms so channels, signs and
    # immediates remain matched even when phase lag limits differ.
    base = torch.rand(P, C, T, generator=g)
    if lag_coverage:
        choice = _covered_neighbour_choice(
            nallow, C, T, self_terms, torch.Generator().manual_seed(seed + 7919))
    else:
        choice = (base * nallow[:, None, None]).long()
    nb_sel = torch.tensor([back[b] for b in dists])[choice]
    if self_terms:
        # the first `self_terms` terms read the cell's OWN tape: with few
        # routing bits a page holds several bytes, and these terms let the
        # candidates tell them apart
        self_sel = [s_ for s_, o in enumerate(OFFSETS) if o == 0][0]
        nb_sel[:, :, :self_terms] = self_sel
    ch_sel = torch.randint(0, K, (P, C, T), generator=g)
    co_sel = torch.where(torch.rand(P, C, T, generator=g) < 0.5, 0, 2)  # -1/+1
    imm = torch.where(torch.rand(P, C, generator=g) < 0.5, -1.0, 1.0) * \
        (4 + 12 * torch.rand(P, C, generator=g))
    with torch.no_grad():
        dev = model.op_code.device

        def put(name, idx):
            x = noise(getattr(model, name))
            x.scatter_add_(-1, idx[..., None],
                           torch.full(idx.shape + (1,), margin))
            getattr(model, name).copy_(x.to(dev))
        put("op_code", torch.full((P, C), spec.VSET if VW else spec.SET,
                                  dtype=torch.long))
        put("op_d", d)
        put("op_a", torch.ones(P, C, dtype=torch.long))             # a = 0
        put("sc_nb", nb_sel)
        put("sc_ch", ch_sel)
        put("sc_co", co_sel)
        model.op_b.copy_(imm.to(dev))
        if VW > 1:
            model.op_v.copy_((torch.where(
                torch.rand(P, C, VW - 1, generator=g) < 0.5, -1.0, 1.0) *
                (4 + 12 * torch.rand(P, C, VW - 1, generator=g))).to(dev))
        model.sc_bias.zero_()
    return {"quarter": quarter, "work_channels": nwork}


def structured_work_probes(n_work: int, ticks: int = 2,
                           skip_groups: tuple = (),
                           work_lags: str = "") -> list:
    """(lag, channel) for the work probes that match structured_rules.

    Two ticks per byte: the head's own features (phases 0 and 1: first half
    of the work channels) at lag 1, then the features the byte before got one
    byte later (phases 2 and 3: second half) at lag 2. Four or more ticks:
    the head runs all four phases itself, so every work channel is read at
    lag 1 and again at lag 2 (where a later tick has rewritten one channel
    of each group).

    An explicit ``work_lags`` uses lag:group-count entries. Each group is a
    complete VSET vector, and groups are spread across the available phases.
    For VSET=6, ``1:16,2:8,8:4,16:4`` allocates 96, 48, 24, 24 probes.
    Empty preserves the original layout exactly."""
    K = spec.TAPE_CHANNELS
    nwork = N_CHANNELS - K
    out = []
    if work_lags:
        width = spec.VSET_WIDTH
        if not width or nwork % width or n_work % width:
            raise ValueError("--work-lags needs whole MICA_VSET groups")
        groups = [g for g in range(nwork // width) if g not in skip_groups]
        if not groups:
            raise ValueError("--work-lags has no available work groups")
        seen = set()
        total = 0
        limit = min(spec.PROBE_WINDOW or N_CELLS - 1, N_CELLS - 1)
        for entry_index, item in enumerate(work_lags.split(",")):
            parts = item.strip().split(":")
            if len(parts) != 2:
                raise ValueError("--work-lags needs comma-separated lag:groups entries")
            try:
                lag, count = (int(part.strip()) for part in parts)
            except ValueError as exc:
                raise ValueError("--work-lags needs integer lag:groups entries") from exc
            if lag in seen or not 1 <= lag <= limit:
                raise ValueError(f"--work-lags lags must be distinct in 1..{limit}")
            if not 1 <= count <= len(groups):
                raise ValueError(f"--work-lags group counts must be in 1..{len(groups)}")
            seen.add(lag)
            total += count * width
            if total > n_work:
                raise ValueError(f"--work-lags specifies more than {n_work} probes")
            for j in range(count):
                group = groups[(j * len(groups) // count + entry_index) % len(groups)]
                out.extend((lag, K + group * width + slot)
                           for slot in range(width))
        if total != n_work:
            raise ValueError(f"--work-lags specifies {total} probes; "
                             f"this run needs exactly {n_work}")
        return out
    if ticks >= N_PHASE:
        width = spec.VSET_WIDTH or max(1, nwork // N_PHASE)
        chans = [K + c for c in range(nwork) if c // width not in skip_groups]
        for i in range(n_work):
            out.append((1 + i // len(chans), chans[i % len(chans)]))
        return out
    half = nwork // 2
    for i in range(n_work):
        if i < half:
            out.append((1, K + i))
        elif i < nwork:
            out.append((2, K + i))
        else:
            out.append((1 + (i // nwork) * 2, K + i % nwork))
    return out


def split_probe_layout(layout) -> tuple:
    """Classify probes into tape, topic and work.

    Tape probes read a context symbol's code. Work probes read a rule
    immediate (provenance). Topic probes read the simulated topic register:
    neither of those. With ``MICA_TOPIC`` unset, nothing is a topic probe
    and the other two classes are exactly the historical split.
    """
    K = spec.TAPE_CHANNELS
    topic_on = spec.TOPIC_CHANNELS
    top = spec.TOPIC_AT
    tape, topic, work = [], [], []
    for p, (lag, ch) in enumerate(layout):
        lag, ch = int(lag), int(ch)
        if topic_on and ch >= top:
            topic.append((p, lag, ch))
        elif ch < K:
            tape.append((p, lag, ch))
        else:
            work.append((p, lag, ch))
    return tape, topic, work


def probe_layout(model) -> list:
    """(lag, channel) of every probe as the integer machine reads it (the
    argmax of its selectors; the same for every symbol after tape init)."""
    with torch.no_grad():
        cell = model.logits_of("pr_cell").argmax(-1)[0].cpu()        # (P,)
        chan = model.pr_chan.argmax(-1)[0].cpu()
    return [(int((N_CELLS - c) % N_CELLS), int(h)) for c, h in zip(cell, chan)]


def _tables(model, dev):
    """The hard (argmax / rounded) values the integer machine uses."""
    from .discretise import to_integer
    m = to_integer(model)
    t = lambda a: torch.as_tensor(np.asarray(a).astype(np.int64), device=dev)
    return m, {k: t(getattr(m, k)) for k in
               ("sc_nb", "sc_ch", "sc_co", "sc_bias", "op_d", "op_code",
                "op_a", "inj_delta")}


def topic_step(prev: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    """engine.topic_step on integer tensors: decay one step toward zero,
    add the code, saturate. Exactly the engine's arithmetic."""
    prev = prev.long()
    d = prev - torch.sign(prev) * (prev.abs() >> spec.TOPIC_SHIFT)
    return (d + code.long()).clamp(SAT_MIN, SAT_MAX)


def _simulate(model, records, dev, ticks: int, batch: int = 256,
              on_tick=None, on_read=None):
    """Run the integer machine over `records` for a structured rule book
    (SET-immediate or VSET programs), tracking which candidate last wrote
    every work channel. Scoring may read any channel, tape or work, exactly
    as the machine does; values come from the model's own immediates.

    on_tick(page, scores, win, phase_rows) is called for the head cell's
    ticks; on_read(b_rows, t, prov, F, pos) before each target position is
    scored. Returns nothing; the callbacks collect what they need."""
    K = spec.TAPE_CHANNELS
    W = spec.WINDOW
    m, tb = _tables(model, dev)
    VW = spec.VSET_WIDTH
    if VW:
        assert (tb["op_code"] == spec.VSET).all(), "needs VSET programs"
    else:
        assert (tb["op_code"] == spec.SET).all() and (tb["op_a"] == 0).all(), \
            "needs SET-immediate programs"
    assert max(ROUTING_CHANNELS) + ROUTING_PAIR_OFFSET < K
    width = VW or 1
    imm = tb["op_b"][..., None] if "op_b" in tb else \
        torch.as_tensor(m.op_b.astype(np.int64), device=dev)[..., None]
    if VW > 1:
        imm = torch.cat([imm, torch.as_tensor(m.op_v.astype(np.int64),
                                              device=dev)], -1)    # (P,C,VW)
    codes = tb["inj_delta"][:, :K]
    # spec.TOPIC_CHANNELS: entries K..K+T-1 are topic codes, the rest work
    T, TOP = spec.TOPIC_CHANNELS, spec.TOPIC_AT
    tcodes = tb["inj_delta"][:, K:K + T]
    wdelta = tb["inj_delta"][:, K + T:]
    wcell = torch.as_tensor(m.inj_cell[:, K + T:].astype(np.int64), device=dev)
    wchan = torch.as_tensor(m.inj_chan[:, K + T:].astype(np.int64), device=dev)
    off = torch.tensor(OFFSETS, device=dev)
    rc = torch.tensor(ROUTING_CHANNELS, device=dev)
    rw = 1 << torch.arange(len(ROUTING_CHANNELS), device=dev)
    C = N_CANDIDATES
    tie = torch.arange(C, device=dev)
    jj = torch.arange(width, device=dev)
    # every write in range (always true for structured_rules): no masking,
    # so no device-to-host sync per tick
    in_range = bool((tb["op_d"] + width - 1 < N_CHANNELS).all())
    for i in range(0, len(records), batch):
        part = records[i:i + batch]
        B, L = len(part), max(len(r) for r in part)
        seq = torch.full((B, L), BOS, dtype=torch.long, device=dev)
        for j, r in enumerate(part):
            symbols = (np.frombuffer(r, np.uint8) if N_SYMBOLS == 258
                       else np.asarray(r, dtype=np.uint16))
            seq[j, :len(r)] = torch.as_tensor(symbols.astype(np.int64),
                                              device=dev)
        F = torch.zeros(B, N_CELLS, N_CHANNELS, dtype=torch.long, device=dev)
        phase = torch.zeros(B, N_CELLS, dtype=torch.long, device=dev)
        prov = torch.full((B, N_CELLS, N_CHANNELS - K), -1, dtype=torch.long,
                          device=dev)
        bi = torch.arange(B, device=dev)

        def ingest(sym, pos):
            head = pos % N_CELLS
            if T:
                prev = F[:, (head - 1) % N_CELLS, TOP:].clone()
            F[:, head] = 0
            F[:, head, :K] = codes[sym]
            if T:
                F[:, head, TOP:] = topic_step(prev, tcodes[sym])
            phase[:, head] = 0
            prov[:, head] = -1
            if wdelta.shape[1]:
                cell = (wcell[sym] + pos) % N_CELLS                    # (B,I')
                ch = wchan[sym]
                for e in range(wdelta.shape[1]):
                    v = F[bi, cell[:, e], ch[:, e]] + wdelta[sym, e]
                    F[bi, cell[:, e], ch[:, e]] = v.clamp(SAT_MIN, SAT_MAX)
            cells = (head - torch.arange(W, device=dev)) % N_CELLS     # (W,)
            for _ in range(ticks):
                x = F[:, cells, :K]                                    # (B,W,K)
                bits = ((x[..., rc] >= x[..., rc + ROUTING_PAIR_OFFSET])
                        .long() * rw).sum(-1)
                ph = phase[:, cells]
                page = PAGE_STRIDE * ph + bits                         # (B,W)
                src = (cells[None, :, None, None] +
                       off[tb["sc_nb"][page]]) % N_CELLS               # (B,W,C,T)
                vals = F[bi[:, None, None, None], src, tb["sc_ch"][page]]
                sc = tb["sc_bias"][page] + (tb["sc_co"][page] * vals).sum(-1)
                win = (sc * (C + 1) - tie).argmax(-1)                  # lowest on ties
                d = tb["op_d"][page, win]                              # (B,W)
                if on_tick is not None:
                    on_tick(page[:, 0], sc[:, 0], win[:, 0], ph[:, 0])
                vals_w = imm[page, win]                                # (B,W,width)
                chn = d[..., None] + jj                                # (B,W,width)
                who = (page * C + win)[..., None].expand_as(chn)
                if in_range:
                    bb = bi[:, None, None].expand_as(chn)
                    cc = cells[None, :, None].expand_as(chn)
                    F[bb, cc, chn] = vals_w
                    prov[bb, cc, chn - K] = who
                else:
                    ok = chn < N_CHANNELS
                    bb = bi[:, None, None].expand_as(chn)[ok]
                    cc = cells[None, :, None].expand_as(chn)[ok]
                    F[bb, cc, chn[ok]] = vals_w[ok]
                    prov[bb, cc, chn[ok] - K] = who[ok]
                phase[:, cells] = (ph + 1) % N_PHASE
            return pos + 1

        pos = ingest(torch.full((B,), BOS, dtype=torch.long, device=dev), 0)
        for t in range(L + 1):
            if on_read is not None:
                on_read(i, part, t, prov, F, pos)
            if t < L:
                pos = ingest(seq[:, t], pos)


def provenance(model, records, work, dev, ticks: int,
               batch: int = 256, by_group: int = 0) -> torch.Tensor:
    """For every target position of `records` (record by record, bytes then
    EOS: the order contexts() uses) and every work probe (lag, channel) in
    `work`: the flat index page * N_CANDIDATES + candidate of the immediate
    that channel holds when the probe reads it, or -1 when it holds 0.

    Valid while the rule book is structured (SET-immediate or VSET programs)
    and the scoring reads only channels whose values do not depend on the
    immediates being fitted (fit_readout_and_rules keeps those fixed).

    by_group=Q: `work` holds (lag, group) pairs instead, a group being Q
    consecutive work channels; the result is the candidate that wrote the
    group's first channel (with VSET, the whole group)."""
    K = spec.TAPE_CHANNELS
    lag = torch.tensor([l for l, _ in work], device=dev)
    ch = torch.tensor([(c * by_group if by_group else c - K) for _, c in work],
                      device=dev)
    out = {}
    # A batch runs in lockstep to its longest record. Group similar lengths
    # and restore original record order before returning the table.
    order = sorted(range(len(records)), key=lambda k: len(records[k]))
    ordered = [records[k] for k in order]

    def on_read(i, part, t, prov, F, pos):
        # int32: a page*candidate index fits, and at 32 phases x 2 lags the
        # table is 64 columns per position -- half the memory of int64
        buf = out.setdefault(i, torch.empty(len(part), max(len(r) for r in part)
                                            + 1, len(work), dtype=torch.int32,
                                            device=dev))
        buf[:, t] = prov[:, (pos - lag) % N_CELLS, ch].int()
    _simulate(model, ordered, dev, ticks, batch=batch, on_read=on_read)
    rows = [None] * len(records)
    for i in sorted(out):
        for j, r in enumerate(ordered[i:i + batch]):
            rows[order[i + j]] = out[i][j, :len(r) + 1]
    return torch.cat(rows)


def contexts(records, lags, dev):
    """Symbols at each lag before every target (record by record, bytes then
    EOS), and the targets. N_SYMBOLS stands for "no byte here yet"."""
    max_lag = max(lags)
    ZERO = N_SYMBOLS
    ctx, tgt = [], []
    for r in records:
        a = (np.frombuffer(r, np.uint8).astype(np.int32) if N_SYMBOLS == 258
             else np.asarray(r, dtype=np.int32))
        seq = np.concatenate([np.full(max_lag, ZERO, np.int32),
                              np.array([BOS], np.int32), a])
        t = np.arange(len(a) + 1)
        ctx.append(np.stack([seq[max_lag + t + 1 - lag] for lag in lags], 1))
        tgt.append(np.concatenate([a, np.array([EOS], np.int32)]))
    return (torch.from_numpy(np.concatenate(ctx)).long().to(dev),
            torch.from_numpy(np.concatenate(tgt)).long().to(dev))


def fit_readout_and_rules(model, records, dev, ticks: int, steps: int = 3000,
                          batch: int = 16384, max_records: int = 20_000,
                          check_every: int = 250, lr_scale: float = 1.0,
                          freeze_phase0_state: bool = False,
                          freeze_phase1_state: bool = False,
                          word_start_weight: float = 1.0,
                          log=print) -> float:
    """Jointly fit every probe coefficient, the readout biases and (with a
    structured rule book) every rule immediate, with the file's rounding.
    The byte codes stay fixed: they decide the routing and the scoring, and
    so which immediate each cell holds. When phase 0 writes a recurrent state,
    its immediates must stay fixed too: changing them would invalidate the
    provenance measured before fitting. An optional weight emphasizes the
    first byte of a word after a space or BOS. Returns held-out bits/byte
    under that weight (ordinary bits/byte when the weight is 1)."""
    if N_SYMBOLS > 258:
        if word_start_weight != 1:
            raise ValueError("byte word-start weighting is invalid for word symbols")
        batch = min(batch, 512)
    if word_start_weight < 1:
        raise ValueError("word_start_weight must be at least 1")
    K = spec.TAPE_CHANNELS
    D = spec.LOGIT_DIVISOR
    layout = probe_layout(model)
    # Third probe kind: a channel in the topic register is the simulated
    # fading sum, not a tape code and not a rule immediate. With MICA_TOPIC
    # unset this split is the old tape/work split and topic_feat stays None.
    tape_p, topic_p, work_p = split_probe_layout(layout)
    stride = max(1, len(records) // max_records)
    sample = [r for r in records[::stride] if len(r)]
    ulags = sorted({l for _, l, _ in tape_p})
    if word_start_weight != 1:
        ulags = sorted(set(ulags) | {1})
    t0 = time.time()
    ctx, tgt = contexts(sample, ulags, dev)
    lcol = torch.tensor([ulags.index(l) for _, l, _ in tape_p], device=dev)
    tch = torch.tensor([c for _, _, c in tape_p], device=dev)
    tpi = torch.tensor([p for p, _, _ in tape_p], device=dev)
    prov = None
    if work_p:
        width_ = spec.VSET_WIDTH or 1
        groups_ = sorted({(l, (c - K) // width_) for _, l, c in work_p})
        prov = provenance(model, sample, groups_, dev, ticks, by_group=width_)
        assert prov.shape[0] == tgt.shape[0], (prov.shape, tgt.shape)
        used = (prov >= 0).float().mean().item()
        log(f"[fit] rule features: {len(work_p)} work probes, {used:.0%} of "
            f"their reads hold a rule's immediate ({time.time() - t0:.0f}s)")
    wpi = torch.tensor([p for p, _, _ in work_p], device=dev, dtype=torch.long)
    width = spec.VSET_WIDTH or 1
    groups = sorted({(l, (c - K) // width) for _, l, c in work_p})
    gidx = {g: i for i, g in enumerate(groups)}
    wg = torch.tensor([gidx[(l, (c - K) // width)] for _, l, c in work_p],
                      device=dev, dtype=torch.long)
    wslot = torch.tensor([(c - K) % width for _, _, c in work_p], device=dev,
                         dtype=torch.long)
    n = tgt.shape[0]
    row_weight = None
    if word_start_weight != 1:
        # This objective asks the model to choose a word given the preceding
        # words. Most byte positions instead reward spelling inside a word.
        previous = ctx[:, ulags.index(1)]
        letter = (((tgt >= 65) & (tgt <= 90)) |
                  ((tgt >= 97) & (tgt <= 122)) |
                  ((tgt >= 48) & (tgt <= 57)) |
                  ((tgt >= 128) & (tgt <= 255)))
        start = ((previous == ord(" ")) | (previous == BOS)) & letter
        row_weight = torch.where(start, float(word_start_weight), 1.0)
        log(f"[fit] word-start weighting: {int(start.sum())}/{n} targets "
            f"at {word_start_weight:g}x")
    n_hold = min(n // 10, 200_000)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)
    hold, fit = perm[:n_hold], perm[n_hold:]
    elig = torch.full((N_SYMBOLS,), float("-inf"), device=dev)
    elig[:BOS] = 0.0
    elig[EOS] = 0.0

    def rnd(x, lo, hi):
        return x + (x.detach().round().clamp(lo, hi) - x.detach())

    with torch.no_grad():
        code = model.inj_delta[:, :K].detach().round().clamp(-127, 127).to(dev)
        code = torch.cat([code, torch.zeros(N_SYMBOLS + 1 - code.shape[0], K,
                                            device=dev)])
    topic_feat = None
    topic_pi = None
    if topic_p:
        from .engine import topic_probe_features
        codes_np = (model.inj_delta[:, K:K + spec.TOPIC_CHANNELS].detach()
                    .round().clamp(-127, 127).cpu().numpy().astype(np.int8))
        feat_np = topic_probe_features(sample, codes_np, topic_p, spec.TOPIC_SHIFT)
        if feat_np.shape[0] != int(tgt.shape[0]):
            raise RuntimeError(f"topic features {feat_np.shape} do not match "
                               f"{tuple(tgt.shape)} targets")
        topic_feat = torch.as_tensor(feat_np, dtype=torch.float32, device=dev)
        topic_pi = torch.tensor([p for p, _, _ in topic_p], device=dev,
                                dtype=torch.long)
        log(f"[fit] topic probes: {len(topic_p)} read the simulated register, "
            f"not tape codes or rule immediates")
    w = model.pr_w.detach().clone().to(dev).requires_grad_()
    bias = model.pr_bias.detach().clone().to(dev).requires_grad_()
    imm0 = model.op_b.detach().clone()[..., None]
    if spec.VSET_WIDTH > 1:
        imm0 = torch.cat([imm0, model.op_v.detach().clone()], -1)
    imm0 = imm0.reshape(-1, width).to(dev)
    imm = imm0.clone().requires_grad_()
    state_phase = (torch.arange(N_PAGES * N_CANDIDATES, device=dev) //
                   N_CANDIDATES // PAGE_STRIDE)
    state_rows = (state_phase < (2 if freeze_phase1_state else 1))[:, None]

    def effective_immediates():
        return torch.where(state_rows, imm0, imm) if freeze_phase0_state else imm
    groups = [{"params": [w], "lr": 1.0 * lr_scale},
              {"params": [bias], "lr": 0.01 * D * lr_scale},
              {"params": [imm], "lr": 1.0 * lr_scale}]
    opt = torch.optim.Adam(groups)

    def logits(rows):
        wr = rnd(w, -127, 127)
        sc = rnd(bias, -32767, 32767) + \
            code[ctx[rows][:, lcol], tch[None, :]] @ wr[:, tpi].T
        if prov is not None:
            pv = prov[rows][:, wg].long()                         # (n,Q)
            iv = torch.where(pv >= 0,
                             rnd(effective_immediates(), -127, 127)[
                                 pv.clamp(min=0), wslot],
                             torch.zeros((), device=dev))
            sc = sc + iv @ wr[:, wpi].T
        if topic_feat is not None:
            sc = sc + topic_feat[rows] @ wr[:, topic_pi].T
        return sc / D + elig

    def held_out():
        with torch.no_grad():
            chunk = 512 if N_SYMBOLS > 258 else 32768
            if row_weight is None:
                return sum(float(Fn.cross_entropy(logits(hold[i:i + chunk]),
                                                  tgt[hold[i:i + chunk]],
                                                  reduction="sum"))
                           for i in range(0, n_hold, chunk)) / n_hold / math.log(2)
            total = 0.0
            for i in range(0, n_hold, chunk):
                ids = hold[i:i + chunk]
                total += float((Fn.cross_entropy(logits(ids), tgt[ids],
                                                 reduction="none") *
                                row_weight[ids]).sum())
            return total / float(row_weight[hold].sum()) / math.log(2)
    # Early stopping on the held-out positions: the parameters written back
    # are the best ones seen, starting with the ones we came in with, so a
    # round on a small or unlucky sample can never make the model worse on
    # its own held-out text (a 400-record fit went 3.47 -> 3.82 without this).
    h = held_out()
    log(f"[fit] start: held-out {h:.4f} bits/byte")
    best_h = h
    keep = [t.detach().clone() for t in (w, bias, imm)]
    for it in range(steps):
        rows = fit[torch.randint(0, len(fit), (batch,), device=dev)]
        if row_weight is None:
            loss = Fn.cross_entropy(logits(rows), tgt[rows])
        else:
            weights = row_weight[rows]
            loss = (Fn.cross_entropy(logits(rows), tgt[rows],
                                     reduction="none") * weights).sum() / weights.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % check_every == 0 or it == steps - 1:
            h = held_out()
            if h < best_h:
                best_h = h
                keep = [t.detach().clone() for t in (w, bias, imm)]
            if (it + 1) % 1000 == 0 or it == steps - 1:
                log(f"[fit] step {it + 1:5d}  held-out {h:.4f} bits/byte "
                    f"(best {best_h:.4f}, {time.time() - t0:.0f}s)")
    w, bias, imm = keep
    if freeze_phase0_state:
        imm = torch.where(state_rows, imm0, imm)
    with torch.no_grad():
        model.pr_w.copy_(w.round().clamp(-127, 127).to(model.pr_w))
        model.pr_bias.copy_(bias.round().clamp(-32767, 32767).to(model.pr_bias))
        ir = imm.round().clamp(-127, 127).reshape(N_PAGES, N_CANDIDATES, width)
        model.op_b.copy_(ir[..., 0].to(model.op_b))
        if spec.VSET_WIDTH > 1:
            model.op_v.copy_(ir[..., 1:].to(model.op_v))
    return best_h


def _page_score_row_blocks(model, records, dev, cells=None):
    """_page_score_rows one cell at a time: yields (page, scores) for each
    cell in turn, so a caller can drop a phase's rows before the next."""
    K = spec.TAPE_CHANNELS
    # every phase, at the head: with tape-only scoring a page's score rows
    # have the same distribution at any cell, so the head stands for all
    cells = cells or tuple((1, p) for p in range(N_PHASE))
    m, tb = _tables(model, dev)
    assert (tb["sc_ch"] < K).all(), "needs tape-only scoring"
    dist_of_sel = torch.tensor([-o for o in OFFSETS], device=dev)     # sel -> dist
    max_d = int(dist_of_sel.max())
    lags = list(range(1, max(l for l, _ in cells) + max_d + 1))
    ctx, _ = contexts(records, lags, dev)                              # (n,lags)
    code = torch.cat([tb["inj_delta"][:, :K],
                      torch.zeros(N_SYMBOLS + 1 - tb["inj_delta"].shape[0], K,
                                  dtype=torch.long, device=dev)])
    rc = torch.tensor(ROUTING_CHANNELS, device=dev)
    rw = 1 << torch.arange(len(ROUTING_CHANNELS), device=dev)
    for lag, ph in cells:
        x = code[ctx[:, lag - 1]]                                      # (n,K)
        bits = ((x[:, rc] >= x[:, rc + ROUTING_PAIR_OFFSET]).long() * rw).sum(-1)
        page = PAGE_STRIDE * ph + bits                                 # (n,)
        S = torch.empty(page.shape[0], N_CANDIDATES, dtype=torch.int32,
                        device=dev)
        for i in range(0, page.shape[0], 8192):
            pg = page[i:i + 8192]
            col = lag - 1 + dist_of_sel[tb["sc_nb"][pg]]               # (b,C,T)
            sym = torch.gather(ctx[i:i + 8192][:, None, None, :].expand(
                -1, N_CANDIDATES, N_SCORE_TERMS, -1), 3, col[..., None])[..., 0]
            vals = code[sym, tb["sc_ch"][pg]]                          # (b,C,T)
            S[i:i + 8192] = (tb["sc_co"][pg] * vals).sum(-1).int()
        yield page, S


def _page_score_rows(model, records, dev, cells=None):
    """Every (page, candidate-score vector) the structured rule book sees at
    the cells the work probes read: the head's two ticks (phases 0 and 1) and
    the ticks of the cell one byte older (phases 2 and 3). Scores read tape
    channels only, so they follow from the bytes alone -- no simulation."""
    pages, scores = [], []
    for page, S in _page_score_row_blocks(model, records, dev, cells):
        pages.append(page)
        scores.append(S)
    return torch.cat(pages), torch.cat(scores)


def _bounded_bucket_sample(records: list[bytes], max_positions: int) -> list[bytes]:
    """Keep an even, deterministic spread of records within a score-row budget.

    Each byte (plus at most one BOS position) yields a row for every phase.
    The bucket fitter keeps all score rows on GPU and concatenates them once,
    so a record-count cap alone is unsafe when corpus record lengths change.
    """
    if max_positions < 1:
        raise ValueError("score-row budget must allow at least one position")
    positions = sum(len(record) + 1 for record in records)
    if positions <= max_positions:
        return records
    count = max(1, len(records) * max_positions // positions)
    while True:
        selected = [records[i * len(records) // count] for i in range(count)]
        if (sum(len(record) + 1 for record in selected) <= max_positions or
                count == 1):
            return selected
        count -= 1


def balance_buckets(model, records, dev, iters: int = 80,
                    max_records: int = 3000, log=print) -> dict:
    """Set each candidate's scoring bias so that, page by page, the candidates
    win about equally often on real text.

    Measured on the random structured rule book: of a page's 128 candidates
    only ~25 ever win, the most common one takes 39% of the page's cells,
    and the win distribution is worth ~13 buckets. Fitted immediates can only
    tell apart contexts that land in different buckets, so this is capacity
    thrown away. The bias is the one scoring parameter that shifts where a
    candidate wins without changing what it reads; a few dozen rounds of
    "lower the bias of candidates that win too often, raise the rest" spread
    the page's contexts over its candidates (dual ascent on a balanced
    assignment). Returns before/after effective bucket counts.

    A page belongs to one phase, and a page's bias moves only with its own
    win counts, so the ascent runs phase by phase: the same biases as running
    all phases together, holding one phase's score rows at a time."""
    # one score row per record byte per phase: keep the rows about constant
    # whatever the phase count
    max_records = max(500, max_records * 4 // N_PHASE)
    stride = max(1, len(records) // max_records)
    sample = [r for r in records[::stride] if len(r)]
    # safety net for very long records: at most 1.5 GiB of int32 score rows
    # per phase (the rows of one phase are all that is held at once)
    score_budget = (1536 << 20) // (4 * N_CANDIDATES)
    bounded = _bounded_bucket_sample(sample, score_budget)
    if len(bounded) != len(sample):
        log(f"[fit] bucket score rows capped: {len(sample):,} -> "
            f"{len(bounded):,} records, {sum(len(r) + 1 for r in bounded):,} "
            f"of {score_budget:,} budgeted positions per phase")
    sample = bounded
    P, C = N_PAGES, N_CANDIDATES
    beta = model.sc_bias.detach().round().to(dev).int().clone()
    eligible = beta > -20000
    tie = torch.arange(C, device=dev, dtype=torch.int32)

    def eff(cnt):
        used = cnt.sum(1) > 0
        p = cnt[used].float() / cnt[used].sum(1, keepdim=True).float()
        return float((-(p * p.clamp(min=1e-12).log()).sum(1)).exp().mean())

    cnt0 = torch.zeros(P * C, dtype=torch.long, device=dev)
    cnt1 = torch.zeros(P * C, dtype=torch.long, device=dev)
    rows = 0
    for pages, S in _page_score_row_blocks(model, sample, dev):
        rows += S.shape[0]

        def winners():
            out = torch.empty(S.shape[0], dtype=torch.long, device=dev)
            for i in range(0, S.shape[0], 65536):
                sc = (S[i:i + 65536] + beta[pages[i:i + 65536]]).long()
                out[i:i + 65536] = (sc * (C + 1) - tie).argmax(-1)
            return out

        cnt0 += torch.bincount(pages * C + winners(), minlength=P * C)
        for it in range(iters):
            cnt = torch.bincount(pages * C + winners(),
                                 minlength=P * C).reshape(P, C)
            target = cnt.sum(1, keepdim=True).float() / \
                eligible.sum(1, keepdim=True).clamp(min=1)
            step = max(1, 64 >> (it // 10))
            beta -= (step * torch.sign(cnt.float() - target)).int() * \
                eligible.int() * (cnt.sum(1, keepdim=True) > 0).int()
        cnt1 += torch.bincount(pages * C + winners(), minlength=P * C)
        del pages, S
    before = eff(cnt0.reshape(P, C))
    after = eff(cnt1.reshape(P, C))
    with torch.no_grad():
        model.sc_bias.copy_(beta.float().clamp(-32767, 32767).to(model.sc_bias))
    log(f"[fit] rule buckets balanced: effective buckets per page "
        f"{before:.1f} -> {after:.1f} ({rows:,} cell-ticks)")
    return {"before": before, "after": after}


def choose_channels(model, records, dev, ticks: int, steps: int = 2000,
                    batch: int = 16384, max_records: int = 20_000,
                    log=print) -> float:
    """Let every rule candidate pick WHICH channel of its group it writes.

    A candidate's contribution to the output is its immediate times the
    coefficient vector of the probe that reads its channel, and those vectors
    are shared by every page: with the channel fixed as k mod Q, a bucket
    must use whatever direction its index dealt it. Here each bucket first
    gets one immediate per channel of its group (a relaxation the machine
    cannot run: it writes one channel per tick), all fitted jointly with the
    probe coefficients; then each keeps only the channel where its fitted
    contribution is largest. fit_readout_and_rules then refits the real,
    one-channel machine. Returns the relaxation's held-out bits."""
    K = spec.TAPE_CHANNELS
    D = spec.LOGIT_DIVISOR
    Q = max(1, (N_CHANNELS - K) // N_PHASE)
    C = N_CANDIDATES
    if not spec.VSET_WIDTH:
        raise NotImplementedError(
            "choose_channels needs group-level provenance for single-channel "
            "writes, which the general simulator does not track; with VSET "
            "every candidate writes its whole group and it is not needed")
    layout = probe_layout(model)
    tape_p, topic_p, work_p = split_probe_layout(layout)
    assert work_p, "no work probes"
    groups = sorted({(l, (c - K) // Q) for _, l, c in work_p})
    gidx = {g: i for i, g in enumerate(groups)}
    stride = max(1, len(records) // max_records)
    sample = [r for r in records[::stride] if len(r)]
    ulags = sorted({l for _, l, _ in tape_p})
    t0 = time.time()
    ctx, tgt = contexts(sample, ulags, dev)
    lcol = torch.tensor([ulags.index(l) for _, l, _ in tape_p], device=dev)
    tch = torch.tensor([c for _, _, c in tape_p], device=dev)
    tpi = torch.tensor([p for p, _, _ in tape_p], device=dev)
    wpi = torch.tensor([p for p, _, _ in work_p], device=dev)
    wg = torch.tensor([gidx[(l, (c - K) // Q)] for _, l, c in work_p], device=dev)
    wslot = torch.tensor([(c - K) % Q for _, _, c in work_p], device=dev)
    prov = provenance(model, sample, groups, dev, ticks, by_group=Q)
    n = tgt.shape[0]
    n_hold = min(n // 10, 4096 if N_SYMBOLS > 258 else 200_000)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1)).to(dev)
    hold, fit = perm[:n_hold], perm[n_hold:]
    elig = torch.full((N_SYMBOLS,), float("-inf"), device=dev)
    elig[:BOS] = 0.0
    elig[EOS] = 0.0

    def rnd(x, lo, hi):
        return x + (x.detach().round().clamp(lo, hi) - x.detach())

    with torch.no_grad():
        code = model.inj_delta[:, :K].detach().round().clamp(-127, 127).to(dev)
        code = torch.cat([code, torch.zeros(N_SYMBOLS + 1 - code.shape[0], K,
                                            device=dev)])
        topic_feat = None
        topic_pi = None
        if topic_p:
            from .engine import topic_probe_features
            codes_np = (model.inj_delta[:, K:K + spec.TOPIC_CHANNELS].detach()
                        .round().clamp(-127, 127).cpu().numpy().astype(np.int8))
            feat_np = topic_probe_features(sample, codes_np, topic_p,
                                           spec.TOPIC_SHIFT)
            topic_feat = torch.as_tensor(feat_np, dtype=torch.float32, device=dev)
            topic_pi = torch.tensor([p for p, _, _ in topic_p], device=dev,
                                    dtype=torch.long)
        d_now = model.logits_of("op_d").argmax(-1).reshape(-1).to(dev) - K
        imm0 = torch.zeros(N_PAGES * C, Q, device=dev)
        imm0[torch.arange(N_PAGES * C, device=dev), d_now % Q] = \
            model.op_b.detach().reshape(-1).to(dev)
    w = model.pr_w.detach().clone().to(dev).requires_grad_()
    bias = model.pr_bias.detach().clone().to(dev).requires_grad_()
    imm = imm0.clone().requires_grad_()
    opt = torch.optim.Adam([{"params": [w], "lr": 1.0},
                            {"params": [bias], "lr": 0.01 * D},
                            {"params": [imm], "lr": 1.0}])

    def logits(rows):
        wr = rnd(w, -127, 127)
        sc = rnd(bias, -32767, 32767) + \
            code[ctx[rows][:, lcol], tch[None, :]] @ wr[:, tpi].T
        pv = prov[rows][:, wg].long()                              # (n,Qw)
        iv = torch.where(pv >= 0, rnd(imm, -127, 127)[pv.clamp(min=0), wslot],
                         torch.zeros((), device=dev))
        sc = sc + iv @ wr[:, wpi].T
        if topic_feat is not None:
            sc = sc + topic_feat[rows] @ wr[:, topic_pi].T
        return sc / D + elig

    def held_out():
        with torch.no_grad():
            return sum(float(Fn.cross_entropy(logits(hold[i:i + 32768]),
                                              tgt[hold[i:i + 32768]],
                                              reduction="sum"))
                       for i in range(0, n_hold, 32768)) / n_hold / math.log(2)
    best_h, keep = held_out(), None
    for it in range(steps):
        rows = fit[torch.randint(0, len(fit), (batch,), device=dev)]
        loss = Fn.cross_entropy(logits(rows), tgt[rows])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % 250 == 0 or it == steps - 1:
            h = held_out()
            if h < best_h:
                best_h, keep = h, (w.detach().clone(), imm.detach().clone())
    if keep is None:
        log("[fit] channel choice: the relaxation did not improve; unchanged")
        return best_h
    w_best, imm_best = keep
    # contribution of slot s for a bucket: |immediate| x |coefficient vector|
    wn = w_best.round().clamp(-127, 127).norm(dim=0)                # (P,)
    slot_norm = torch.zeros(len(groups), Q, device=dev)
    slot_norm[wg, wslot] = wn[wpi]
    page_of = torch.arange(N_PAGES * C, device=dev) // C
    phase_of = page_of // PAGE_STRIDE
    # a bucket's group is its page's phase; find which (lag, group) row reads it
    row_of_group = torch.full((N_PHASE,), -1, dtype=torch.long, device=dev)
    for (l, g), i in gidx.items():
        if row_of_group[g] < 0:
            row_of_group[g] = i
    r = row_of_group[phase_of]
    contrib = imm_best.abs() * torch.where(r[:, None] >= 0,
                                           slot_norm[r.clamp(min=0)],
                                           torch.zeros((), device=dev))
    slot = contrib.argmax(-1)                                        # (P*C,)
    moved = float((slot != d_now % Q).float().mean())
    with torch.no_grad():
        new_d = (K + phase_of * Q + slot).reshape(N_PAGES, C)
        x = model.op_d.detach().clone()
        mg = x.max(-1, keepdim=True).values - x.min(-1, keepdim=True).values
        x.scatter_add_(-1, new_d[..., None].to(x.device),
                       mg + 3.0 * torch.ones_like(mg))
        model.op_d.copy_(x)
        model.op_b.copy_(imm_best[torch.arange(N_PAGES * C, device=dev), slot]
                         .round().clamp(-127, 127).reshape(N_PAGES, C)
                         .to(model.op_b))
        model.pr_w.copy_(w_best.round().clamp(-127, 127).to(model.pr_w))
    log(f"[fit] channel choice: relaxation held-out {best_h:.4f}; "
        f"{moved:.0%} of candidates moved to another channel "
        f"({time.time() - t0:.0f}s)")
    return best_h


def template_rules(model, records, dev, max_back: tuple = (1, 2, 2, 2),
                   max_records: int = 20_000, margin: float = 3.0,
                   ranking: str = "frequency",
                   phases: tuple[int, ...] | None = None,
                   log=print) -> dict:
    """Data-driven scoring for the structured rule book: each page's
    candidates match the page's most frequent contexts.

    A page is (phase, the byte at the cell). Its contexts are the bytes 1..m
    cells older (m = max_back[phase]). Candidate k gets the k-th most frequent
    context as a template: its six scoring terms read the channels where
    those bytes' tape codes are largest, with their signs, and its bias
    subtracts the template's own score, so the template scores 0 on its own
    context and less elsewhere. Frequent contexts then get a bucket each
    (an exact trigram bucket for phase 0, which looks one byte back); rarer
    ones fall to the nearest template. Ties go to the lower index, which is
    the more frequent context."""
    K = spec.TAPE_CHANNELS
    m, tb = _tables(model, dev)
    code = torch.cat([tb["inj_delta"][:, :K],
                      torch.zeros(N_SYMBOLS + 1 - tb["inj_delta"].shape[0], K,
                                  dtype=torch.long, device=dev)])
    back = {-o: s for s, o in enumerate(OFFSETS) if o < 0}
    dists = sorted(back)
    stride = max(1, len(records) // max_records)
    sample = [r for r in records[::stride] if len(r)]
    maxm = max(max_back)
    if ranking not in ("frequency", "information"):
        raise ValueError("template ranking must be frequency or information")
    if phases is None:
        phases = tuple(range(N_PHASE))
    if not phases or any(phase < 0 or phase >= N_PHASE for phase in phases):
        raise ValueError("template phases must be nonempty and in range")
    ctx, tgt = contexts(sample, list(range(1, 2 + dists[maxm - 1] + 1)), dev)
    rc = torch.tensor(ROUTING_CHANNELS, device=dev)
    rw = 1 << torch.arange(len(ROUTING_CHANNELS), device=dev)
    P, C, T = N_PAGES, N_CANDIDATES, N_SCORE_TERMS
    nb = tb["sc_nb"].clone(); ch = tb["sc_ch"].clone()
    co = tb["sc_co"].clone() + 1                    # values -1/0/1 -> indices
    bias = torch.zeros(P, C, dtype=torch.long, device=dev)
    has_template = torch.zeros(P, C, dtype=torch.bool, device=dev)
    V = N_SYMBOLS + 1
    covered = []
    for phase in range(N_PHASE):
        if phase not in phases:
            continue
        mb = min(max_back[phase % len(max_back)], len(dists))
        # With WINDOW=1 all phase ticks occur on the same current-byte cell.
        # The lag-2 split applies only to the older two-cell schedule.
        lag = 1 if spec.WINDOW == 1 or phase < 2 else 2
        x = code[ctx[:, lag - 1]]
        page = PAGE_STRIDE * phase + ((x[:, rc] >= x[:, rc + ROUTING_PAIR_OFFSET])
                                      .long() * rw).sum(-1)
        if N_SYMBOLS > 258:
            # A flattened base-V key overflows int64 for four word symbols.
            # Unique rows preserve the full hard predicate without hashing.
            key = torch.stack([page] + [ctx[:, lag - 1 + dists[j]]
                                         for j in range(mb)], dim=1)
            uk, inverse, cnt = torch.unique(
                key, dim=0, return_inverse=True, return_counts=True)
            upage = uk[:, 0]
        else:
            key = page.clone()
            for j in range(mb):
                key = key * V + ctx[:, lag - 1 + dists[j]]
            uk, inverse, cnt = torch.unique(
                key, return_inverse=True, return_counts=True)
            upage = uk
            for j in range(mb):
                upage = upage // V
        if ranking == "information":
            if N_SYMBOLS > 258:
                raise ValueError("information template ranking needs a sparse word histogram")
            # Pick contexts whose next-byte distribution differs from their
            # page's baseline. A page-conditioned Dirichlet prior limits the
            # reward for sparse contexts; a simple support penalty keeps
            # one-off next bytes from monopolising the rule slots.
            page_hist = torch.bincount(page * V + tgt, minlength=P * V)
            page_hist = page_hist.reshape(P, V).float()
            baseline = (page_hist[upage] + 0.5) / \
                (page_hist.sum(-1)[upage, None] + 0.5 * V)
            hist = torch.bincount(inverse * V + tgt,
                                  minlength=len(uk) * V)
            hist = hist.reshape(len(uk), V).float()
            posterior = (hist + 12.0 * baseline) / \
                (cnt[:, None] + 12.0)
            gain = (hist * (posterior.clamp_min(1e-12).log() -
                            baseline.clamp_min(1e-12).log())).sum(-1)
            support = (hist > 0).sum(-1).float()
            gain -= 0.5 * (support - 1).clamp(min=0)
            gain = torch.where(cnt >= 3, gain,
                               torch.full_like(gain, -1e9))
            order = torch.argsort(gain, descending=True, stable=True)
            order = order[torch.argsort(upage[order], stable=True)]
        else:
            order = torch.argsort(upage * (cnt.max() + 1) - cnt)
        uk, cnt, upage = uk[order], cnt[order], upage[order]
        # rank within page
        first = torch.ones_like(upage, dtype=torch.bool)
        first[1:] = upage[1:] != upage[:-1]
        start = torch.cummax(torch.where(first, torch.arange(len(upage), device=dev),
                                         torch.zeros_like(upage)), 0).values
        rank = torch.arange(len(upage), device=dev) - start
        keep = rank < C
        uk, upage, rank = uk[keep], upage[keep], rank[keep]
        covered.append(float(cnt[keep].sum()) / float(cnt.sum()))
        # decode the context symbols, nearest byte first
        if N_SYMBOLS > 258:
            syms = [uk[:, j + 1] for j in range(mb)]
        else:
            syms = []
            rest = uk.clone()
            for j in range(mb):
                syms.append(rest % V)
                rest = rest // V
            syms = syms[::-1]                                       # dist 1 .. mb
        # terms per distance: spread T terms over mb distances, nearer first
        per = [T // mb + (1 if j < T % mb else 0) for j in range(mb)]
        t = 0
        tb_nb, tb_ch, tb_co = [], [], []
        own = torch.zeros(len(uk), dtype=torch.long, device=dev)
        for j in range(mb):
            cv = code[syms[j]]                                      # (u,K)
            top = cv.abs().argsort(-1, descending=True)[:, :per[j]]  # (u,per)
            vals = torch.gather(cv, 1, top)
            tb_nb.append(torch.full_like(top, back[dists[j]]))
            tb_ch.append(top)
            tb_co.append(torch.where(vals >= 0, 2, 0))              # +1 / -1
            own += vals.abs().sum(-1)
        nb[upage, rank] = torch.cat(tb_nb, 1)
        ch[upage, rank] = torch.cat(tb_ch, 1)
        co[upage, rank] = torch.cat(tb_co, 1)
        bias[upage, rank] = -own
        has_template[upage, rank] = True
    # in a page with templates, a leftover random candidate could outscore
    # every template (they peak at 0): keep it out of the running
    page_has = has_template.any(1, keepdim=True)
    bias = torch.where(page_has & ~has_template,
                       torch.full_like(bias, -30000), bias)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        def put(name, idx):
            p = getattr(model, name)
            x = torch.randn(p.shape, generator=g) * 0.05
            x.scatter_add_(-1, idx.cpu()[..., None],
                           torch.full(idx.shape + (1,), margin))
            p.copy_(x.to(p.device))
        put("sc_nb", nb)
        put("sc_ch", ch)
        put("sc_co", co)
        model.sc_bias.copy_(bias.float().to(model.sc_bias))
    log(f"[fit] template rules ({ranking}, phases {phases}): "
        f"{C} contexts per page "
        f"cover {', '.join(f'{c:.0%}' for c in covered)} of cells by phase")
    return {"covered": covered}


def from_integer(m, ticks: int, margin: float = 6.0, compact: bool = None):
    """A SoftMica whose hard forward pass is exactly the integer model `m`
    (every selector's logit peaked at the file's choice, every integer
    parameter copied). For analysis, sampling, and continuing training from
    a .mica file.

    compact (default: word alphabets) keeps one shared row of injection and
    probe selectors, as fit mode trains word models; the model's wiring must
    then be the same for every symbol. At 16,384 symbols the full rows would
    need over 12 GB."""
    from .soft import SoftMica
    if compact is None:
        compact = N_SYMBOLS > 258
    sm = SoftMica(ticks=ticks, tau=0.5, hard=True, sel_init=0.0, sel_tau=1.0,
                  compact_selectors=compact)
    return load_integer(sm, m, margin)


def load_integer(sm, m, margin: float = 6.0):
    """Copy integer model `m` into an existing SoftMica `sm` (any device), so
    that sm's hard forward pass is exactly `m`. Returns sm."""
    compact = bool(getattr(sm, "compact_selectors", False))

    def rows(a):
        a = np.asarray(a)
        if not compact:
            return a
        if not (a == a[:1]).all():
            raise ValueError("compact selectors need the same wiring for every symbol")
        return a[:1]

    def peak(name, idx):
        p = getattr(sm, name)
        x = torch.zeros(p.shape, device=p.device)
        x.scatter_(-1, torch.as_tensor(np.asarray(idx).astype(np.int64),
                                       device=p.device)[..., None], margin)
        p.data.copy_(x)
    with torch.no_grad():
        peak("inj_cell", rows(m.inj_cell)); peak("inj_chan", rows(m.inj_chan))
        sm.inj_delta.copy_(torch.as_tensor(m.inj_delta.astype(np.float32)))
        peak("sc_nb", m.sc_nb); peak("sc_ch", m.sc_ch)
        peak("sc_co", m.sc_co.astype(np.int64) + 1)
        sm.sc_bias.copy_(torch.as_tensor(m.sc_bias.astype(np.float32)))
        peak("op_code", m.op_code); peak("op_d", m.op_d); peak("op_n", m.op_n)
        peak("op_c", m.op_c); peak("op_u", m.op_u)
        peak("op_a", m.op_a.astype(np.int64) + 1)
        sm.op_b.copy_(torch.as_tensor(m.op_b.astype(np.float32)))
        if spec.VSET_WIDTH > 1:
            sm.op_v.copy_(torch.as_tensor(m.op_v.astype(np.float32)))
        peak("pr_cell", rows(m.pr_cell)); peak("pr_chan", rows(m.pr_chan))
        if spec.WIDE_PROBE_COEF:
            sm.pr_w.copy_(torch.as_tensor(m.pr_co.astype(np.float32)))
        else:
            peak("pr_co", rows(m.pr_co.astype(np.int64) + 1))
        sm.pr_bias.copy_(torch.as_tensor(m.pr_bias.astype(np.float32)))
    return sm


def add_topic_register(m, codes: np.ndarray, phases, terms: int = 2,
                       seed: int = 0) -> dict:
    """Make an integer model without the topic register the start of one
    (spec.TOPIC_CHANNELS), in place.

    * Injection entries K..K+T-1 become the topic codes, addressed (head,
      topic channel t). They must have been unused (zero deltas).
    * In each phase of `phases`, the last `terms` scoring terms of every
      candidate read the head cell's OWN topic channels (a random channel per
      term, distinct within a candidate) with a random sign. The other terms,
      the routing and every immediate are untouched; phases not listed are
      exactly as before. Biases must be rebalanced afterwards
      (balance_by_simulation), and the immediates refitted.

    With `codes` all zero the register stays zero, so the listed phases lose
    `terms` of their terms and nothing else: the matched control."""
    K, T, TOP = spec.TAPE_CHANNELS, spec.TOPIC_CHANNELS, spec.TOPIC_AT
    if not T:
        raise ValueError("set MICA_TOPIC to add a topic register")
    codes = np.asarray(codes)
    if codes.shape != (N_SYMBOLS, T) or codes.dtype != np.int8 or (codes == -128).any():
        raise ValueError(f"topic codes must be int8 {(N_SYMBOLS, T)} without -128")
    if (m.inj_delta[:, K:K + T] != 0).any():
        raise ValueError("injection entries for the topic codes are in use")
    if not 1 <= terms <= min(T, N_SCORE_TERMS):
        raise ValueError(f"terms must be 1..{min(T, N_SCORE_TERMS)}")
    phases = sorted(set(int(p) for p in phases))
    if not phases or phases[0] < 0 or phases[-1] >= N_PHASE:
        raise ValueError(f"phases must be within 0..{N_PHASE - 1}")
    r = np.random.default_rng(seed)
    m.inj_cell[:, K:K + T] = 0
    m.inj_chan[:, K:K + T] = TOP + np.arange(T, dtype=np.uint8)
    m.inj_delta[:, K:K + T] = codes
    self_sel = list(OFFSETS).index(0)
    C, S = N_CANDIDATES, N_SCORE_TERMS
    for p in phases:
        pages = slice(p * PAGE_STRIDE, (p + 1) * PAGE_STRIDE)
        n = PAGE_STRIDE
        chans = np.argsort(r.random((n, C, T)), -1)[..., :terms]     # distinct
        m.sc_nb[pages, :, S - terms:] = self_sel
        m.sc_ch[pages, :, S - terms:] = (TOP + chans).astype(np.uint8)
        m.sc_co[pages, :, S - terms:] = np.where(
            r.random((n, C, terms)) < 0.5, -1, 1).astype(np.int8)
    m.validate()
    return {"phases": phases, "terms": terms,
            "content_symbols": int((codes != 0).any(1).sum())}


def fit_topic_bias(registers, base_scores, targets, *, steps: int = 400,
                   lr: float = 0.2, l2: float = 1e-4, divisor: int = None,
                   eligible=None, seed: int = 0, log=print) -> dict:
    """Fit an int8 word×topic bias on frozen scores.

    ``registers`` is int ``[N, T]``: the simulated topic register at each
    position (``engine.topic_registers``). It is not a tape code and not a
    rule immediate. ``base_scores`` is int ``[N, V]``, the frozen automaton
    (and, if wanted, memory) scores in engine logit units. This function
    does not read or write a ``.mica`` file and does not change rules.

    The forward pass rounds the weights to int8 with a straight-through
    gradient. The matrix written back is that int8 table, and it starts from
    zeros: held-out positions keep the zeros when the bias does not help.
    Do not pass Tiny Theory-of-Mind rows here.
    """
    if divisor is None:
        divisor = spec.LOGIT_DIVISOR
    registers = np.asarray(registers, np.int64)
    base_scores = np.asarray(base_scores, np.int64)
    targets = np.asarray(targets, np.int64)
    if registers.ndim != 2 or base_scores.ndim != 2:
        raise ValueError("registers must be [N, T] and base_scores [N, V]")
    if registers.shape[0] != base_scores.shape[0] or targets.shape != (registers.shape[0],):
        raise ValueError("registers, base_scores and targets must share N")
    n, width = registers.shape
    vocab = base_scores.shape[1]
    if n < 2:
        raise ValueError("need at least two positions")
    dev = torch.device("cpu")
    torch.manual_seed(seed)
    reg = torch.as_tensor(registers, dtype=torch.float64, device=dev)
    base = torch.as_tensor(base_scores, dtype=torch.float64, device=dev)
    tgt = torch.as_tensor(targets, dtype=torch.long, device=dev)
    if eligible is None:
        elig = torch.zeros(vocab, dtype=torch.float64, device=dev)
    else:
        mask = torch.as_tensor(np.asarray(eligible, bool), device=dev)
        if mask.shape != (vocab,):
            raise ValueError("eligible must be a bool mask over the vocabulary")
        elig = torch.where(mask, 0.0, float("-inf"))
    weight = torch.zeros(vocab, width, dtype=torch.float64, device=dev,
                         requires_grad=True)
    opt = torch.optim.Adam([weight], lr=lr)
    n_hold = max(1, min(n // 5, 200_000))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    hold, train = perm[:n_hold], perm[n_hold:]
    if len(train) == 0:
        train = perm

    def rounded():
        return weight + (weight.detach().round().clamp(-127, 127) - weight.detach())

    def logits(idx):
        bonus = reg[idx] @ rounded().T
        return (base[idx] + bonus) / float(divisor) + elig

    def bits(idx):
        with torch.no_grad():
            loss = Fn.cross_entropy(logits(idx), tgt[idx])
        return float(loss) / math.log(2)

    start = bits(hold)
    best = start
    keep = torch.zeros_like(weight)
    log(f"[topic] start held-out {start:.4f} bits/word on {n_hold} positions")
    for step in range(steps):
        rows = train[torch.randint(0, len(train), (min(256, len(train)),))]
        pred = logits(rows)
        loss = Fn.cross_entropy(pred, tgt[rows]) + l2 * weight.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0 or step == steps - 1:
            held = bits(hold)
            if held < best:
                best = held
                keep = weight.detach().round().clamp(-127, 127).clone()
            if (step + 1) % 200 == 0 or step == steps - 1:
                log(f"[topic] step {step + 1:4d}  held-out {held:.4f} "
                    f"(best {best:.4f})")
    table = keep.cpu().numpy().astype(np.int8)
    return {"W": table, "heldout_bits_start": start, "heldout_bits_best": best,
            "divisor": int(divisor), "channels": int(width)}


# bytes that continue a word: letters, digits, apostrophe, and the bytes of
# multi-byte UTF-8 characters
WORD_BYTES = frozenset(list(range(48, 58)) + list(range(65, 91)) +
                       list(range(97, 123)) + [39] + list(range(128, 256)))


def reserve_word_boundary_route(code: torch.Tensor) -> torch.Tensor:
    """Reserve one pairdiff routing bit for word versus separator bytes.

    ``word_state_rules`` can reset at separators only if no word byte shares
    their phase-0 page. Random byte codes almost always mix both classes in
    every page. Pinning one comparison bit gives disjoint page sets while the
    partner channel and all other tape channels remain trainable. The partner
    is kept within [-126, 126], so the comparison stays strict after integer
    rounding, including for BOS/EOS.
    """
    if not ROUTING_CHANNELS:
        raise ValueError("word route needs a routing channel")
    route = ROUTING_CHANNELS[0]
    partner = route + ROUTING_PAIR_OFFSET
    if partner >= code.shape[1]:
        raise ValueError("word route needs both pairdiff tape channels")
    word = torch.zeros(code.shape[0], dtype=torch.bool, device=code.device)
    word[list(WORD_BYTES)] = True
    out = code.clone()
    out[:, partner] = code[:, partner].clamp(-126, 126)
    out[:, route] = torch.where(word, 127.0, -127.0).to(code.dtype)
    return out


def reserve_letter_routes(code: torch.Tensor) -> torch.Tensor:
    """Give English letters distinct MICA page ids using the existing 5 bits.

    This changes only the learned byte-code constraints during fitting: the
    integer engine, routing rule, page count and inference file stay the same.
    Classes 1..26 name letters, 27 digits, 28 apostrophe, 29 UTF-8 bytes;
    separators, BOS and EOS use 0. Other tape channels remain trainable.
    """
    if len(ROUTING_CHANNELS) < 5 or code.shape[1] <= max(ROUTING_CHANNELS) + ROUTING_PAIR_OFFSET:
        raise ValueError("letter routing needs five existing pairdiff bits")
    route = torch.zeros(code.shape[0], dtype=torch.long, device=code.device)
    for index in range(26):
        for byte in (65 + index, 97 + index):
            if byte < len(route):
                route[byte] = index + 1
    route[48:58] = 27
    route[39] = 28
    route[128:256] = 29
    out = code.clone()
    for bit, channel in enumerate(ROUTING_CHANNELS[:5]):
        partner = channel + ROUTING_PAIR_OFFSET
        out[:, partner] = code[:, partner].clamp(-126, 126)
        out[:, channel] = torch.where(((route >> bit) & 1).bool(),
                                      127.0, -127.0).to(code.dtype)
    return out


def byte_patterns(model) -> torch.Tensor:
    """The routing pattern (page within a phase) each symbol's tape code
    selects, from the model's own codes. (N_SYMBOLS,)"""
    K = spec.TAPE_CHANNELS
    code = model.inj_delta.detach()[:, :K].round().clamp(-127, 127).cpu()
    rc = torch.tensor(ROUTING_CHANNELS)
    rw = 1 << torch.arange(len(ROUTING_CHANNELS))
    return ((code[:, rc] >= code[:, rc + ROUTING_PAIR_OFFSET]).long() * rw).sum(-1)


def word_state_rules(model, sel_init: float, margin: float = 3.0,
                     seed: int = 4, reset_on_nonword: bool = True) -> dict:
    """Give the rule book a recurrent byte state (needs VSET and a
    structured rule book; call after structured_rules).

    Phase 0 is the WORD STATE. On a page whose byte continues a word, the
    candidates are scored on the state the cell one byte older holds, and the
    winner SETs a fresh 6-value state: state(t) = hash(byte t, state(t-1)).
    With ``reset_on_nonword=True``, a page whose byte ends a word (space,
    punctuation) resets to that page's constant. With False, every byte,
    including spaces, updates the state, allowing it to cross word boundaries.
    The state immediates are large random values and are never fitted -- they
    are the hash, and the features below are keyed on them.

    Phase 1 is a FEATURE of the state: scored on the cell's own state, its
    winner's immediates are fitted like any other rule feature. So the head
    carries a fitted vector for (this byte, everything since the word began),
    where before it saw at most the last three bytes."""
    K = spec.TAPE_CHANNELS
    VW = spec.VSET_WIDTH
    assert VW and K, "word state needs VSET and the tape"
    P, C, T = N_PAGES, N_CANDIDATES, N_SCORE_TERMS
    g = torch.Generator().manual_seed(seed)
    back1 = [s for s, o in enumerate(OFFSETS) if o == -1][0]
    self_sel = [s for s, o in enumerate(OFFSETS) if o == 0][0]
    pat = byte_patterns(model)
    word_pat = torch.zeros(PAGE_STRIDE, dtype=torch.bool)
    if reset_on_nonword:
        for b in WORD_BYTES:
            word_pat[pat[b]] = True
    else:
        word_pat[:] = True
    phase = torch.arange(P) // PAGE_STRIDE
    patt = torch.arange(P) % PAGE_STRIDE
    state_ch = K + torch.arange(T) % VW                             # G0 channels
    with torch.no_grad():
        def peak(name, pages, idx):
            p = getattr(model, name)
            x = p.detach().cpu().clone()
            x[pages] = torch.randn(x[pages].shape, generator=g) * sel_init
            x[pages] = x[pages].scatter_add(-1, idx[..., None],
                                            torch.full(idx.shape + (1,), margin))
            p.copy_(x.to(p.device))
        # phase 0, word pages: read the older cell's state
        w0 = torch.nonzero((phase == 0) & word_pat[patt]).flatten()
        n = len(w0)
        peak("sc_nb", w0, torch.full((n, C, T), back1))
        peak("sc_ch", w0, state_ch.expand(n, C, T).clone())
        peak("sc_co", w0, torch.where(torch.rand(n, C, T, generator=g) < 0.5, 0, 2))
        # phase 0, other pages: constant state (only candidate 0 can win)
        r0 = torch.nonzero((phase == 0) & ~word_pat[patt]).flatten()
        peak("sc_co", r0, torch.ones(len(r0), C, T, dtype=torch.long))  # coef 0
        bias = model.sc_bias.detach().cpu().clone()
        bias[r0] = -30000.0
        bias[r0, 0] = 0.0
        bias[w0] = 0.0
        model.sc_bias.copy_(bias.to(model.sc_bias.device))
        # phase 0 immediates: the state vectors, large and random
        sv = torch.where(torch.rand(P, C, VW, generator=g) < 0.5, -1.0, 1.0) * \
            (48 + 79 * torch.rand(P, C, VW, generator=g))
        ph0 = phase == 0
        ob = model.op_b.detach().cpu().clone(); ob[ph0] = sv[ph0, :, 0]
        model.op_b.copy_(ob.to(model.op_b.device))
        if VW > 1:
            ov = model.op_v.detach().cpu().clone(); ov[ph0] = sv[ph0, :, 1:]
            model.op_v.copy_(ov.to(model.op_v.device))
        # phase 1: features scored on the cell's own state
        f1 = torch.nonzero(phase == 1).flatten()
        n1 = len(f1)
        peak("sc_nb", f1, torch.full((n1, C, T), self_sel))
        peak("sc_ch", f1, state_ch.expand(n1, C, T).clone())
        peak("sc_co", f1, torch.where(torch.rand(n1, C, T, generator=g) < 0.5, 0, 2))
    return {"word_pages": int(word_pat.sum()), "reset_pages": int((~word_pat).sum())}


def reversible_state_rules(model, sel_init: float, margin: float = 3.0,
                           seed: int = 5, multiplier: int = 4,
                           tracks: int = 1) -> dict:
    """Initialize a non-collapsing 243-state rule path in the existing VSET book.

    Five ternary work channels encode each state track. The six existing score
    terms make the hard winning candidate an exact decoder of the previous
    cell's code. Each state phase writes a page-conditioned permutation. The
    following feature phases select by the new states and their VSET
    immediates are learned by the normal integer-rule fitter. One or two
    tracks use the same cells, channels, opcodes, probes and engine paths.
    """
    K = spec.TAPE_CHANNELS
    if spec.VSET_WIDTH != 6 or N_SCORE_TERMS != 6 or PAGE_STRIDE < 2:
        raise ValueError("reversible state requires the existing VSET6 rule book")
    if tracks not in (1, 2) or N_PHASE < 2 * tracks:
        raise ValueError("reversible state requires one or two phase pairs")
    states = 3 ** 5
    if math.gcd(multiplier, states) != 1:
        raise ValueError("reversible multiplier must be coprime to 243")
    if N_CANDIDATES < states:
        raise ValueError("reversible state needs at least 243 candidates")
    g = torch.Generator().manual_seed(seed)
    pages = torch.arange(2 * tracks * PAGE_STRIDE)
    old = torch.arange(states)
    powers = 3 ** torch.arange(5)
    trits = ((old[:, None] // powers[None, :]) % 3 - 1).long()
    codes = torch.cat([trits * 127, torch.zeros(states, 1, dtype=torch.long)], 1)
    back = [s for s, offset in enumerate(OFFSETS) if offset == -1][0]
    own = [s for s, offset in enumerate(OFFSETS) if offset == 0][0]
    neighbour = torch.full((len(pages), N_CANDIDATES, N_SCORE_TERMS),
                           own, dtype=torch.long)
    channels = torch.empty_like(neighbour)
    for track in range(tracks):
        state_pages = slice(track * PAGE_STRIDE, (track + 1) * PAGE_STRIDE)
        feature_pages = slice((tracks + track) * PAGE_STRIDE,
                              (tracks + track + 1) * PAGE_STRIDE)
        neighbour[state_pages] = back
        channels[state_pages] = K + track * spec.VSET_WIDTH + \
            torch.arange(N_SCORE_TERMS)
        channels[feature_pages] = K + track * spec.VSET_WIDTH + \
            torch.arange(N_SCORE_TERMS)
    coefficients = torch.ones_like(neighbour)  # selector 1 is coefficient zero
    coefficients[:, :states, :5] = trits[None, :, :] + 1

    with torch.no_grad():
        def select(name: str, chosen: torch.Tensor) -> None:
            parameter = getattr(model, name)
            current = parameter[pages].detach().cpu()
            noise = torch.randn(current.shape, generator=g) * sel_init
            noise.scatter_add_(-1, chosen[..., None],
                               torch.full(chosen.shape + (1,), margin))
            parameter[pages] = noise.to(parameter.device)

        select("sc_nb", neighbour)
        select("sc_ch", channels)
        select("sc_co", coefficients)
        biases = model.sc_bias[pages].detach().cpu().clone()
        biases[:, :states] = -64 * (trits != 0).sum(-1)[None, :]
        biases[:, states:] = -30000
        model.sc_bias[pages] = biases.to(model.sc_bias.device)

        # A coprime multiplier makes each page's transition invertible mod 243.
        # The page offset depends on the current byte's 5-bit routing class.
        b = model.op_b[:tracks * PAGE_STRIDE].detach().cpu().clone()
        v = model.op_v[:tracks * PAGE_STRIDE].detach().cpu().clone()
        for track in range(tracks):
            a = multiplier if track == 0 else (5 if multiplier != 5 else 2)
            for page in range(PAGE_STRIDE):
                offset = page + 1 if track == 0 else 17 * page + 1
                next_state = (a * old + offset) % states
                dest = track * PAGE_STRIDE + page
                b[dest, :states] = codes[next_state, 0]
                v[dest, :states] = codes[next_state, 1:]
        model.op_b[:tracks * PAGE_STRIDE] = b.to(model.op_b.device)
        model.op_v[:tracks * PAGE_STRIDE] = v.to(model.op_v.device)
    return {"states": states, "state_channels": 5,
            "permutation_multiplier": multiplier,
            "tracks": tracks, "state_pages": tracks * PAGE_STRIDE}


def lexical_state_transition(old: int, page: int, is_word: bool,
                             track: int) -> int:
    """Two MICA integer-rule tracks: previous word and current word.

    Current-word states 0..120 encode a live word; 121..241 encode the same
    word across separators. The previous-word track always uses 0..120.
    A word byte after a separator starts a new hash, while the previous-word
    track keeps the finished word. This transition is written into VSET rule
    immediates; no external state or inference code is involved.
    """
    if not 0 <= old < 242 or not 0 <= page < PAGE_STRIDE or track not in (0, 1):
        raise ValueError("invalid lexical rule state")
    h = old % 121
    if track == 0:
        return h
    if not is_word:
        return 121 + h
    offset = page + 1
    return offset if old >= 121 else (5 * h + offset) % 121


def lexical_state_rules(model, sel_init: float, margin: float = 3.0,
                        seed: int = 7,
                        decode_state_features: bool = True) -> dict:
    """Install two word-aware tracks in the existing integer VSET rule book.

    Phase 0 copies the previous-word state from the preceding cell on word
    pages, or captures its current-word state on separator pages. Phase 1
    updates the current-word hash and keeps it through punctuation/spaces.
    By default, phases 2 and 3 decode the two states into learnable rule
    features. With decode_state_features=False only phases 0/1 are replaced;
    phases 2/3 remain ordinary local rules for a state-conditioned bucket
    pilot. Every selected rule remains an ordinary integer rule entry.
    """
    K, VW = spec.TAPE_CHANNELS, spec.VSET_WIDTH
    if VW != 6 or N_SCORE_TERMS != 6 or N_PHASE < 4 or N_CANDIDATES < 243:
        raise ValueError("lexical state needs VSET6, six score terms, four phases and 243 candidates")
    pat = byte_patterns(model)
    word_pat = torch.zeros(PAGE_STRIDE, dtype=torch.bool)
    word_pat[pat[list(WORD_BYTES)]] = True
    other = [b for b in range(N_SYMBOLS) if b not in WORD_BYTES]
    if bool(word_pat[pat[other]].any()):
        raise ValueError("word and separator pages overlap; reserve a routing bit")

    g = torch.Generator().manual_seed(seed)
    pages = torch.arange((4 if decode_state_features else 2) * PAGE_STRIDE)
    old = torch.arange(243)
    powers = 3 ** torch.arange(5)
    trits = ((old[:, None] // powers[None, :]) % 3 - 1).long()
    codes = torch.cat([trits * 127, torch.zeros(243, 1, dtype=torch.long)], 1)
    back = [s for s, offset in enumerate(OFFSETS) if offset == -1][0]
    own = [s for s, offset in enumerate(OFFSETS) if offset == 0][0]
    n, c, t = len(pages), N_CANDIDATES, N_SCORE_TERMS
    neighbour = torch.full((n, c, t), own, dtype=torch.long)
    neighbour[:2 * PAGE_STRIDE] = back
    channels = torch.empty_like(neighbour)
    channels[:PAGE_STRIDE] = (torch.where(word_pat, K, K + VW)[:, None, None]
                              + torch.arange(t)[None, None, :])
    channels[PAGE_STRIDE:2 * PAGE_STRIDE] = K + VW + torch.arange(t)
    if decode_state_features:
        channels[2 * PAGE_STRIDE:3 * PAGE_STRIDE] = K + torch.arange(t)
        channels[3 * PAGE_STRIDE:] = K + VW + torch.arange(t)
    coefficients = torch.ones_like(neighbour)
    coefficients[:, :242, :5] = trits[None, :242, :] + 1

    with torch.no_grad():
        def select(name: str, chosen: torch.Tensor) -> None:
            p = getattr(model, name)
            current = p[pages].detach().cpu()
            logits = torch.randn(current.shape, generator=g) * sel_init
            logits.scatter_add_(-1, chosen[..., None],
                                torch.full(chosen.shape + (1,), margin))
            p[pages] = logits.to(p.device)

        select("sc_nb", neighbour)
        select("sc_ch", channels)
        select("sc_co", coefficients)
        bias = model.sc_bias[pages].detach().cpu().clone()
        bias[:, :242] = -64 * (trits[:242] != 0).sum(-1)[None, :]
        bias[:, 242:] = -30000
        model.sc_bias[pages] = bias.to(model.sc_bias.device)

        b = model.op_b[:2 * PAGE_STRIDE].detach().cpu().clone()
        v = model.op_v[:2 * PAGE_STRIDE].detach().cpu().clone()
        for page in range(PAGE_STRIDE):
            word = bool(word_pat[page])
            for old_state in range(242):
                previous = lexical_state_transition(old_state, page, word, 0)
                current = lexical_state_transition(old_state, page, word, 1)
                b[page, old_state] = codes[previous, 0]
                v[page, old_state] = codes[previous, 1:]
                b[PAGE_STRIDE + page, old_state] = codes[current, 0]
                v[PAGE_STRIDE + page, old_state] = codes[current, 1:]
        model.op_b[:2 * PAGE_STRIDE] = b.to(model.op_b.device)
        model.op_v[:2 * PAGE_STRIDE] = v.to(model.op_v.device)
    return {"word_pages": int(word_pat.sum()),
            "separator_pages": int((~word_pat).sum()),
            "word_hash_states": 121,
            "encoded_states": 242,
            "feature_phases": 2 if decode_state_features else 0}


def condition_local_buckets_on_previous_word(model, margin: float = 3.0) -> dict:
    """Make four local integer-rule selectors depend on previous-word state.

    The lexical transition has already written five ternary state channels in
    the head cell by these phases. Two scoring terms read different pairs of
    those channels in each phase; the other four terms and all VSET programs
    retain their structured local-rule initialization. This is hard winner
    selection inside MICA, not an added decoder or external predictor.
    """
    if (N_PHASE != 16 or N_SCORE_TERMS != 6 or spec.VSET_WIDTH != 6 or
            N_CANDIDATES < 243):
        raise ValueError("previous-word bucket coupling needs 16 phases, six terms, VSET6 and at least 243 candidates")
    own = next((i for i, offset in enumerate(OFFSETS) if offset == 0), None)
    if own is None:
        raise ValueError("previous-word bucket coupling needs self-cell scoring")
    pairs = {7: (0, 1), 11: (2, 3), 14: (4, 0), 15: (1, 3)}
    with torch.no_grad():
        for phase, trits in pairs.items():
            pages = slice(phase * PAGE_STRIDE, (phase + 1) * PAGE_STRIDE)
            for term, trit in zip((4, 5), trits):
                nb = model.sc_nb[pages, :, term]
                ch = model.sc_ch[pages, :, term]
                nb.fill_(-margin)
                ch.fill_(-margin)
                nb[..., own] = margin
                ch[..., spec.TAPE_CHANNELS + trit] = margin
    return {"phases": tuple(pairs), "state_trit_pairs": pairs,
            "terms_per_phase": 2, "scoring_offset": 0}


def balance_by_simulation(model, records, dev, ticks: int, rounds: int = 2,
                          iters: int = 80, max_records: int = 1500,
                          row_stride: int = 1, log=print) -> dict:
    """balance_buckets for a rule book whose scoring reads rule-written state
    (word_state_rules): the score rows come from running the machine, and
    since moving a bias moves the states the next cell is scored on, collect
    and balance twice. Pages with shut-out candidates (bias -30000) keep
    their biases. row_stride samples score rows across all records to bound
    memory use; a stride coprime to the number of phases covers each phase.
    """
    if row_stride < 1:
        raise ValueError("row_stride must be positive")
    stride = max(1, len(records) // max_records)
    sample = [r for r in records[::stride] if len(r)]
    P, C = N_PAGES, N_CANDIDATES
    tie = torch.arange(C, device=dev, dtype=torch.int32)
    base = model.sc_bias.detach().round().to(dev).int()
    frozen = (base <= -20000).any(1)                                # reset pages
    out = {}
    for rnd_ in range(rounds):
        pages, rows = [], []
        tick_index = 0

        def on_tick(page, sc, win, ph):
            nonlocal tick_index
            keep = tick_index % row_stride == 0
            tick_index += 1
            if not keep:
                return
            pages.append(page)
            rows.append((sc - model.sc_bias.detach().round().to(dev).long()[page])
                        .int())
        _simulate(model, sample, dev, ticks, on_tick=on_tick)
        log(f"[fit] balancing from {len(rows)} sampled ticks "
            f"(row stride {row_stride})")
        pg = torch.cat(pages)
        S = torch.cat(rows)
        beta = torch.where(frozen[:, None], base,
                           torch.zeros_like(base))

        def winners():
            o = torch.empty(S.shape[0], dtype=torch.long, device=dev)
            for i in range(0, S.shape[0], 65536):
                sc = (S[i:i + 65536] + beta[pg[i:i + 65536]]).long()
                o[i:i + 65536] = (sc * (C + 1) - tie).argmax(-1)
            return o

        def eff(cnt):
            used = (cnt.sum(1) > 0) & ~frozen
            p = cnt[used].float() / cnt[used].sum(1, keepdim=True).float()
            return float((-(p * p.clamp(min=1e-12).log()).sum(1)).exp().mean())
        cnt = torch.bincount(pg * C + winners(), minlength=P * C).reshape(P, C)
        before = eff(cnt)
        for it in range(iters):
            cnt = torch.bincount(pg * C + winners(), minlength=P * C).reshape(P, C)
            target = cnt.sum(1, keepdim=True).float() / C
            step = max(1, 64 >> (it // 10))
            upd = (step * torch.sign(cnt.float() - target)).int()
            upd = upd * ((cnt.sum(1, keepdim=True) > 0) & ~frozen[:, None]).int()
            beta -= upd
        cnt = torch.bincount(pg * C + winners(), minlength=P * C).reshape(P, C)
        after = eff(cnt)
        with torch.no_grad():
            model.sc_bias.copy_(beta.float().to(model.sc_bias))
        out[rnd_] = (before, after)
        log(f"[fit] rule buckets balanced (by simulation, pass {rnd_ + 1}): "
            f"effective buckets per page {before:.1f} -> {after:.1f}")
    return out
