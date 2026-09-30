"""The instruments section 14 names.

"If the search cannot lower target loss or reproduce local byte patterns,
 inspect routing usage, active-set size, score ties, saturation counts, and
 mutation acceptance."

Mutation acceptance comes from the search history; the other four are measured
here by re-running the scalar engine with counters attached.
"""

from __future__ import annotations

import numpy as np

from . import spec
from .engine import Model, Session, ingest, run_ticks, _page_ids
from .spec import N_CELLS, N_CHANNELS, N_PAGES, N_CANDIDATES, OFFSETS, BOS, EOS

OFF = np.array(OFFSETS, dtype=np.int32)


def instrument(model: Model, records: list[bytes]) -> dict:
    m = model
    pages_used = np.zeros(N_PAGES, np.int64)
    cand_used = np.zeros(N_CANDIDATES, np.int64)
    opcode_used = np.zeros(spec.N_OPCODES, np.int64)
    ties = 0
    updates = 0
    symbols = 0
    active_sizes = []
    sat_counts = []

    for rec in records:
        s = Session()
        stream = [BOS] + list(rec) + [EOS]
        for sym in stream[:-1]:
            # replicate ingest, but count what happens inside the ticks
            active = np.zeros(N_CELLS, np.uint8)
            for k in range(spec.N_INJECT):
                i = (int(m.inj_cell[sym, k]) + s.position) % N_CELLS
                ch = int(m.inj_chan[sym, k])
                s.F[i, ch] = spec.sat(int(s.F[i, ch]) + int(m.inj_delta[sym, k]))
                active[i] = 1

            for _t in range(spec.MAX_TICKS):
                cells = np.flatnonzero(active)
                if cells.size == 0:
                    break
                active_sizes.append(cells.size)
                updates += cells.size

                F = s.F
                pid = _page_ids(F, s.phase, cells)
                pages_used += np.bincount(pid, minlength=N_PAGES)

                nb = m.sc_nb[pid]; ch = m.sc_ch[pid]
                co = m.sc_co[pid].astype(np.int32)
                src = (cells[:, None, None] + OFF[nb]) % N_CELLS
                scores = m.sc_bias[pid].astype(np.int32) + (co * F[src, ch]).sum(2)
                win = scores.argmax(1)
                best = scores.max(1, keepdims=True)
                ties += int(((scores == best).sum(1) > 1).sum())
                cand_used += np.bincount(win, minlength=N_CANDIDATES)
                opcode_used += np.bincount(m.op_code[pid, win].astype(np.int64),
                                           minlength=spec.N_OPCODES)

                G = F.copy(); phase_next = s.phase.copy()
                d = m.op_d[pid, win].astype(np.int32)
                n_ = m.op_n[pid, win].astype(np.int32)
                c_ = m.op_c[pid, win].astype(np.int32)
                a = m.op_a[pid, win].astype(np.int32)
                b = m.op_b[pid, win].astype(np.int32)
                u = m.op_u[pid, win].astype(np.int32)
                opc = m.op_code[pid, win].astype(np.int32)
                v = F[(cells + OFF[n_]) % N_CELLS, c_]
                own_d = F[cells, d]; own_u = F[cells, u]
                cl = lambda x: np.clip(x, spec.SAT_MIN, spec.SAT_MAX)
                for code, fn in ((spec.ADD, lambda s_: cl(own_d[s_] + a[s_] * v[s_] + b[s_])),
                                 (spec.SET, lambda s_: cl(a[s_] * v[s_] + b[s_])),
                                 (spec.DECAY, lambda s_: own_d[s_] - np.sign(own_d[s_])),
                                 (spec.TURN, lambda s_: -own_d[s_])):
                    sel = opc == code
                    if sel.any():
                        G[cells[sel], d[sel]] = fn(sel)
                sel = (opc == spec.SWAP) & (d != u)
                if sel.any():
                    G[cells[sel], d[sel]] = own_u[sel]
                    G[cells[sel], u[sel]] = own_d[sel]
                phase_next[cells] = (s.phase[cells].astype(np.int32) + 1) % 4

                nxt = np.zeros(N_CELLS, np.uint8)
                changed = cells[(G[cells] != F[cells]).any(1)]
                if changed.size:
                    nxt[changed] = 1
                    for off in OFFSETS[1:]:
                        nxt[(changed + off) % N_CELLS] = 1
                sel = opc == spec.PULSE
                if sel.any():
                    nxt[(cells[sel] + OFF[n_[sel]]) % N_CELLS] = 1
                s.F, s.phase = G, phase_next
                active = nxt

            sat_counts.append(float((np.abs(s.F) == 127).mean()))
            s.position = (s.position + 1) % N_CELLS
            symbols += 1

    total_sel = max(int(cand_used.sum()), 1)
    return {
        "symbols": symbols,
        "updates_per_symbol": round(updates / max(symbols, 1), 1),
        "updates_per_symbol_max": spec.MAX_UPDATES_PER_SYMBOL,
        "active_set_mean": round(float(np.mean(active_sizes)), 1) if active_sizes else 0.0,
        "active_set_max": int(np.max(active_sizes)) if active_sizes else 0,
        "active_fraction_of_field": round(float(np.mean(active_sizes)) / N_CELLS, 3)
                                    if active_sizes else 0.0,
        "pages_reached": int((pages_used > 0).sum()),
        "pages_total": N_PAGES,
        "page_usage_top5": [int(x) for x in np.argsort(-pages_used)[:5]],
        "page_gini": round(float(_gini(pages_used)), 3),
        "candidates_reached": int((cand_used > 0).sum()),
        "candidates_total": N_CANDIDATES,
        "score_tie_fraction": round(ties / total_sel, 4),
        "opcode_mix": {spec.OPCODE_NAMES[i]: round(float(opcode_used[i] / total_sel), 3)
                       for i in range(spec.N_OPCODES)},
        "saturated_fraction_mean": round(float(np.mean(sat_counts)), 4),
        "saturated_fraction_max": round(float(np.max(sat_counts)), 4),
    }


def _gini(x: np.ndarray) -> float:
    """0 means every page used equally, 1 means one page takes everything."""
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
