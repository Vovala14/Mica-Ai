"""MICA R1 constants — the reference configuration, fixed by the spec.

Section 2, "Reference deployment": 192 cells, 24 channels, 64 rule pages,
32 candidate programs per page, six scoring terms per candidate, twelve
injection entries per input symbol, eight readout entries per output symbol,
and at most twelve ticks per symbol.

Nothing in this module is learnable. Everything here is either a dimension the
file header repeats or a routing rule R1 marks as fixed.
"""

from __future__ import annotations

import os

# ---- dimensions (header offsets 16..30, in this order) --------------------
# R1 section 2 fixes the reference configuration, and these are its values.
# They can be overridden by environment variable so that a scale study can run
# one geometry per process: spec configures itself at import, and every other
# module binds to the configured values. Within a process the geometry is
# constant, exactly as the engine assumes.
#
#   MICA_CELLS MICA_CHANNELS MICA_PAGES MICA_CANDIDATES
#   MICA_SCORE_TERMS MICA_INJECT MICA_PROBE MICA_TICKS MICA_OFFSETS
#
# A model file records its own dimensions in the header, so a file written
# under one geometry is rejected by a loader running another.
def _env(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None else int(v)


N_CELLS = _env("MICA_CELLS", 192)
N_CHANNELS = _env("MICA_CHANNELS", 24)
N_PAGES = _env("MICA_PAGES", 0)          # 0 = derive from the routing width
N_CANDIDATES = _env("MICA_CANDIDATES", 32)
N_SCORE_TERMS = _env("MICA_SCORE_TERMS", 6)
N_INJECT = _env("MICA_INJECT", 12)
N_PROBE = _env("MICA_PROBE", 8)
MAX_TICKS = _env("MICA_TICKS", 12)

IS_R1_REFERENCE = (N_CELLS, N_CHANNELS, N_PAGES, N_CANDIDATES, N_SCORE_TERMS,
                   N_INJECT, N_PROBE, MAX_TICKS) == (192, 24, 64, 32, 6, 12, 8, 12)

# ---- vocabulary -----------------------------------------------------------
N_SYMBOLS = _env("MICA_SYMBOLS", 258)
if not 258 <= N_SYMBOLS <= 65535:
    raise ValueError("MICA_SYMBOLS must fit a uint16 alphabet of at least 258 symbols")
BOS = N_SYMBOLS - 2
EOS = N_SYMBOLS - 1

# ---- topology (section 3) -------------------------------------------------
# "Neighbor selector 0 means self; selectors 1 through 6 mean offsets
#  +1, -1, +16, -16, +37, -53."
_off = os.environ.get("MICA_OFFSETS")
OFFSETS = ((0,) + tuple(int(x) for x in _off.split(","))) if _off else \
          (0, 1, -1, 16, -16, 37, -53)
N_SELECTORS = len(OFFSETS)

# ---- routing (section 5) --------------------------------------------------
# "bits = sum((1 if F[i][c] >= 0 else 0) << c for c in range(4))"
# "page_id = 16 * phase[i] + bits"
#
# R1 fixes four routing channels, which with four phases gives 64 pages. The
# rule book is 75% of the model file, so the page count is the main axis along
# which the file grows: each extra routing bit doubles it. The width is a
# geometry parameter like any other, and the header records it.
_rc = os.environ.get("MICA_ROUTING_CHANNELS")
ROUTING_CHANNELS = (tuple(int(x) for x in _rc.split(",")) if _rc
                    else (0, 1, 2, 3))
N_ROUTING_BITS = len(ROUTING_CHANNELS)
PAGE_STRIDE = 1 << N_ROUTING_BITS                  # 16 at the R1 geometry
# "pairdiff" routing compares channel c against c + this. At R1 the routing
# channels are 0..3 and the partners are 4..7, so the offset is the bit count.
ROUTING_PAIR_OFFSET = _env("MICA_ROUTING_PAIR_OFFSET", N_ROUTING_BITS)
# R1 fixes four phases. MICA_PHASES changes that (a geometry parameter, like
# the routing width): more phases give more rule pages per routing pattern,
# so with as many ticks per byte, more distinct rule features per cell.
N_PHASE = _env("MICA_PHASES", 4)
if not N_PAGES:
    N_PAGES = N_PHASE * PAGE_STRIDE                # 64 at the R1 geometry

# ---- opcodes (section 6 table) -------------------------------------------
HOLD, ADD, SET, SWAP, DECAY, TURN, PULSE, QUIET = range(8)
OPCODE_NAMES = ("HOLD", "ADD", "SET", "SWAP", "DECAY", "TURN", "PULSE", "QUIET")
N_OPCODES = 8
VSET = 8                    # tape extension (MICA_VSET): vector SET

# ---- arithmetic (section 3) ----------------------------------------------
SAT_MIN, SAT_MAX = -127, 127      # "-128 is forbidden in field values"
# Section 12 says "divide each probe score by 16". That constant is an
# architecture parameter, not a file field: the 86,820-byte model does not
# store it, exactly as it does not store the tick cap or the neighbour
# offsets. R1's value of 16 is badly mis-scaled against the field it reads.
# Measured on a random model over 256-byte records (docs/r1-findings.md 5.5):
# at steady state the across-symbol spread of probe scores is 36 raw units,
# so at divisor 16 the field alone injects 2.26 nats of logit spread, on a
# distribution whose entire useful range is about 4.8 nats. Random probe
# coefficients therefore start the model WORSE than uniform and the gap grows
# with record length, while zeroing them disconnects the field from the output
# entirely. A larger divisor puts one probe coefficient's contribution on the
# same scale as the byte statistics it has to improve on.
LOGIT_DIVISOR = int(os.environ.get("MICA_LOGIT_DIVISOR", "16"))

# Probe addressing. R1 section 8 reads ABSOLUTE cells, while section 4 writes
# each byte at base_cell + position, a write head that advances one cell per
# byte. So the most recent byte lands somewhere new every step, and a fixed
# probe sees only "which byte arrived at absolute position k - base_cell":
# no stable view of recent context. The first 4.25 MB gradient run sat at the
# byte-frequency level for 1,100 steps. With a rolling readout a probe reads
# probe_cell + position, the same frame as the writes, so "the byte written j
# steps ago" is always at the same relative offset. Off by default: R1 as
# written stays conformant. Like the logit divisor, it is not stored in the
# model file, and a decoder must be told.
ROLLING_READOUT = os.environ.get("MICA_ROLLING_READOUT", "0") == "1"

# ---- tape extensions (all off by default: R1 as written is unchanged) -----
# Measured on the 4.25 MB run after 4,800 steps: the field did not hold even
# the previous byte cleanly. A dense linear readout of 512 field values scored
# 3.99 bits/byte, worse than a one-hot of the last byte alone (3.72), while the
# probes reading a CLEAN tape of the last few bytes can reach 3.47 (int8
# coefficients) or 3.64 (ternary) on the same text. The rules were spending
# their capacity failing to preserve the input. These switches make the memory
# structural and leave the rules free to compute.
#
#   MICA_TAPE=K          channels 0..K-1 are the byte tape. Ingest clears the
#                        write-head cell (all channels, and its phase) and SETs
#                        the tape channels to the symbol's first K injection
#                        deltas. No rule and no other injection entry may write
#                        a tape channel, so the last N_CELLS bytes are always
#                        exactly recoverable.
#   MICA_WINDOW=W        rules update exactly the W cells behind the write head,
#                        MAX_TICKS times per symbol, instead of R1's change-
#                        triggered activation. Older cells are frozen history.
#                        Work per symbol is W * MAX_TICKS cell updates, known in
#                        advance, and training and inference run the same
#                        machine (no activation gap).
#   MICA_PROBE_WINDOW=R  probes may read only the R cells behind the head.
#   MICA_PROBE_COEF=int8 probe coefficients are int8 rather than ternary. The
#                        file already stores a whole byte per coefficient.
TAPE_CHANNELS = _env("MICA_TAPE", 0)
WINDOW = _env("MICA_WINDOW", 0)
PROBE_WINDOW = _env("MICA_PROBE_WINDOW", 0)
WIDE_PROBE_COEF = os.environ.get("MICA_PROBE_COEF", "ternary") == "int8"
#   MICA_VSET=L          a ninth opcode, VSET (8): the winning candidate SETs
#                        L consecutive channels d, d+1, ... to L immediates --
#                        its op_b and L-1 more stored in the candidate record's
#                        reserved bytes, so the file does not grow. Measured
#                        with fitted rule immediates: letting each rule bucket
#                        write a vector instead of one value is worth ~0.19
#                        bits/byte on documents the fit never saw. Channels
#                        past the last one are simply not written.
VSET_WIDTH = _env("MICA_VSET", 0)
EXTENDED = bool(TAPE_CHANNELS or WINDOW or PROBE_WINDOW or WIDE_PROBE_COEF
                or VSET_WIDTH or N_SYMBOLS != 258)
PROBE_CO_MIN, PROBE_CO_MAX = (-127, 127) if WIDE_PROBE_COEF else (-1, 1)

# ---- file layout (sections 10 and 11) ------------------------------------
# A cell address is one byte in R1's layout, which caps the field at 256
# cells. Above that the address widens to u16 and takes over the entry's
# reserved byte, so the entry stays four bytes and the file size is unchanged.
WIDE_CELLS = N_CELLS > 256

MAGIC = b"MICAR001"
HEADER_LEN = 128
INJECT_ENTRY_BYTES = 4             # base_cell, channel, delta, reserved
# 3 bytes per scoring triple, 2 for the bias, 7 operand bytes, then VSET's
# extra immediates, padded to a multiple of 4 so records stay aligned. At the
# R1 geometry (six terms) this is 32 and the offsets are 18, 20 and 27, as
# they always were; other term counts move the bias and operands after the
# last triple instead of overwriting it.
CAND_BIAS_AT = 3 * N_SCORE_TERMS
CAND_OPS_AT = CAND_BIAS_AT + 2
CAND_VSET_AT = CAND_OPS_AT + 7
_cand_raw = CAND_VSET_AT
CANDIDATE_BYTES = ((_cand_raw + 3) // 4) * 4 + (
    4 if ((_cand_raw + 3) // 4) * 4 == _cand_raw else 0)
if N_SCORE_TERMS == 6:
    CANDIDATE_BYTES = 32
else:
    CANDIDATE_BYTES = max(CANDIDATE_BYTES, (
        (_cand_raw + max(0, VSET_WIDTH - 1) + 3) // 4) * 4)
PROBE_ENTRY_BYTES = 4              # cell, channel, coefficient, reserved
PROBE_RECORD_BYTES = N_PROBE * PROBE_ENTRY_BYTES + 2      # + int16 bias

INJECT_SECTION = N_SYMBOLS * N_INJECT * INJECT_ENTRY_BYTES        # 12,384
RULE_SECTION = N_PAGES * N_CANDIDATES * CANDIDATE_BYTES           # 65,536
PROBE_SECTION = N_SYMBOLS * PROBE_RECORD_BYTES                    #  8,772
TOTAL_BYTES = HEADER_LEN + INJECT_SECTION + RULE_SECTION + PROBE_SECTION

OFF_INJECT = HEADER_LEN                     # 128
OFF_RULES = OFF_INJECT + INJECT_SECTION     # 12,512
OFF_PROBES = OFF_RULES + RULE_SECTION       # 78,048

# ---- work bound (section 10) ---------------------------------------------
MAX_UPDATES_PER_SYMBOL = N_CELLS * MAX_TICKS                       # 2,304
MAX_SCORE_READS = MAX_UPDATES_PER_SYMBOL * N_CANDIDATES * N_SCORE_TERMS  # 442,368


def sat(v: int) -> int:
    return SAT_MAX if v > SAT_MAX else (SAT_MIN if v < SAT_MIN else v)


def wrap(v: int) -> int:
    return ((v % N_CELLS) + N_CELLS) % N_CELLS


def sign(v: int) -> int:
    return -1 if v < 0 else (1 if v > 0 else 0)


def _check_layout():
    """At the R1 reference geometry, the file layout must match section 10.

    Under any other geometry these numbers are simply recomputed; the header
    carries the dimensions, so a model file is never ambiguous about which
    geometry produced it.
    """
    assert 0 <= TAPE_CHANNELS <= min(N_INJECT, N_CHANNELS - 1), TAPE_CHANNELS
    assert 0 <= VSET_WIDTH <= 1 + CANDIDATE_BYTES - CAND_VSET_AT, (
        f"MICA_VSET={VSET_WIDTH}: a candidate record has room for "
        f"{CANDIDATE_BYTES - CAND_VSET_AT} extra immediates")
    assert 0 <= WINDOW <= N_CELLS and 0 <= PROBE_WINDOW <= N_CELLS
    if TAPE_CHANNELS or WINDOW or PROBE_WINDOW:
        # all three are defined in the write head's frame
        assert ROLLING_READOUT, "the tape extensions need MICA_ROLLING_READOUT=1"
    if not IS_R1_REFERENCE:
        assert N_CHANNELS >= N_ROUTING_BITS + ROUTING_PAIR_OFFSET, (
            f"pairdiff routing compares channel c against c+"
            f"{ROUTING_PAIR_OFFSET}; {N_CHANNELS} channels is not enough for "
            f"{N_ROUTING_BITS} routing bits")
        assert N_PAGES >= N_PHASE * PAGE_STRIDE, (
            f"{N_ROUTING_BITS} routing bits and {N_PHASE} phases address "
            f"{N_PHASE * PAGE_STRIDE} pages, but only {N_PAGES} exist")
        assert N_CANDIDATES >= 1 and N_CELLS > max(abs(o) for o in OFFSETS)
        return
    assert INJECT_SECTION == 12_384, INJECT_SECTION
    assert RULE_SECTION == 65_536, RULE_SECTION
    assert PROBE_SECTION == 8_772, PROBE_SECTION
    assert TOTAL_BYTES == 86_820, TOTAL_BYTES
    assert (OFF_INJECT, OFF_RULES, OFF_PROBES) == (128, 12_512, 78_048)
    assert MAX_SCORE_READS == 442_368, MAX_SCORE_READS


def describe() -> dict:
    """Everything a result file needs to identify the geometry it used."""
    return {"cells": N_CELLS, "channels": N_CHANNELS, "pages": N_PAGES,
            "candidates": N_CANDIDATES, "score_terms": N_SCORE_TERMS,
            "inject_entries": N_INJECT, "probe_entries": N_PROBE,
            "max_ticks": MAX_TICKS, "offsets": OFFSETS[1:],
            "model_file_bytes": TOTAL_BYTES,
            "field_state_bytes": N_CELLS * N_CHANNELS,
            "is_r1_reference": IS_R1_REFERENCE,
            "rolling_readout": ROLLING_READOUT,
            "logit_divisor": LOGIT_DIVISOR,
            "tape_channels": TAPE_CHANNELS, "window": WINDOW,
            "probe_window": PROBE_WINDOW,
            "probe_coef": "int8" if WIDE_PROBE_COEF else "ternary",
            "vset_width": VSET_WIDTH, "phases": N_PHASE}


def probe_cell_allowed(k: int) -> bool:
    """With a probe window, probe cell k (relative to the head frame, where
    k = N_CELLS - lag) must read one of the last PROBE_WINDOW bytes."""
    return not PROBE_WINDOW or k >= N_CELLS - PROBE_WINDOW


_check_layout()
