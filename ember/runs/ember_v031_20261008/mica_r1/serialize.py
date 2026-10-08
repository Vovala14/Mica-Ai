"""MICA R1 model file: exact byte layout, writer, and validating loader.

Sections 10 and 11. The format is byte-aligned and little endian; structures
are decoded explicitly rather than cast, as the spec requires. The total is
86,820 bytes for the reference configuration and the writer asserts it.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

from . import spec
from .engine import Model
from .spec import (N_CELLS, N_CHANNELS, N_PAGES, N_CANDIDATES, N_SCORE_TERMS,
                   N_INJECT, N_PROBE, MAX_TICKS, N_SYMBOLS)


class FormatError(ValueError):
    pass


def _header(payload: bytes) -> bytes:
    h = bytearray(spec.HEADER_LEN)
    h[0:8] = spec.MAGIC
    struct.pack_into("<H", h, 8, 1)                      # version
    struct.pack_into("<H", h, 10, spec.HEADER_LEN)
    struct.pack_into("<I", h, 12, spec.TOTAL_BYTES)
    for k, v in enumerate((N_CELLS, N_CHANNELS, N_PAGES, N_CANDIDATES,
                           N_SCORE_TERMS, N_INJECT, N_PROBE, MAX_TICKS)):
        struct.pack_into("<H", h, 16 + 2 * k, v)
    struct.pack_into("<I", h, 32, spec.OFF_INJECT)
    struct.pack_into("<I", h, 36, spec.OFF_RULES)
    struct.pack_into("<I", h, 40, spec.OFF_PROBES)
    ext = _extension()
    struct.pack_into("<I", h, 44, 1 if ext else 0)       # flags
    h[48:80] = hashlib.sha256(payload).digest()
    if ext:
        if N_SYMBOLS != 258 and len(ext) > 46:
            raise FormatError("word alphabet needs header bytes 126..127")
        h[80:80 + len(ext)] = ext
    if N_SYMBOLS != 258:
        # This reserved field is zero in every legacy byte model. A zero
        # value means the original 258-symbol alphabet.
        struct.pack_into("<H", h, 126, N_SYMBOLS)
    # otherwise bytes 80..127 stay zero
    return bytes(h)


_ROUTING_CODES = ("r1", "positive", "pairdiff", "parity")


def _extension() -> bytes:
    """Flag bit 0: the file needs the tape extensions (spec.EXTENDED), and
    bytes 80.. say exactly which machine it was trained as, so a decoder can
    refuse to run it as anything else. R1 files keep flags 0 and a zero tail.

      80 u8 block version (1)    81 u8 rolling readout    82 u8 int8 probes
      84 u16 tape channels       86 u16 rule window       88 u16 probe window
      90 u16 logit divisor       92 u8 routing mode       93 u8 offset count
      94.. i16 neighbour offsets (selector 1 onward)
    """
    if not spec.EXTENDED:
        return b""
    from . import engine
    offs = spec.OFFSETS[1:]
    if len(offs) > 16:
        raise FormatError("at most 16 neighbour offsets fit the header")
    b = bytearray(94 + 2 * len(offs) - 80)
    struct.pack_into("<BBBB", b, 0, 1, int(spec.ROLLING_READOUT),
                     int(spec.WIDE_PROBE_COEF), spec.VSET_WIDTH)
    struct.pack_into("<4H", b, 4, spec.TAPE_CHANNELS, spec.WINDOW,
                     spec.PROBE_WINDOW, spec.LOGIT_DIVISOR)
    struct.pack_into("<BB", b, 12, _ROUTING_CODES.index(engine.ROUTING_MODE),
                     len(offs))
    for j, o in enumerate(offs):
        struct.pack_into("<h", b, 14 + 2 * j, o)
    if spec.N_PHASE != 4:
        # R1's four phases need no mention; anything else is recorded after
        # the offsets, so files with four phases are byte-identical to before
        if 80 + len(b) + 1 > 128:
            raise FormatError("no header room for the phase count")
        b += bytes([spec.N_PHASE])
    if spec.TOPIC_CHANNELS:
        # "T", channel count, decay shift. Only topic files carry it, so every
        # earlier file is byte-identical, and a decoder without the register
        # refuses a topic file (its tail is not zero) and vice versa.
        if 80 + len(b) + 3 > 126:
            raise FormatError("no header room for the topic register")
        b += bytes([ord("T"), spec.TOPIC_CHANNELS, spec.TOPIC_SHIFT])
    return bytes(b)


def dumps(m: Model) -> bytes:
    m.validate()

    inj = np.zeros((N_SYMBOLS, N_INJECT, 4), np.uint8)
    if spec.WIDE_CELLS:
        # cell u16 in bytes 0-1, channel in 2, delta in 3: the reserved byte
        # becomes the address high byte, so the entry is still four bytes
        inj[:, :, 0] = (m.inj_cell & 0xFF).astype(np.uint8)
        inj[:, :, 1] = (m.inj_cell >> 8).astype(np.uint8)
        inj[:, :, 2] = m.inj_chan
        inj[:, :, 3] = m.inj_delta.view(np.uint8)
    else:
        inj[:, :, 0] = m.inj_cell
        inj[:, :, 1] = m.inj_chan
        inj[:, :, 2] = m.inj_delta.view(np.uint8)
        # byte 3 reserved zero

    cand = np.zeros((N_PAGES, N_CANDIDATES, spec.CANDIDATE_BYTES), np.uint8)
    for t in range(N_SCORE_TERMS):
        cand[:, :, 3 * t + 0] = m.sc_nb[:, :, t]
        cand[:, :, 3 * t + 1] = m.sc_ch[:, :, t]
        cand[:, :, 3 * t + 2] = m.sc_co[:, :, t].view(np.uint8)
    bias = m.sc_bias.astype("<i2").view(np.uint8).reshape(N_PAGES, N_CANDIDATES, 2)
    B, O, V = spec.CAND_BIAS_AT, spec.CAND_OPS_AT, spec.CAND_VSET_AT
    cand[:, :, B:B + 2] = bias
    cand[:, :, O + 0] = m.op_code
    cand[:, :, O + 1] = m.op_d
    cand[:, :, O + 2] = m.op_n
    cand[:, :, O + 3] = m.op_c
    cand[:, :, O + 4] = m.op_a.view(np.uint8)
    cand[:, :, O + 5] = m.op_b.view(np.uint8)
    cand[:, :, O + 6] = m.op_u
    # the rest reserved zero -- except VSET's extra immediates
    # (bytes 27..31 at six scoring terms)
    if spec.VSET_WIDTH > 1:
        cand[:, :, V:V + spec.VSET_WIDTH - 1] = m.op_v.view(np.uint8)

    probe = np.zeros((N_SYMBOLS, spec.PROBE_RECORD_BYTES), np.uint8)
    for t in range(N_PROBE):
        if spec.WIDE_CELLS:
            probe[:, 4 * t + 0] = (m.pr_cell[:, t] & 0xFF).astype(np.uint8)
            probe[:, 4 * t + 1] = (m.pr_cell[:, t] >> 8).astype(np.uint8)
            probe[:, 4 * t + 2] = m.pr_chan[:, t]
            probe[:, 4 * t + 3] = m.pr_co[:, t].view(np.uint8)
        else:
            probe[:, 4 * t + 0] = m.pr_cell[:, t]
            probe[:, 4 * t + 1] = m.pr_chan[:, t]
            probe[:, 4 * t + 2] = m.pr_co[:, t].view(np.uint8)
    _bias_at = 4 * N_PROBE          # NOT a literal 32: that is only
                                    # correct at the R1 geometry's 8 reads
    probe[:, _bias_at:_bias_at + 2] = (
        m.pr_bias.astype("<i2").view(np.uint8).reshape(N_SYMBOLS, 2))

    payload = inj.tobytes() + cand.tobytes() + probe.tobytes()
    if len(payload) != spec.TOTAL_BYTES - spec.HEADER_LEN:
        raise FormatError(f"payload is {len(payload)} bytes, expected "
                          f"{spec.TOTAL_BYTES - spec.HEADER_LEN}")
    blob = _header(payload) + payload
    assert len(blob) == spec.TOTAL_BYTES
    return blob


def loads(blob: bytes, validate: bool = True) -> Model:
    """Reject everything section 11 says to reject, before allocating."""
    if len(blob) != spec.TOTAL_BYTES:
        raise FormatError(f"file is {len(blob)} bytes, expected {spec.TOTAL_BYTES}")
    if blob[0:8] != spec.MAGIC:
        raise FormatError("bad magic")
    version, hlen = struct.unpack_from("<HH", blob, 8)
    total, = struct.unpack_from("<I", blob, 12)
    if version != 1 or hlen != spec.HEADER_LEN or total != spec.TOTAL_BYTES:
        raise FormatError("bad version or lengths")
    dims = struct.unpack_from("<8H", blob, 16)
    if dims != (N_CELLS, N_CHANNELS, N_PAGES, N_CANDIDATES, N_SCORE_TERMS,
                N_INJECT, N_PROBE, MAX_TICKS):
        raise FormatError(f"dimensions {dims} do not match the reference config")
    starts = struct.unpack_from("<3I", blob, 32)
    if starts != (spec.OFF_INJECT, spec.OFF_RULES, spec.OFF_PROBES):
        raise FormatError("bad section offsets")
    flags, = struct.unpack_from("<I", blob, 44)
    if flags not in (0, 1):
        raise FormatError("unknown flags")
    if flags == 0:
        if any(blob[80:128]):
            raise FormatError("nonzero reserved header bytes")
        if spec.EXTENDED:
            raise FormatError("an R1 file, but this decoder runs the tape "
                              "extensions (MICA_TAPE/WINDOW/PROBE_*)")
    else:
        want = _extension()
        if not want:
            raise FormatError("the file needs the tape extensions; set the "
                              "MICA_* environment it was trained with")
        got = bytes(blob[80:80 + len(want)])
        if N_SYMBOLS != 258 and len(want) > 46:
            raise FormatError("word alphabet needs header bytes 126..127")
        tail_end = 126 if len(want) <= 46 else 128
        file_symbols, = struct.unpack_from("<H", blob, 126)
        if (got != want or any(blob[80 + len(want):tail_end]) or
                (len(want) <= 46 and file_symbols not in (0, 258)
                 and file_symbols != N_SYMBOLS) or
                (len(want) <= 46 and N_SYMBOLS != (file_symbols or 258))):
            raise FormatError("the file's machine (tape, window, divisor, "
                              "routing, offsets or symbols) differs from this decoder's")
    payload = blob[spec.HEADER_LEN:]
    if hashlib.sha256(payload).digest() != blob[48:80]:
        raise FormatError("payload digest mismatch")

    inj = np.frombuffer(blob, np.uint8, count=spec.INJECT_SECTION,
                        offset=spec.OFF_INJECT).reshape(N_SYMBOLS, N_INJECT, 4)
    if not spec.WIDE_CELLS and inj[:, :, 3].any():
        raise FormatError("nonzero reserved byte in an injection entry")
    cand = np.frombuffer(blob, np.uint8, count=spec.RULE_SECTION,
                         offset=spec.OFF_RULES).reshape(N_PAGES, N_CANDIDATES,
                                                        spec.CANDIDATE_BYTES)
    vw = max(0, spec.VSET_WIDTH - 1)
    B, O, V = spec.CAND_BIAS_AT, spec.CAND_OPS_AT, spec.CAND_VSET_AT
    if cand[:, :, V + vw:].any():
        raise FormatError("nonzero reserved bytes in a candidate record")
    probe = np.frombuffer(blob, np.uint8, count=spec.PROBE_SECTION,
                          offset=spec.OFF_PROBES).reshape(N_SYMBOLS,
                                                          spec.PROBE_RECORD_BYTES)
    if not spec.WIDE_CELLS and probe[:, [4 * t + 3 for t in range(N_PROBE)]].any():
        raise FormatError("nonzero reserved byte in a probe entry")

    i8 = lambda a: a.view(np.int8)
    if spec.WIDE_CELLS:
        inj_cell = (inj[:, :, 0].astype(np.uint16)
                    | (inj[:, :, 1].astype(np.uint16) << 8))
        inj_chan, inj_delta = inj[:, :, 2].copy(), i8(inj[:, :, 3].copy())
    else:
        inj_cell = inj[:, :, 0].copy()
        inj_chan, inj_delta = inj[:, :, 1].copy(), i8(inj[:, :, 2].copy())
    m = Model(
        inj_cell=inj_cell, inj_chan=inj_chan,
        inj_delta=inj_delta,
        sc_nb=np.stack([cand[:, :, 3 * t + 0] for t in range(N_SCORE_TERMS)], -1).copy(),
        sc_ch=np.stack([cand[:, :, 3 * t + 1] for t in range(N_SCORE_TERMS)], -1).copy(),
        sc_co=i8(np.stack([cand[:, :, 3 * t + 2] for t in range(N_SCORE_TERMS)], -1).copy()),
        sc_bias=cand[:, :, B:B + 2].copy().view("<i2").reshape(N_PAGES, N_CANDIDATES),
        op_code=cand[:, :, O + 0].copy(), op_d=cand[:, :, O + 1].copy(),
        op_n=cand[:, :, O + 2].copy(), op_c=cand[:, :, O + 3].copy(),
        op_a=i8(cand[:, :, O + 4].copy()), op_b=i8(cand[:, :, O + 5].copy()),
        op_u=cand[:, :, O + 6].copy(),
        pr_cell=(np.stack([probe[:, 4 * t + 0].astype(np.uint16)
                           | (probe[:, 4 * t + 1].astype(np.uint16) << 8)
                           for t in range(N_PROBE)], -1)
                 if spec.WIDE_CELLS else
                 np.stack([probe[:, 4 * t + 0] for t in range(N_PROBE)], -1).copy()),
        pr_chan=np.stack([probe[:, 4 * t + (2 if spec.WIDE_CELLS else 1)]
                          for t in range(N_PROBE)], -1).copy(),
        pr_co=i8(np.stack([probe[:, 4 * t + (3 if spec.WIDE_CELLS else 2)]
                           for t in range(N_PROBE)], -1).copy()),
        pr_bias=probe[:, 4 * N_PROBE:4 * N_PROBE + 2]
                .copy().view("<i2").reshape(N_SYMBOLS),
    )
    if vw:
        m.op_v = cand[:, :, V:V + vw].copy().view(np.int8)
        if (m.op_v == -128).any():
            raise FormatError("forbidden -128 immediate")
    if (m.inj_delta == -128).any():
        raise FormatError("forbidden -128 delta")
    if validate:
        m.validate()                              # selectors, coefficients, opcodes
    return m


def load_other_channels(path) -> Model:
    """Read a model written under this geometry except for the channel count
    (for example a model without the topic register, loaded to become the
    start of one). The payload layout does not depend on the channel count,
    so only the header's channel field and machine block are ignored; the
    payload digest is still checked. The result is validated by the caller's
    own rules after it has been adapted (fit.add_topic_register)."""
    blob = Path(path).read_bytes()
    if len(blob) != spec.TOTAL_BYTES or blob[0:8] != spec.MAGIC:
        raise FormatError("not a model file of this size")
    dims = list(struct.unpack_from("<8H", blob, 16))
    want = [N_CELLS, N_CHANNELS, N_PAGES, N_CANDIDATES, N_SCORE_TERMS,
            N_INJECT, N_PROBE, MAX_TICKS]
    if dims[:1] + dims[2:] != want[:1] + want[2:]:
        raise FormatError(f"dimensions {dims} differ from {want} beyond the channel count")
    if hashlib.sha256(blob[spec.HEADER_LEN:]).digest() != blob[48:80]:
        raise FormatError("payload digest mismatch")
    want_ext = _extension()
    base_ext = want_ext[:-3] if spec.TOPIC_CHANNELS else want_ext
    if bytes(blob[80:80 + len(base_ext)]) != base_ext or \
            any(blob[80 + len(base_ext):126]):
        raise FormatError("the file's machine (tape, window, divisor, routing, "
                          "offsets) differs beyond the channel count")
    fixed = _header(blob[spec.HEADER_LEN:])
    return loads(fixed + blob[spec.HEADER_LEN:], validate=False)


def save(m: Model, path) -> int:
    blob = dumps(m)
    Path(path).write_bytes(blob)
    return len(blob)


def _assert_candidate_layout():
    # The candidate record's offsets now follow the scoring-term count
    # (spec.CAND_BIAS_AT / CAND_OPS_AT / CAND_VSET_AT); at six terms they are
    # the original 18, 20 and 27.
    if N_SCORE_TERMS == 6:
        assert (spec.CAND_BIAS_AT, spec.CAND_OPS_AT, spec.CAND_VSET_AT,
                spec.CANDIDATE_BYTES) == (18, 20, 27, 32)


def load(path) -> Model:
    return loads(Path(path).read_bytes())


def payload_hash(m: Model) -> str:
    return hashlib.sha256(dumps(m)[spec.HEADER_LEN:]).hexdigest()
