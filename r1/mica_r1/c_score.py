"""Optional CPU scorer. Build once per process; requires a C compiler.

Each stack owns compiled immutable scoring rules. Rebuild the stack after a
mutation, as the search already does. Model files and the Python baseline stay
unchanged. Temporary native code is held in a private directory, not the repo.
"""
from __future__ import annotations
import ctypes
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from . import spec

_LIBRARY = None
_BUILD_DIR = None

# ctypes releases the GIL around a CDLL call, so the row loop can be split
# across Python threads and run genuinely in parallel -- no OpenMP, no extra
# runtime DLL to ship. Rows are independent: each writes one entry of `out`.
_THREADS = int(os.environ.get("MICA_SCORE_THREADS", "0")) or (os.cpu_count() or 1)
_POOL = ThreadPoolExecutor(max_workers=max(1, _THREADS)) if _THREADS > 1 else None
_MIN_ROWS_PER_THREAD = 2048


def _prebuilt() -> Path | None:
    """A binary shipped next to the source, so no compiler is needed.

    Windows has no `cc` by default, which would otherwise make the compiled
    scorer unavailable exactly where it is most wanted.
    """
    import sys as _sys
    names = ["score.dll"] if _sys.platform == "win32" else ["score.so", "score.dylib"]
    for n in names:
        cand = Path(__file__).with_name(n)
        if cand.exists():
            return cand
    return None


def _kernel():
    global _LIBRARY, _BUILD_DIR
    if _LIBRARY is None:
        have = _prebuilt()
        if have is not None:
            try:
                _LIBRARY = ctypes.CDLL(str(have))
                _LIBRARY.mica_winners.argtypes = [ctypes.c_int64]*7 + [ctypes.c_void_p]*12
                _LIBRARY.mica_winners.restype = None
                return _LIBRARY.mica_winners
            except OSError:
                _LIBRARY = None          # fall through to building it
        _BUILD_DIR = tempfile.TemporaryDirectory(prefix="mica-score-")
        target = Path(_BUILD_DIR.name) / "score.so"
        command = shlex.split(os.environ.get("CC", "cc")) + [
            "-O3", "-std=c99", "-shared", "-fPIC",
            str(Path(__file__).with_name("score_kernel.c")), "-o", str(target)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("C scorer build failed; install a C compiler or use "
                               "--backend numpy. " + str(getattr(exc, "stderr", exc))) from exc
        _LIBRARY = ctypes.CDLL(str(target))
        _LIBRARY.mica_winners.argtypes = [ctypes.c_int64]*7 + [ctypes.c_void_p]*12
        _LIBRARY.mica_winners.restype = None
    return _LIBRARY.mica_winners


class CompiledScorer:
    def __init__(self, ms):
        self.fn = _kernel()
        self.bias = np.ascontiguousarray(ms.sc_bias, dtype=np.int32)
        self.nb = np.zeros(ms.sc_nb.shape, np.int32)
        self.ch = np.zeros_like(self.nb)
        self.co = np.zeros_like(self.nb)
        self.count = np.zeros(ms.sc_bias.shape, np.int32)
        # Merge duplicate reads exactly; merged coefficients need not be ternary.
        for idx in np.ndindex(ms.sc_bias.shape):
            combined = {}
            for n, c, a in zip(ms.sc_nb[idx], ms.sc_ch[idx], ms.sc_co[idx]):
                key = (int(n), int(c))
                combined[key] = combined.get(key, 0) + int(a)
            live = [(n,c,a) for (n,c),a in combined.items() if a]
            self.count[idx] = len(live)
            for t, (n,c,a) in enumerate(live):
                self.nb[idx+(t,)] = n
                self.ch[idx+(t,)] = c
                self.co[idx+(t,)] = a
        self.bases = np.ascontiguousarray(
            ((np.arange(spec.N_CELLS)[:,None] + np.array(spec.OFFSETS))
             % spec.N_CELLS) * spec.N_CHANNELS, dtype=np.int32)

    def winners(self, field, batch, cell, model, page):
        # Internal API: geometry and legal rule indices come from validated models.
        if field.dtype != np.int32 or not field.flags.c_contiguous:
            raise ValueError("compiled scorer requires contiguous int32 field")
        indices = [np.ascontiguousarray(a, dtype=np.int64)
                   for a in (batch, cell, model, page)]
        n = len(batch)
        out = np.empty(n, np.int64)
        rules = [self.bases, self.nb, self.ch, self.co, self.count, self.bias]

        def call(lo: int, hi: int) -> None:
            # slice the index arrays and the output; the field and the rules are
            # read-only and shared, so no copying is needed
            args = [field, indices[0][lo:hi], indices[1][lo:hi],
                    indices[2][lo:hi], indices[3][lo:hi], *rules, out[lo:hi]]
            self.fn(hi - lo, spec.N_CELLS, spec.N_CHANNELS, spec.N_PAGES,
                    spec.N_CANDIDATES, spec.N_SCORE_TERMS, spec.N_SELECTORS,
                    *(a.ctypes.data for a in args))

        if _POOL is None or n < 2 * _MIN_ROWS_PER_THREAD:
            call(0, n)
            return out
        parts = min(_THREADS, max(1, n // _MIN_ROWS_PER_THREAD))
        edges = [round(k * n / parts) for k in range(parts + 1)]
        list(_POOL.map(lambda e: call(*e), zip(edges[:-1], edges[1:])))
        return out
