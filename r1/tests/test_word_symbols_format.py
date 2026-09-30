"""Word alphabet conformance in a separate process; byte defaults stay fixed."""

import os
import subprocess
import sys
from pathlib import Path


def test_word_format_records_and_shared_integer_readout(tmp_path):
    code = r'''
import struct
import sys
from pathlib import Path
import numpy as np
from data.make_records import load_records
from mica_r1 import engine, serialize, spec
from mica_r1.discretise import to_integer
from mica_r1.soft import SoftMica

assert (spec.N_SYMBOLS, spec.BOS, spec.EOS) == (16384, 16382, 16383)
path = Path(sys.argv[1])
path.write_text("000001010140\n")  # <sep>, OOV 256, word 16385? invalid
try:
    load_records(path)
except ValueError:
    pass
else:
    raise AssertionError("invalid symbol was accepted")
path.write_text("00000101013f\n")  # 0, 257, 16129 (little-endian)
records = load_records(path)
assert len(records) == 1 and list(records[0]) == [0, 257, 16129]
soft = SoftMica(compact_selectors=True)
assert soft.pr_cell.shape[0] == 1 and soft.inj_delta.shape[0] == spec.N_SYMBOLS
discrete = to_integer(soft)
assert discrete.pr_cell.shape[0] == spec.N_SYMBOLS
assert discrete.pr_co.shape[0] == spec.N_SYMBOLS

m = engine.random_model(4)
m.pr_cell[:] = m.pr_cell[0]
m.pr_chan[:] = m.pr_chan[0]
blob = serialize.dumps(m)
assert struct.unpack_from("<H", blob, 126)[0] == spec.N_SYMBOLS
loaded = serialize.loads(blob)
assert serialize.dumps(loaded) == blob
s = engine.new_session(loaded)
fast = engine.probe_scores(loaded, s)
assert fast.shape == (spec.N_SYMBOLS,)
cells = loaded.pr_cell.astype(np.int32)
cells = (cells + s.position) % spec.N_CELLS if spec.ROLLING_READOUT else cells
vals = s.F[cells, loaded.pr_chan.astype(np.int32)]
slow = loaded.pr_bias.astype(np.int32) + (loaded.pr_co.astype(np.int32) * vals).sum(1)
assert np.array_equal(fast, slow)
assert isinstance(loaded.copy(), engine.Model)
tampered = bytearray(blob)
struct.pack_into("<H", tampered, 126, 258)
try:
    serialize.loads(bytes(tampered))
except serialize.FormatError:
    pass
else:
    raise AssertionError("symbol mismatch was accepted")
'''
    env = os.environ.copy()
    env["MICA_SYMBOLS"] = "16384"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "word.jsonl")],
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
