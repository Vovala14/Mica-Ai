from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from memory import Memory

HERE = Path(__file__).resolve().parent

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8-sig'))
    path = HERE / manifest['memory']
    assert sha(path) == manifest['memory_sha256']
    mem = Memory(path)
    history_oldest_first = np.frombuffer(b'The model is', dtype=np.uint8).astype(np.int64)
    scalar = mem.bonus(history_oldest_first)
    batched = mem.bonus_batch(history_oldest_first[::-1][None, :])[0]
    assert scalar.shape == (258,)
    assert scalar.dtype.kind in 'iu'
    np.testing.assert_array_equal(scalar, batched)
    assert np.all(mem.bonus(np.empty(0, dtype=np.int64)) == 0)
    print('Ember v0.3.1 integer memory hash and scalar/batch conformance: OK')

if __name__ == '__main__':
    main()
