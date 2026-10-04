"""Native integer-rule Ember word suggestions from frozen cellular probe states."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
R1 = REPO / "r1"

def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

class EmberWord:
    def __init__(self):
        manifest = json.loads((HERE / "manifest.json").read_text())
        source = REPO / manifest["source_model"]
        pack = HERE / manifest["word_heads"]
        if _sha(source) != manifest["source_sha256"] or _sha(pack) != manifest["word_heads_sha256"]:
            raise RuntimeError("Ember word artifact hash mismatch")
        info = json.loads((source.parent / "run_info.json").read_text())
        if "mica_r1.spec" in sys.modules:
            raise RuntimeError("load EmberWord before importing mica_r1.spec")
        for key in tuple(os.environ):
            if key.startswith("MICA_"):
                del os.environ[key]
        os.environ.update(info["env"])
        sys.path.insert(0, str(R1))
        from mica_r1 import engine, serialize, spec
        self.engine, self.spec = engine, spec
        self.model = serialize.load(source)
        with np.load(pack, allow_pickle=False) as p:
            self.words = p["words"].tolist()
            self.W = {t: p[f"{t}_W"].astype(np.int32) for t in ("next", "prefix2")}
            self.B = {t: p[f"{t}_B"].astype(np.int32) for t in ("next", "prefix2")}
        assert len(self.words) == self.W["next"].shape[0]
        self.cells = self.model.pr_cell[0].astype(np.int32)
        self.chans = self.model.pr_chan[0].astype(np.int32)

    def features(self, context: str) -> np.ndarray:
        session = self.engine.new_session(self.model)
        for byte in context.encode("utf-8"):
            self.engine.ingest(self.model, session, int(byte))
        return session.F[(self.cells + session.position) % self.spec.N_CELLS, self.chans].astype(np.int32)

    def suggest(self, context: str, *, prefix2: bool = False, top_k: int = 5) -> list[str]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        task = "prefix2" if prefix2 else "next"
        # Keep ranking in int64: negating int32 minimum would overflow and
        # make masked completion words rank above valid prefix matches.
        scores = self.W[task].astype(np.int64) @ self.features(context).astype(np.int64) + self.B[task].astype(np.int64)
        if prefix2:
            prefix = context[-2:].lower()
            if len(prefix) != 2 or not prefix.isascii() or not prefix.isalpha():
                return []
            mask = np.asarray([w.startswith(prefix) for w in self.words], bool)
            if not mask.any():
                return []
            valid = np.flatnonzero(mask)
            top = valid[np.argsort(scores[valid], kind="stable")[::-1][:top_k]]
            return [self.words[int(i)] for i in top]
        top = np.argsort(scores, kind="stable")[::-1][:top_k]
        return [self.words[int(i)] for i in top]

if __name__ == "__main__":
    m = EmberWord()
    print(m.suggest("The weather today is "))
    print(m.suggest("The weather today is su", prefix2=True))
