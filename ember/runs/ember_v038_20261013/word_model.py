"""Native integer-rule Ember word suggestions from cellular probe states.

MICA Ember v0.3.8, Hugging Face layout: the automaton (model.mica), the word readouts and word rules
(word_heads.npz), the engine geometry (package.json) and the engine (mica_r1/) are all in this folder.
numpy only.

Two integer linear readouts score a training-only vocabulary: 'next' at a word boundary and 'prefix2'
after the first two letters of a word. Each reads the automaton's 240 probes at the current position and,
as a word memory, at the starts of the previous K words. Before the readout, T extra MICA rule ticks run
at the word: each picks a page by pairdiff routing, scores its candidates with six +-1 terms plus an
integer bias, and the winner (ties to the lowest index) writes six integer values that the readout also
reads. In 'words' mode a term reads the current probe state or a fixed 16-value code of one of the
previous three words (a hash of the word, like the byte tape's 16 channels per byte); in 'probes' mode it
reads the probe states at the previous three word starts. Cellular updates and readouts have integer
results; optional conditional probabilities use floating-point normalization.
Completion works for any number of typed letters: one letter uses the next-word readout restricted to
that letter; two or more use the configured integer head blend at the second letter, restricted to all
typed letters. The completion policy is separate from the frozen learned cellular weights.

    python word_model.py "Thank you for "                 # next word
    python word_model.py --complete "Thank you for be"    # finish the word being typed
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = b" \t\n\r\x0b\x0c"
WCODE = 16
TYPED = re.compile(r"(?:^|\s)([A-Za-z]+)$")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def word_code(data: bytes, start: int) -> np.ndarray:
    """Fixed int8 code of the word starting at `start`: a hash of its lower-cased ASCII letters/digits."""
    end = start
    while end < len(data) and data[end] < 128 and chr(data[end]).isalnum():
        end += 1
    raw = np.frombuffer(hashlib.blake2b(data[start:end].lower(), digest_size=WCODE).digest(), np.uint8)
    return np.maximum(raw.astype(np.int16) - 128, -127).astype(np.int8)


def word_starts(data: bytes) -> list[int]:
    """Byte positions where a word begins: an ASCII letter or digit at the start or after ASCII whitespace."""
    return [p for p in range(len(data))
            if data[p] < 128 and chr(data[p]).isalnum() and (p == 0 or data[p - 1] in WS)]


class EmberWord:
    def __init__(self):
        manifest = json.loads((HERE / "manifest.json").read_text())
        source = HERE / manifest["source_model"]
        pack = HERE / manifest["word_heads"]
        if _sha(source) != manifest["source_sha256"] or _sha(pack) != manifest["word_heads_sha256"]:
            raise RuntimeError("Ember word artifact hash mismatch")
        geometry = json.loads((HERE / "package.json").read_text())["geometry_env"]
        if "mica_r1.spec" in sys.modules:
            raise RuntimeError("load EmberWord before importing mica_r1.spec")
        for key in tuple(os.environ):
            if key.startswith("MICA_"):
                del os.environ[key]
        os.environ.update(geometry)
        sys.path.insert(0, str(HERE))
        from mica_r1 import engine, serialize, spec
        self.engine, self.spec = engine, spec
        self.model = serialize.load(source)
        with np.load(pack, allow_pickle=False) as p:
            self.words = p["words"].tolist()
            self.K = int(p["anchors"]) if "anchors" in p.files else 0
            self.W = {t: p[f"{t}_W"] for t in ("next", "prefix2")}
            self.B = {t: p[f"{t}_B"].astype(np.int64) for t in ("next", "prefix2")}
            self.T = int(p["ticks"]) if "ticks" in p.files else 0
            self.mode = str(p["rule_mode"]) if self.T else ""
            self.rules = {t: {k: p[f"{t}_{k}"].astype(np.int64) for k in ("flat", "co", "ra", "rb", "bias", "I")}
                          for t in ("next", "prefix2")} if self.T else {}
        dim = 240 * (self.K + 1) + 6 * self.T
        for t in self.W:
            assert self.W[t].dtype == np.int8 and self.W[t].shape == (len(self.words), dim)
        # Every partial sum of int8 weights times int8 probes stays below 2**24 in magnitude, so float32
        # BLAS products are exact integers: fast scoring with the same integer result.
        assert 128 * 128 * dim < 2 ** 24
        self.Wf = {t: self.W[t].astype(np.float32) for t in self.W}
        self.cells = self.model.pr_cell[0].astype(np.int32)
        self.chans = self.model.pr_chan[0].astype(np.int32)
        self.by1, self.by2 = {}, {}
        for i, w in enumerate(self.words):
            self.by1.setdefault(w[:1], []).append(i)
            self.by2.setdefault(w[:2], []).append(i)
        self.by1 = {k: np.asarray(v) for k, v in self.by1.items()}
        self.by2 = {k: np.asarray(v) for k, v in self.by2.items()}
        self.completion_alpha, self.completion_probability_gain = 0, 16
        policy_path = HERE / "completion_policy.json"
        if policy_path.exists():
            policy = json.loads(policy_path.read_text())
            if (policy['source']['model_sha256'] != manifest['source_sha256'] or
                    policy['source']['heads_sha256'] != manifest['word_heads_sha256']):
                raise RuntimeError('Completion policy is not calibrated for these artifacts')
            self.completion_alpha = policy['alpha']
            self.completion_probability_gain = policy['gain']

    def probes(self, data: bytes) -> np.ndarray:
        """[len(data)+1, 240] probe states: row t is the state after BOS and the first t bytes."""
        session = self.engine.new_session(self.model)
        out = np.empty((len(data) + 1, len(self.cells)), np.int32)
        for t, byte in enumerate(data):
            out[t] = session.F[(self.cells + session.position) % self.spec.N_CELLS, self.chans]
            self.engine.ingest(self.model, session, int(byte))
        out[len(data)] = session.F[(self.cells + session.position) % self.spec.N_CELLS, self.chans]
        return out

    def features(self, context: str) -> np.ndarray:
        """Probe state after the whole context (the v0.3.x scalar probe)."""
        return self.probes(context.encode("utf-8"))[-1]

    def _input(self, task, data, P, starts, pos, word_start):
        prev = [s for s in starts if s < word_start][::-1]
        anchors = [P[prev[k]] if k < len(prev) else np.zeros(P.shape[1], P.dtype) for k in range(max(self.K, 3))]
        x = np.concatenate([P[pos]] + anchors[:self.K]).astype(np.int64)
        if not self.T:
            return x
        if self.mode == "words":
            codes = [word_code(data, prev[k]) if k < len(prev) else np.zeros(WCODE, np.int8) for k in range(3)]
            S = np.concatenate([P[pos]] + codes).astype(np.int64)
        else:
            S = np.concatenate([P[pos]] + anchors[:3]).astype(np.int64)
        r, pages, cands = self.rules[task], self.rules[task]["bias"].shape[1], self.rules[task]["bias"].shape[2]
        work = []
        for t in range(self.T):
            page = int(sum(int(S[a] >= S[b]) << k for k, (a, b) in enumerate(zip(r["ra"][t], r["rb"][t]))))
            scores = r["bias"][t, page] + (r["co"][t, page] * S[r["flat"][t, page]]).sum(-1)
            win = int(np.argmax(scores))                      # first maximum: ties go to the lowest index
            work.append(r["I"][(t * pages + page) * cands + win])
        return np.concatenate([x] + work)

    def _scores(self, task, x):
        dot = self.Wf[task] @ x.astype(np.float32)
        return np.rint(dot).astype(np.int64) + self.B[task]

    @staticmethod
    def _rank(scores, top_k, idx=None):
        if idx is None:
            return np.argsort(scores, kind="stable")[::-1][:top_k]
        return idx[np.argsort(scores[idx], kind="stable")[::-1][:top_k]]

    def candidates(self, prefix: str):
        prefix = prefix.lower()
        if len(prefix) == 1:
            return self.by1.get(prefix, np.empty(0, int))
        base = self.by2.get(prefix[:2], np.empty(0, int))
        if len(prefix) > 2 and len(base):
            base = base[[self.words[i].startswith(prefix) for i in base]]
        return base

    def next_words(self, context: str, top_k: int = 5) -> list[str]:
        if top_k < 1:
            raise ValueError('top_k must be positive')
        data = context.encode("utf-8")
        P = self.probes(data)
        x = self._input("next", data, P, word_starts(data), len(data), len(data))
        from fast_completion import unique_words
        return unique_words(self.words, self._rank(self._scores("next", x), len(self.words)), top_k)

    def complete(self, context: str, top_k: int = 5) -> list[str]:
        """Configured native integer scores, restricted to matching words.

        Stateless by default. For repeated keystrokes in one user's session,
        use new_completion_session(); the model itself stores no user context.
        """
        from fast_completion import CompletionEngine
        return CompletionEngine(self, alpha=self.completion_alpha,
                                probability_gain=self.completion_probability_gain).complete(context, top_k)

    def complete_with_probabilities(self, context: str, top_k: int = 5):
        """Conditional prefix probabilities; no byte or unconditional LM claim."""
        from fast_completion import CompletionEngine
        return CompletionEngine(self, alpha=self.completion_alpha,
                                probability_gain=self.completion_probability_gain).complete_with_probabilities(context, top_k)

    def new_completion_session(self, *, ttl_seconds=60, max_context_bytes=8192):
        """Bounded ephemeral reuse; clear/discard at logout or model replacement."""
        from fast_completion import CompletionEngine
        return CompletionEngine(self, alpha=self.completion_alpha,
                                probability_gain=self.completion_probability_gain).new_session(
            ttl_seconds=ttl_seconds, max_context_bytes=max_context_bytes)

    def suggest(self, context: str, *, prefix2: bool = False, top_k: int = 5) -> list[str]:
        """v0.3.x interface: next word, or (prefix2=True) finish a word from its typed letters."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if prefix2:
            return self.complete(context, top_k)
        return self.next_words(context, top_k)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MICA Ember v0.3.8 word suggestions")
    ap.add_argument("text", nargs="?", help="text so far; end with a space for the next word")
    ap.add_argument("--complete", action="store_true", help="finish the word whose first letters end the text")
    ap.add_argument("--top", type=int, default=5, help="number of suggestions (default 5)")
    a = ap.parse_args()
    m = EmberWord()
    if a.text is None:
        print(m.suggest("The weather today is "))
        print(m.suggest("The weather today is su", prefix2=True))
    else:
        print(m.suggest(a.text, prefix2=a.complete, top_k=a.top))
