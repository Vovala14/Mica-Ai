"""Exact prefix scoring for a frozen EmberWord; numpy only.

Wrap an existing loaded EmberWord with CompletionEngine. The default is stateless;
call new_session() for bounded, ephemeral reuse while one user types. No model
weights, cellular transitions, vocabulary or tie rules change. alpha=0 retains
released scoring; alpha=7 combines existing prefix/next integer scores9:7.
probability_gain=12 calibrates conditional probabilities, preserving rankings.
"""
from __future__ import annotations

import re
import threading
import time
import numpy as np

TYPED = re.compile(r"(?:^|\s)([A-Za-z]+)$")
WS = b" \t\n\r\x0b\x0c"


def starts(data):
    return [i for i, b in enumerate(data)
            if b < 128 and chr(b).isalnum() and (i == 0 or data[i-1] in WS)]


class CompletionEngine:
    """Wrap a frozen model. Create a new wrapper after replacing its artifacts."""
    def __init__(self, model, *, alpha=0, probability_gain=16):
        if not isinstance(alpha, int) or not 0 <= alpha <= 16:
            raise ValueError('alpha must be an integer0..16')
        if not isinstance(probability_gain, int) or probability_gain <= 0:
            raise ValueError('probability_gain must be a positive integer')
        self.model = model
        self.alpha = alpha
        self.probability_gain = probability_gain

    def new_session(self, *, ttl_seconds=60, max_context_bytes=8192):
        return CompletionSession(self, ttl_seconds, max_context_bytes)

    def complete(self, context, top_k=5):
        return self._complete(context, top_k, None)

    def complete_with_probabilities(self, context, top_k=5):
        """Probabilities conditional on ALL legal prefix candidates, not bytes."""
        return self._complete(context, top_k, None, probabilities=True)

    def _complete(self, context, top_k, session, probabilities=False):
        if top_k < 1:
            raise ValueError('top_k must be positive')
        match = TYPED.search(context)
        if not match:
            if session is not None:
                session._clear()
            return []
        typed = match.group(1)
        model = self.model
        idx = model.candidates(typed)
        if not len(idx):
            # An invalid prefix must not keep stale context alive.
            if session is not None:
                session._clear()
            return []
        before = context[:match.start(1)].encode('utf8')
        task = 'next' if len(typed) == 1 else 'prefix2'
        # Case is deliberately preserved: cellular inputs are case sensitive.
        data = before if task == 'next' else before + typed[:2].encode('ascii')
        blend = task == 'prefix2' and self.alpha > 0
        key = (id(model), task, data, blend)
        now = time.monotonic()
        if session is not None and session._key == key and now < session._expires:
            features = session._x
        else:
            P = model.probes(data)
            x = model._input(task, data, P, starts(data), len(data), len(before))
            other = model._input('next', data, P, starts(data), len(before), len(before)) if blend else None
            features = (x, other)
            if session is not None:
                session._clear()
                if len(data) <= session.max_context_bytes:
                    x.setflags(write=False)
                    if other is not None:
                        other.setflags(write=False)
                    session._key, session._x = key, features
                    session._expires = now + session.ttl_seconds
        # Same int8 x int8 integer dot as full scoring, with only eligible rows.
        def restricted(head, x):
            return (np.rint(model.Wf[head][idx] @ x.astype(np.float32)).astype(np.int64)
                    + model.B[head][idx])
        score = restricted(task, features[0])
        denominator = 1024
        if blend:
            score = (16-self.alpha)*score + self.alpha*restricted('next', features[1])
            denominator *= 16
        order = np.argsort(score, kind='stable')[::-1]
        if not probabilities:
            return unique_words(model.words, idx[order], top_k)
        gain = self.probability_gain if task == 'prefix2' else 16
        z = score.astype(np.float64)*gain/(16*denominator)
        prob = np.exp(z-z.max())
        prob /= prob.sum()
        # Retain highest-ranked spelling; merge alias probabilities if a future
        # vocabulary contains case-insensitive duplicates. Current037 is unique.
        totals = {}
        for i, value in zip(idx, prob):
            key = model.words[int(i)].casefold()
            totals[key] = totals.get(key, 0.) + float(value)
        return [(word, totals[word.casefold()])
                for word in unique_words(model.words, idx[order], top_k)]


def unique_words(words, ranked_ids, top_k):
    """Stable score order and highest-ID ties; emit each spelling only once."""
    result, seen = [], set()
    for i in ranked_ids:
        word = words[int(i)]
        key = word.casefold()
        if key not in seen:
            seen.add(key)
            result.append(word)
            if len(result) == top_k:
                break
    return result


class CompletionSession:
    """Single-user cache, at most two vectors, no disk writes/cross-user reuse.

    TTL bounds reuse, not autonomous erasure: clear() removes references immediately;
    otherwise the next call clears expired data. Close/discard the session at logout.
    A session is bound to one CompletionEngine and therefore one frozen model.
    """
    def __init__(self, engine, ttl_seconds, max_context_bytes):
        if ttl_seconds <= 0 or max_context_bytes < 0:
            raise ValueError('positive TTL and nonnegative context limit required')
        self.engine = engine
        self.ttl_seconds = ttl_seconds
        self.max_context_bytes = max_context_bytes
        self._lock = threading.Lock()
        self._clear()

    def _clear(self):
        self._key, self._x, self._expires = None, None, 0

    def clear(self):
        with self._lock:
            self._clear()

    def complete(self, context, top_k=5):
        with self._lock:
            if time.monotonic() >= self._expires:
                self._clear()
            return self.engine._complete(context, top_k, self)
