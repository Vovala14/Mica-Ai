"""Flame-W 0.3.2: same integer cellular automaton, two integer memory banks."""

from __future__ import annotations

import numpy as np

from memory import Memory, MemoryMicaWord


class DualMemoryMicaWord(MemoryMicaWord):
    def __init__(self, base, short_memory: Memory, long_memory: Memory):
        super().__init__(base, short_memory)
        self.long_memory = long_memory
        self.info["dual_memory_after_64"] = True

    def scores(self, st) -> np.ndarray:
        bank = self.long_memory if len(st.hist) > 64 else self.memory
        return self.base.scores(st.s).astype(np.int64) + bank.bonus(st.hist)
