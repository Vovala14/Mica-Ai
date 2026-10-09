"""Conservative integer short-memory blend on the original cellular automaton."""
import numpy as np
from memory import MemoryMicaWord


class QuarterBlendMicaWord(MemoryMicaWord):
    def __init__(self, base, released, fitted, long):
        super().__init__(base, released)
        self.fitted = fitted
        self.long = long
        self.info["short_integer_blend_q4"] = True

    def scores(self, st):
        if len(st.hist) > 64:
            bonus = self.long.bonus(st.hist)
        else:
            a = self.memory.bonus(st.hist)
            b = self.fitted.bonus(st.hist)
            bonus = np.floor_divide(3*a+b+2, 4)
        return self.base.scores(st.s).astype(np.int64) + bonus
