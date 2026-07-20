"""PCG32 RNG, ported from the TypeScript implementation (rng.ts) for
seed-for-seed compatibility between the Python and TypeScript ports.
"""

import math

MASK_64 = (1 << 64) - 1
MASK_32 = (1 << 32) - 1
MULTIPLIER = 6364136223846793005
DEFAULT_INC = 1442695040888963407  # must be odd


class Pcg32Rng:
    def __init__(self, seed: int):
        self.state = 0
        self.inc = ((DEFAULT_INC << 1) | 1) & MASK_64  # ensure odd
        self._step()
        self.state = (self.state + int(seed)) & MASK_64
        self._step()

    def _step(self) -> None:
        self.state = (self.state * MULTIPLIER + self.inc) & MASK_64

    def _next_uint32(self) -> int:
        oldstate = self.state
        self.state = (oldstate * MULTIPLIER + self.inc) & MASK_64

        xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & MASK_32
        rot = (oldstate >> 59) & 31
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & MASK_32

    def next_float(self) -> float:
        """[0, 1) の一様乱数"""
        return self._next_uint32() / 0x100000000


def next_int(rng: Pcg32Rng, max_value: int) -> int:
    """[0, max) の整数乱数"""
    if max_value <= 0:
        raise ValueError("max must be positive")
    return math.floor(rng.next_float() * max_value)


def permutation(rng: Pcg32Rng, n: int) -> list[int]:
    """0..n-1 のランダム順列"""
    arr = list(range(n))
    for i in range(n - 1, 0, -1):
        j = next_int(rng, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr
