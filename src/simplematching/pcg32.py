"""PCG32 RNG, ported from the TypeScript implementation (rng.ts) for
seed-for-seed compatibility between the Python and TypeScript ports.
"""

import math

import numpy as np

MASK_64 = (1 << 64) - 1
MASK_32 = (1 << 32) - 1
MULTIPLIER = 6364136223846793005
DEFAULT_INC = 1442695040888963407  # must be odd
INC = ((DEFAULT_INC << 1) | 1) & MASK_64  # ensure odd


class Pcg32Rng:
    def __init__(self, seed: int):
        self.state = 0
        self.inc = INC
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

    def next_uint32_bulk(self, count: int) -> np.ndarray:
        """count個のuint32を一括生成する(_next_uint32()をcount回呼ぶのと等価)。

        PCG32の内部状態遷移は線形合同漸化式 state_{i+1} = A*state_i + C (mod 2^64)
        であるため、A^i および C*(1+A+...+A^(i-1)) の前処理列を
        numpyのcumprod/cumsumで一括計算すれば、Pythonループを使わずに
        state_0..state_{count-1} をすべて求められる。出力への変換
        (xorshift + 可変ローテーション)も要素ごとに独立なため一括で計算できる。
        """
        if count <= 0:
            self.state = self.state  # no-op; keep interface symmetric
            return np.empty(0, dtype=np.uint32)

        A = np.uint64(MULTIPLIER)
        C = np.uint64(self.inc)

        with np.errstate(over='ignore'):
            a_pow = np.cumprod(np.full(count, A, dtype=np.uint64))  # A^1..A^count
            pow0 = np.empty(count, dtype=np.uint64)
            pow0[0] = np.uint64(1)
            if count > 1:
                pow0[1:] = a_pow[:-1]  # A^0..A^(count-1)
            g = np.cumsum(pow0)  # sum_{k=0}^{i} A^k, mod 2**64 via uint64 wraparound
            c_arr = C * g

            state0 = np.uint64(self.state)
            oldstates = np.empty(count, dtype=np.uint64)
            oldstates[0] = state0
            if count > 1:
                oldstates[1:] = a_pow[:count - 1] * state0 + c_arr[:count - 1]

            self.state = int(a_pow[count - 1]) * self.state + int(c_arr[count - 1])
            self.state &= MASK_64

            xorshifted = (((oldstates >> np.uint64(18)) ^ oldstates) >> np.uint64(27)) & np.uint64(MASK_32)
            rot = (oldstates >> np.uint64(59)) & np.uint64(31)
            neg_rot = (np.uint64(32) - rot) & np.uint64(31)
            out = (xorshifted >> rot) | (xorshifted << neg_rot)
            out &= np.uint64(MASK_32)

        return out.astype(np.uint32)

    def next_floats_bulk(self, count: int) -> np.ndarray:
        """[0, 1) の一様乱数をcount個一括生成する(next_float()をcount回呼ぶのと等価)"""
        return self.next_uint32_bulk(count).astype(np.float64) / 0x100000000


def next_int(rng: Pcg32Rng, max_value: int) -> int:
    """[0, max) の整数乱数"""
    if max_value <= 0:
        raise ValueError("max must be positive")
    return math.floor(rng.next_float() * max_value)


def permutation(rng: Pcg32Rng, n: int) -> list[int]:
    """0..n-1 のランダム順列"""
    return permutations(rng, 1, n)[0]


def permutations(rng: Pcg32Rng, count: int, n: int) -> list[list[int]]:
    """0..n-1 のランダム順列をcount個生成する。

    permutation()をcount回呼ぶのと等価だが、必要な乱数をまとめて
    一括生成し、count本のFisher-Yatesシャッフルを(独立であることを
    利用して)列ごとに一括でswapすることで高速化している。
    """
    if n <= 1:
        return [list(range(n)) for _ in range(count)]
    if count == 0:
        return []

    draws_per_perm = n - 1
    floats = rng.next_floats_bulk(count * draws_per_perm).reshape(count, draws_per_perm)

    arr = np.tile(np.arange(n), (count, 1))
    rows = np.arange(count)
    for col in range(draws_per_perm):
        i = n - 1 - col
        j = np.floor(floats[:, col] * (i + 1)).astype(np.intp)
        tmp = arr[rows, i].copy()
        arr[rows, i] = arr[rows, j]
        arr[rows, j] = tmp

    return arr.tolist()
