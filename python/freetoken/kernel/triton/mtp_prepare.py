"""Single-request MTP metadata from one host parameter transfer."""

import triton
import triton.language as tl


@triton.jit
def prepare(P, TABLE, META, OUT, CU, INITIAL, N: tl.constexpr, W: tl.constexpr,
            TS0: tl.constexpr, TS1: tl.constexpr, PAGE: tl.constexpr, B: tl.constexpr):
    start, end = tl.load(P), tl.load(P + 1)
    slot, linear = tl.load(P + 2), tl.load(P + 3)
    i = tl.arange(0, B)
    tl.store(META + i, start + i, i < N)
    tl.store(META + N + i, 0, i < N)
    loc = tl.load(TABLE + slot * TS0 + (start + i) * TS1, i < N, 0)
    tl.store(OUT + i, loc, i < N)
    tl.store(META + 2 * N + i, i * N, i < 2)
    tl.store(CU + i, i * N, i < 2)
    tl.store(META + 2 * N + 2, end)
    tl.store(META + 2 * N + 3, slot)
    tl.store(META + 2 * N + 4, N - 1)
    tl.store(META + 2 * N + 5, linear)
    tl.store(INITIAL, 1)
    base = tl.load(TABLE + slot * TS0 + i * PAGE * TS1, i < W, 0).to(tl.int64)
    page = tl.where(base < 0, -((-base + PAGE - 1) // PAGE), base // PAGE)
    tl.store(META + 2 * N + 6 + i, page, i < W)
