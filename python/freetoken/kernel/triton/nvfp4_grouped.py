"""Small NVFP4 route groups sharing weight loads and scalar accumulations."""
import triton
import triton.language as tl
from freetoken.kernel.triton.nvfp4_fused_moe import _e2m1_arithmetic


@triton.jit
def prepare_route_groups(IDS, MEMBERS, COUNTS, R: tl.constexpr, B: tl.constexpr):
    r = tl.arange(0, B)
    q = tl.load(IDS + r, r < R, -1)
    same = (q[:, None] == q[None, :]) & (r[None, :] < R)
    rank = tl.sum((same & (r[None, :] < r[:, None])).to(tl.int32), 1)
    # Bound each group even when a caller repeats an expert within one token's routes.
    same &= rank[:, None] // 4 == rank[None, :] // 4
    owner = tl.min(tl.where(same, r[None, :], R), 1)
    count = tl.sum(same.to(tl.int32), 1)
    tl.store(COUNTS + r, tl.where(owner == r, count, 0), r < R)
    tl.store(MEMBERS + owner * 4 + rank % 4, r, r < R)


@triton.jit
def _project_group(A, P, S, G, C, W, MEM, owner, slot,
                   N: tl.constexpr, K: tl.constexpr, TOP: tl.constexpr,
                   ROUTE_INPUT: tl.constexpr, MUL: tl.constexpr,
                   BN: tl.constexpr, BKW: tl.constexpr, GM: tl.constexpr):
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BKW)
    # Separate accumulators retain the single-route layout and reduction order.
    if GM > 0:
        r0 = tl.load(MEM + owner * 4 + 0)
        row0 = r0 if ROUTE_INPUT else r0 // TOP
        acc0 = tl.zeros((BKW, BN), tl.float32)
    if GM > 1:
        r1 = tl.load(MEM + owner * 4 + 1)
        row1 = r1 if ROUTE_INPUT else r1 // TOP
        acc1 = tl.zeros((BKW, BN), tl.float32)
    if GM > 2:
        r2 = tl.load(MEM + owner * 4 + 2)
        row2 = r2 if ROUTE_INPUT else r2 // TOP
        acc2 = tl.zeros((BKW, BN), tl.float32)
    if GM > 3:
        r3 = tl.load(MEM + owner * 4 + 3)
        row3 = r3 if ROUTE_INPUT else r3 // TOP
        acc3 = tl.zeros((BKW, BN), tl.float32)
    for start in range(tl.cdiv(K // 8, BKW)):
        kw = start * BKW + k
        word = tl.load(P + slot * N * (K // 8) + n[None, :] * (K // 8) + kw[:, None],
                       (kw[:, None] < K // 8) & (n[None, :] < N), 0)
        scale = tl.load(S + slot * N * (K // 16) + n[None, :] * (K // 16) + kw[:, None] // 2,
                        (kw[:, None] < K // 8) & (n[None, :] < N), 0.0).to(tl.float32)
        if GM > 0:
            part0 = tl.zeros((BKW, BN), tl.float32)
        if GM > 1:
            part1 = tl.zeros((BKW, BN), tl.float32)
        if GM > 2:
            part2 = tl.zeros((BKW, BN), tl.float32)
        if GM > 3:
            part3 = tl.zeros((BKW, BN), tl.float32)
        for j in tl.static_range(8):
            b = _e2m1_arithmetic((word >> (4 * j)) & 15)
            if GM > 0:
                a0 = tl.load(A + row0 * K + kw * 8 + j, kw < K // 8, 0.0).to(tl.float32)
                part0 += a0[:, None] * b
            if GM > 1:
                a1 = tl.load(A + row1 * K + kw * 8 + j, kw < K // 8, 0.0).to(tl.float32)
                part1 += a1[:, None] * b
            if GM > 2:
                a2 = tl.load(A + row2 * K + kw * 8 + j, kw < K // 8, 0.0).to(tl.float32)
                part2 += a2[:, None] * b
            if GM > 3:
                a3 = tl.load(A + row3 * K + kw * 8 + j, kw < K // 8, 0.0).to(tl.float32)
                part3 += a3[:, None] * b
        if GM > 0:
            acc0 += part0 * scale
        if GM > 1:
            acc1 += part1 * scale
        if GM > 2:
            acc2 += part2 * scale
        if GM > 3:
            acc3 += part3 * scale
    glob = tl.load(G + slot * N + n, n < N, 0.0).to(tl.float32)
    if GM > 0:
        y0 = tl.sum(acc0, 0) * glob
        if MUL:
            y0 *= tl.load(W + r0)
        tl.store(C + r0 * N + n, y0, n < N)
    if GM > 1:
        y1 = tl.sum(acc1, 0) * glob
        if MUL:
            y1 *= tl.load(W + r1)
        tl.store(C + r1 * N + n, y1, n < N)
    if GM > 2:
        y2 = tl.sum(acc2, 0) * glob
        if MUL:
            y2 *= tl.load(W + r2)
        tl.store(C + r2 * N + n, y2, n < N)
    if GM > 3:
        y3 = tl.sum(acc3, 0) * glob
        if MUL:
            y3 *= tl.load(W + r3)
        tl.store(C + r3 * N + n, y3, n < N)


@triton.jit
def decode_grouped_routes(A, P, S, G, C, W, IDS, MEM, COUNTS,
                          N: tl.constexpr, K: tl.constexpr, TOP: tl.constexpr,
                          ROUTE_INPUT: tl.constexpr, MUL: tl.constexpr,
                          BN: tl.constexpr, BKW: tl.constexpr):
    owner = tl.program_id(0)
    count = tl.load(COUNTS + owner)
    if count == 0:
        return
    slot = tl.load(IDS + owner).to(tl.int64)
    if count == 1:
        _project_group(A, P, S, G, C, W, MEM, owner, slot, N, K, TOP, ROUTE_INPUT, MUL, BN, BKW, 1)
    elif count == 2:
        _project_group(A, P, S, G, C, W, MEM, owner, slot, N, K, TOP, ROUTE_INPUT, MUL, BN, BKW, 2)
    elif count == 3:
        _project_group(A, P, S, G, C, W, MEM, owner, slot, N, K, TOP, ROUTE_INPUT, MUL, BN, BKW, 3)
    elif count == 4:
        _project_group(A, P, S, G, C, W, MEM, owner, slot, N, K, TOP, ROUTE_INPUT, MUL, BN, BKW, 4)
