"""Shared small-row projections and fused FP8/BF16 GDN prefill."""
import torch
import triton
import triton.language as tl


@triton.jit
def _project(X, W, P, T, S, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             XS0: tl.constexpr, XS1: tl.constexpr, WS0: tl.constexpr,
             SPLITS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             FP8: tl.constexpr = False, Q: tl.constexpr = 0,
             BM: tl.constexpr = 16, PREFILL: tl.constexpr = False):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    m = tl.program_id(2) * BM + tl.arange(0, BM)
    k = split * BK + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(triton.cdiv(K, SPLITS * BK)):
        offset = k + block * SPLITS * BK
        x = tl.load(X + m[:, None] * XS0 + offset[None, :] * XS1,
                    (m[:, None] < M) & (offset[None, :] < K), 0)
        if FP8:
            packed = tl.load(W + n[None, :] * WS0 + offset[:, None],
                             (n[None, :] < Q) & (offset[:, None] < K), 0.).to(x.dtype)
            tail = tl.load(T + (n[None, :] - Q) * K + offset[:, None],
                           (n[None, :] >= Q) & (n[None, :] < N) & (offset[:, None] < K), 0)
            w = tl.where(n[None, :] < Q, packed, tail)
        else:
            w = tl.load(W + n[None, :] * WS0 + offset[:, None],
                        (n[None, :] < N) & (offset[:, None] < K), 0)
        acc = tl.dot(x, w, acc)
    if PREFILL:
        scale = tl.load(S + n, n < Q, 1.)
        acc *= scale[None, :]
    tl.store(P + (split * M + m[:, None]) * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _reduce(P, Y, S, M: tl.constexpr, N: tl.constexpr, SPLITS: tl.constexpr,
            BLOCK: tl.constexpr, FP8: tl.constexpr = False, Q: tl.constexpr = 0):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, SPLITS)
    values = tl.load(P + s[:, None] * M * N + offset[None, :],
                     offset[None, :] < M * N, 0)
    acc = tl.sum(values, axis=0)
    if FP8:
        n = offset % N
        scale = tl.load(S + n, (offset < M * N) & (n < Q), 1.)
        acc *= scale
    tl.store(Y + offset, acc, offset < M * N)


def bf16_shared_linear(x, weight, *, splits=4, block_n=64, block_k=128):
    assert x.is_cuda and weight.device == x.device
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.ndim == weight.ndim == 2 and 1 <= x.shape[0] <= 16
    assert x.shape[1] == weight.shape[1] and weight.stride(1) == 1
    assert splits in (1, 2, 4, 8, 16)
    m, k = x.shape
    n = weight.shape[0]
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _project[(triton.cdiv(n, block_n), splits)](
        x, weight, partial, weight, output, m, n, k, *x.stride(), weight.stride(0),
        splits, block_n, block_k, num_warps=4)
    _reduce[(triton.cdiv(m*n, 256),)](partial, output, output, m, n, splits, 256)
    return output


def fp8_shared_linear(x, weight, scale, tail):
    assert x.is_cuda and x.dtype == tail.dtype == torch.bfloat16
    assert weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert weight.device == scale.device == tail.device == x.device
    assert x.ndim == weight.ndim == tail.ndim == 2 and x.shape[0] >= 1
    assert x.shape[1] == weight.shape[1] == tail.shape[1]
    assert weight.is_contiguous() and tail.is_contiguous() and scale.shape == (weight.shape[0],)
    m, k = x.shape
    q = weight.shape[0]
    n = q + tail.shape[0]
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m > 16:
        bm = 64 if m >= 64 else 32
        _project[(triton.cdiv(n, 64), 1, triton.cdiv(m, bm))](
            x, weight, output, tail, scale, m, n, k, *x.stride(), weight.stride(0),
            1, 64, 128, FP8=True, Q=q, BM=bm, PREFILL=True, num_warps=4)
        return output
    partial = torch.empty((4, m, n), device=x.device, dtype=torch.float32)
    _project[(triton.cdiv(n, 64), 4)](
        x, weight, partial, tail, scale, m, n, k, *x.stride(), weight.stride(0),
        4, 64, 128, FP8=True, Q=q, num_warps=4)
    _reduce[(triton.cdiv(m*n, 256),)](partial, output, scale, m, n, 4, 256, FP8=True, Q=q)
    return output
