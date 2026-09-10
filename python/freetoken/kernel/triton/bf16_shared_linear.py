"""Small-row BF16 projection with a fixed reduction and shared weight tiles."""
import torch
import triton
import triton.language as tl


@triton.jit
def _project(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             XS0: tl.constexpr, XS1: tl.constexpr, WS0: tl.constexpr,
             SPLITS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    m = tl.arange(0, 16)
    k = split * BK + tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for block in range(triton.cdiv(K, SPLITS * BK)):
        offset = k + block * SPLITS * BK
        x = tl.load(X + m[:, None] * XS0 + offset[None, :] * XS1,
                    (m[:, None] < M) & (offset[None, :] < K), 0)
        w = tl.load(W + n[None, :] * WS0 + offset[:, None],
                    (n[None, :] < N) & (offset[:, None] < K), 0)
        acc = tl.dot(x, w, acc)
    tl.store(P + (split * M + m[:, None]) * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _reduce(P, Y, M: tl.constexpr, N: tl.constexpr, SPLITS: tl.constexpr,
            BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, SPLITS)
    values = tl.load(P + s[:, None] * M * N + offset[None, :],
                     offset[None, :] < M * N, 0)
    tl.store(Y + offset, tl.sum(values, axis=0), offset < M * N)


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
        x, weight, partial, m, n, k, *x.stride(), weight.stride(0),
        splits, block_n, block_k, num_warps=4)
    _reduce[(triton.cdiv(m*n, 256),)](partial, output, m, n, splits, 256)
    return output
