"""Device-side speculative acceptance and first-rejection normalization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _accept_prefix(target, draft, tokens, uniforms, accepted,
                   COUNT: tl.constexpr, TARGET_STRIDE: tl.constexpr,
                   DRAFT_STRIDE: tl.constexpr, TARGET_COL: tl.constexpr,
                   DRAFT_COL: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.arange(0, BLOCK)
    mask = rows < COUNT
    token = tl.load(tokens + rows, mask, other=0).to(tl.int64)
    p = tl.load(target + rows * TARGET_STRIDE + token * TARGET_COL, mask, other=1).to(tl.float32)
    q = tl.load(draft + rows * DRAFT_STRIDE + token * DRAFT_COL, mask, other=0).to(tl.float32)
    u = tl.load(uniforms + rows, mask, other=0)
    first = tl.min(tl.where(mask & ~(u * q < p), rows, COUNT), 0)
    tl.store(accepted, first)


def acceptance_prefix_cuda(target, draft, tokens, *, uniforms=None):
    count = tokens.numel()
    if uniforms is None:
        uniforms = torch.rand(count, device=target.device, dtype=torch.float32)
    elif not isinstance(uniforms, torch.Tensor):
        uniforms = torch.full((count,), uniforms, device=target.device, dtype=torch.float32)
    elif uniforms.numel() == 1:
        uniforms = uniforms.expand(count)
    output = torch.empty((), device=target.device, dtype=torch.int64)
    _accept_prefix[(1,)](target, draft, tokens.contiguous(), uniforms.contiguous(), output,
                        count, target.stride(0), draft.stride(0), target.stride(1), draft.stride(1),
                        triton.next_power_of_2(max(count, 1)))
    return output


@triton.jit
def _residual_partial(target, draft, accepted, residual, sums,
                      COUNT: tl.constexpr, VOCAB: tl.constexpr,
                      TARGET_STRIDE: tl.constexpr, DRAFT_STRIDE: tl.constexpr,
                      TARGET_COL: tl.constexpr, DRAFT_COL: tl.constexpr,
                      BLOCK: tl.constexpr):
    block = tl.program_id(0)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < VOCAB
    index = tl.load(accepted)
    p = tl.load(target + index * TARGET_STRIDE + cols * TARGET_COL, mask, other=0).to(tl.float32)
    q = tl.load(draft + index * DRAFT_STRIDE + cols * DRAFT_COL,
                mask & (index < COUNT), other=0).to(tl.float32)
    value = tl.maximum(p - q, 0)
    tl.store(residual + cols, value, mask)
    tl.store(sums + block, tl.sum(value, 0))


@triton.jit
def _residual_normalize(target, accepted, residual, sums,
                        COUNT: tl.constexpr, VOCAB: tl.constexpr,
                        TARGET_STRIDE: tl.constexpr, TARGET_COL: tl.constexpr, PARTS: tl.constexpr,
                        REDUCTION: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < VOCAB
    index = tl.load(accepted)
    parts = tl.arange(0, REDUCTION)
    mass = tl.sum(tl.load(sums + parts, parts < PARTS, other=0), 0)
    value = tl.load(residual + cols, mask, other=0)
    p = tl.load(target + index * TARGET_STRIDE + cols * TARGET_COL, mask, other=0).to(tl.float32)
    # Preserve the bonus distribution and the zero-residual fallback exactly.
    value = tl.where((index < COUNT) & (mass > 0), value / tl.maximum(mass, 1.17549435e-38), p)
    tl.store(residual + cols, value, mask)


def verification_distribution_cuda(target, draft, accepted):
    vocab = target.shape[1]
    block = 1024
    parts = triton.cdiv(vocab, block)
    residual = torch.empty((1, vocab), device=target.device, dtype=torch.float32)
    sums = torch.empty(parts, device=target.device, dtype=torch.float32)
    _residual_partial[(parts,)](target, draft, accepted, residual, sums,
                               draft.shape[0], vocab, target.stride(0), draft.stride(0),
                               target.stride(1), draft.stride(1), block)
    _residual_normalize[(parts,)](target, accepted, residual, sums,
                                 draft.shape[0], vocab, target.stride(0), target.stride(1), parts,
                                 triton.next_power_of_2(parts), block)
    return residual
