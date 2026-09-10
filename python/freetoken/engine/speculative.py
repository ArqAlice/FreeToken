"""Distribution-preserving acceptance and residual sampling for draft tokens."""

import torch


def matching_prefix(verified, draft):
    """Count draft tokens matching independent target samples up to first mismatch."""
    return draft.eq(verified[:draft.numel()]).to(torch.int32).cumprod(0).sum()


def rejection_probabilities(target, draft, tokens, *, uniforms=None):
    """Return per-proposal acceptance flags and normalized positive residuals."""
    return acceptance_flags(target, draft, tokens, uniforms=uniforms), residual_probabilities(target, draft)


def acceptance_flags(target, draft, tokens, *, uniforms=None):
    target, draft = target.float(), draft.float()
    indices = tokens.long().reshape(-1, 1)
    p = target.gather(1, indices).squeeze(1)
    q = draft.gather(1, indices).squeeze(1)
    if uniforms is None:
        uniforms = torch.rand_like(p)
    return uniforms * q < p


def acceptance_prefix(target, draft, tokens, *, uniforms=None):
    """Count consecutive accepted proposals without reading their flags on the CPU."""
    if target.is_cuda:
        from freetoken.kernel.triton.speculative import acceptance_prefix_cuda

        return acceptance_prefix_cuda(target, draft, tokens, uniforms=uniforms)
    flags = acceptance_flags(target, draft, tokens, uniforms=uniforms)
    return flags.to(torch.int64).cumprod(0).sum()


def verification_distribution(target, draft, accepted):
    """Select the bonus or normalized first-rejection distribution on the device."""
    if target.is_cuda:
        from freetoken.kernel.triton.speculative import verification_distribution_cuda

        return verification_distribution_cuda(target, draft, accepted)
    index = int(accepted)
    if index == draft.shape[0]:
        return target[index:index + 1]
    return residual_probabilities(target[index:index + 1], draft[index:index + 1])


def residual_probabilities(target, draft):
    target, draft = target.float(), draft.float()
    residual = (target - draft).clamp_min(0)
    mass = residual.sum(-1, keepdim=True)
    # A zero residual is unreachable on rejection in exact arithmetic; keep a
    # valid distribution if fp32 rounding eliminates the residual mass.
    residual = torch.where(mass > 0, residual / mass.clamp_min(torch.finfo(mass.dtype).tiny), target)
    return residual
