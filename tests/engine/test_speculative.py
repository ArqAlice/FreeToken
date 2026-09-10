import torch
import pytest

from freetoken.engine.speculative import (
    acceptance_flags,
    acceptance_prefix,
    matching_prefix,
    rejection_probabilities,
    residual_probabilities,
    verification_distribution,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("count", range(1, 5))
def test_matching_prefix_ignores_matches_after_rejection(device, count):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    draft = torch.arange(count, dtype=torch.int32, device=device)
    for accepted in range(count + 1):
        verified = torch.cat((draft, draft.new_tensor([9])))
        verified[accepted] = 9
        assert matching_prefix(verified, draft).item() == accepted
    assert matching_prefix(verified, draft[:0]).item() == 0


def test_deterministic_proposal_preserves_joint_target_distribution():
    from itertools import product

    first = torch.tensor([.1, .6, .3], dtype=torch.float64)
    conditional = torch.tensor([[0., .2, .8], [.7, .1, .2], [.3, .7, 0.]], dtype=torch.float64)
    # Include proposals outside target support and non-modal proposals.
    for proposal in product(range(3), repeat=2):
        draft = torch.tensor(proposal)
        joint = torch.zeros(3, 3, dtype=torch.float64)
        for a, b in product(range(3), repeat=2):
            verified = torch.tensor([a, b, 0])
            accepted = int(matching_prefix(verified, draft))
            emitted = torch.cat((draft[:accepted], verified[accepted:accepted + 1]))
            mass = first[a] * conditional[proposal[0], b]
            if accepted == 0:
                # The next target call uses the actual correction, not the rejected draft.
                joint[emitted[0]] += mass * conditional[emitted[0]]
            else:
                joint[emitted[0], emitted[1]] += mass
        torch.testing.assert_close(joint, first[:, None] * conditional)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("backend", ["flashinfer", "triton"])
def test_sampler_probabilities_preserve_filter_order_and_ties(monkeypatch, backend):
    import importlib
    from freetoken.engine.sample import Sampler, BatchSamplingArgs

    module = (pytest.importorskip("flashinfer.sampling") if backend == "flashinfer"
              else importlib.import_module("freetoken.kernel.triton.sampling"))
    monkeypatch.setattr("freetoken.engine.sample._sampling_backend", lambda: module)

    sampler = Sampler(torch.device("cuda"), 4)
    logits = torch.tensor([[2., 2., 1., -1.], [3., 2., 1., -1.]], device="cuda")
    args = BatchSamplingArgs(torch.tensor([1.], device="cuda"),
                             top_k=torch.tensor([2], device="cuda"),
                             top_p=torch.tensor([.6], device="cuda"))
    probabilities = sampler.probabilities(logits, args)
    expected = torch.tensor([[.5, .5, 0., 0.], [1., 0., 0., 0.]], device="cuda")
    torch.testing.assert_close(probabilities, expected)


def test_acceptance_and_residual_reconstruct_target_distribution():
    p = torch.tensor([[.1, .3, .6]]).expand(3, -1)
    q = torch.tensor([[.5, .4, .1]]).expand(3, -1)
    tokens = torch.arange(3)
    accepted, residual = rejection_probabilities(p, q, tokens, uniforms=torch.tensor([.19, .76, .99]))
    assert accepted.tolist() == [True, False, True]
    torch.testing.assert_close(residual[0], torch.tensor([0., 0., 1.]))
    direct_mass = torch.minimum(p[0], q[0])
    corrected = direct_mass + (1 - direct_mass.sum()) * residual[0]
    torch.testing.assert_close(corrected, p[0])


def test_disjoint_support_and_identical_distributions():
    p = torch.tensor([[0., 1.], [.25, .75]])
    q = torch.tensor([[1., 0.], [.25, .75]])
    accepted, residual = rejection_probabilities(p, q, torch.tensor([0, 1]), uniforms=torch.zeros(2))
    assert accepted.tolist() == [False, True]
    torch.testing.assert_close(residual, p)


def test_rejection_sampling_empirical_distribution():
    torch.manual_seed(103)
    p = torch.tensor([[.1, .3, .6]]).expand(60000, -1)
    q = torch.tensor([[.5, .4, .1]]).expand_as(p)
    proposal = torch.multinomial(q, 1).flatten()
    accepted, residual = rejection_probabilities(p, q, proposal)
    correction = torch.multinomial(residual, 1).flatten()
    output = torch.where(accepted, proposal, correction)
    observed = torch.bincount(output, minlength=3) / output.numel()
    torch.testing.assert_close(observed, p[0], atol=.01, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("uniforms, expected", [([0., 0., 0.], 3), ([.2, 0., 0.], 0), ([0., .75, 0.], 1)])
def test_acceptance_prefix_stops_at_first_rejection(device, uniforms, expected):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    target = torch.tensor([[.1, .3, .6]], device=device).expand(3, -1)
    draft = torch.tensor([[.5, .4, .1]], device=device).expand(3, -1)
    tokens = torch.arange(3, device=device)
    result = acceptance_prefix(target, draft, tokens, uniforms=torch.tensor(uniforms, device=device))
    assert result.device.type == device
    assert result.ndim == 0
    assert result.item() == expected


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_acceptance_prefix_handles_empty_proposals_and_shared_uniform(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    target = torch.tensor([[0., 1.], [.25, .75]], device=device)
    draft = torch.tensor([[1., 0.], [.25, .75]], device=device)
    tokens = torch.tensor([0, 1], device=device)
    uniforms = torch.tensor([0.], device=device)
    assert acceptance_prefix(target, draft, tokens, uniforms=uniforms).item() == 0
    assert acceptance_prefix(target, draft, tokens, uniforms=0.).item() == 0
    assert acceptance_prefix(target[:0], draft[:0], tokens[:0], uniforms=uniforms).item() == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_verification_distribution_selects_rejection_or_bonus(device, accepted):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    target = torch.tensor([[.1, .3, .6], [0., 1., 0.], [.25, .25, .5], [.7, .2, .1]], device=device)
    draft = torch.tensor([[.5, .4, .1], [1., 0., 0.], [.25, .25, .5]], device=device)
    result = verification_distribution(target, draft, torch.tensor(accepted, device=device))
    expected = (target[-1:] if accepted == 3 else
                residual_probabilities(target[accepted:accepted + 1], draft[accepted:accepted + 1]))
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("vocab", [17, 1025, 248320])
def test_speculative_cuda_matches_reference_with_strided_inputs(vocab):
    torch.manual_seed(502)
    target = torch.rand((8, vocab * 2), device="cuda")[::2, ::2]
    draft = torch.rand((6, vocab * 2), device="cuda")[::2, ::2]
    target /= target.sum(-1, keepdim=True)
    draft /= draft.sum(-1, keepdim=True)
    tokens = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    uniforms = torch.tensor([.2, .8, .4], device="cuda")
    accepted = acceptance_prefix(target[:-1], draft, tokens, uniforms=uniforms)
    expected_accepted = acceptance_flags(target[:-1], draft, tokens, uniforms=uniforms).int().cumprod(0).sum()
    torch.testing.assert_close(accepted, expected_accepted)
    for index in range(4):
        result = verification_distribution(target, draft, accepted.fill_(index))
        expected = (target[-1:] if index == 3 else
                    residual_probabilities(target[index:index + 1], draft[index:index + 1]))
        torch.testing.assert_close(result, expected, atol=1e-7, rtol=2e-6)
        torch.testing.assert_close(result.sum(), torch.ones((), device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_speculative_cuda_graph_replays_changed_uniforms_and_distributions():
    target = torch.tensor([[.1, .3, .6], [.1, .3, .6], [.1, .3, .6], [.7, .2, .1]], device="cuda")
    draft = torch.tensor([[.5, .4, .1]], device="cuda").repeat(3, 1)
    tokens = torch.arange(3, device="cuda")
    uniforms = torch.zeros(3, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            accepted = acceptance_prefix(target[:-1], draft, tokens, uniforms=uniforms)
            verification_distribution(target, draft, accepted)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        accepted = acceptance_prefix(target[:-1], draft, tokens, uniforms=uniforms)
        result = verification_distribution(target, draft, accepted)
    for draws, index in [([0., 0., 0.], 3), ([.3, 0., 0.], 0), ([0., .9, 0.], 1)]:
        uniforms.copy_(torch.tensor(draws, device="cuda"))
        target[-1].copy_(target[-1].roll(1))
        graph.replay()
        assert accepted.item() == index
        expected = (target[-1:] if index == 3 else
                    residual_probabilities(target[index:index + 1], draft[index:index + 1]))
        torch.testing.assert_close(result, expected)
