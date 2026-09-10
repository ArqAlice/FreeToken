from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from freetoken.layers.linear import LinearReplicated


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('tied,bias', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('prefill', [False, True])
def test_draft_fp8_head_keeps_target_weights(monkeypatch, tied, bias, prefill):
    from freetoken.layers.embedding import DraftFP8LMHead

    torch.manual_seed(518)
    weight = torch.randn(513, 272, device='cuda', dtype=torch.bfloat16)
    before = weight.clone()
    source = SimpleNamespace(weight=weight if not tied else torch.zeros_like(weight),
        tied_embedding=SimpleNamespace(weight=weight) if tied else None,
        bias=torch.randn(513, device='cuda', dtype=torch.bfloat16) if bias else None, tp_size=1)
    head = DraftFP8LMHead(source)
    batch = SimpleNamespace(size=2, is_prefill=prefill,
        attn_metadata=SimpleNamespace(get_last_indices=lambda n: torch.tensor([1, 3], device='cuda')))
    # Keep metadata addresses fixed for graph replay.
    indices = torch.tensor([1, 3], device='cuda')
    batch.attn_metadata.get_last_indices = lambda n: indices
    monkeypatch.setattr('freetoken.layers.embedding.get_global_ctx', lambda: SimpleNamespace(batch=batch))
    x = torch.randn(4, 544, device='cuda', dtype=torch.bfloat16)[:, ::2]

    def expected():
        value = x[indices] if prefill else x
        out = ((value.float() @ head.weight.float().T) * head.weight_scale).bfloat16()
        return out + source.bias if bias else out

    torch.testing.assert_close(head.forward(x), expected(), rtol=.005, atol=.002)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = head.forward(x)
    x.mul_(.8)
    graph.replay()
    torch.testing.assert_close(actual, expected(), rtol=.005, atol=.002)
    assert torch.equal(weight, before)
    assert head._weight_bytes == weight.numel() + 4 * weight.shape[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('tied,bias', [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize('phase', ['verify', 'prefill', 'decode'])
def test_bf16_lm_head_fast_verification(monkeypatch, enabled, tied, bias, phase):
    from freetoken.layers.embedding import ParallelLMHead
    from freetoken.layers.linear import rowwise_linear

    monkeypatch.setenv('FREETOKEN_FAST_LINEAR', str(int(enabled)))
    torch.manual_seed(806)
    weight = torch.randn(513, 272, device='cuda', dtype=torch.bfloat16)
    head = SimpleNamespace(weight=weight if not tied else torch.zeros_like(weight),
                           tied_embedding=SimpleNamespace(weight=weight) if tied else None,
                           bias=torch.randn(513, device='cuda', dtype=torch.bfloat16) if bias else None,
                           tp_size=1)
    indices = torch.tensor([1, 3], device='cuda')
    batch = SimpleNamespace(size=1 if phase == 'verify' else 2 if phase == 'prefill' else 4,
                            is_prefill=phase != 'decode',
                            attn_metadata=SimpleNamespace(get_last_indices=lambda bs: indices))
    monkeypatch.setattr('freetoken.layers.embedding.get_global_ctx', lambda: SimpleNamespace(batch=batch))
    x = torch.randn(4, 544, device='cuda', dtype=torch.bfloat16)[:, ::2]
    all_tokens = phase == 'verify'
    values = x[indices].contiguous() if phase == 'prefill' else x
    fn = rowwise_linear if all_tokens and not enabled else F.linear
    expected = fn(values, weight, head.bias)
    calls = []
    linear = F.linear

    def record(values, weights, bias=None):
        calls.append(values.shape)
        return linear(values, weights, bias)

    with monkeypatch.context() as patch:
        patch.setattr(F, 'linear', record)
        actual = ParallelLMHead.forward(head, x, all_tokens=all_tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == ([] if all_tokens and not enabled else [values.shape])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ParallelLMHead.forward(head, x, all_tokens=all_tokens)
    x.add_(.125)
    graph.replay()
    values = x[indices].contiguous() if phase == 'prefill' else x
    torch.testing.assert_close(actual, fn(values, weight, head.bias), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('phase', ['decode', 'verify', 'prefill'])
@pytest.mark.parametrize('bias', [False, True])
def test_fast_bf16_projection_dispatch(monkeypatch, shared, phase, bias):
    from freetoken.kernel.triton.bf16_shared_linear import bf16_shared_linear

    monkeypatch.setenv('FREETOKEN_FAST_LINEAR', '1')
    batch = SimpleNamespace(is_decode=phase == 'decode', use_decode_moe=phase == 'verify',
                            mtp_batched_linear=False)
    monkeypatch.setattr('freetoken.core.get_global_ctx', lambda: SimpleNamespace(batch=batch))
    op = LinearReplicated(272, 513, has_bias=bias)
    op.weight = torch.randn(513, 272, device='cuda', dtype=torch.bfloat16)
    op.bias = torch.randn(513, device='cuda', dtype=torch.bfloat16) if bias else None
    op._mtp_rowwise = True
    op._shared_decode = shared
    x = torch.randn(4, 272, device='cuda', dtype=torch.bfloat16)
    expected = (bf16_shared_linear(x, op.weight) if shared and not bias and phase != 'prefill'
                else F.linear(x, op.weight, op.bias))
    torch.testing.assert_close(op.forward(x), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('count', [1, 2, 3, 4, 5, 16])
@pytest.mark.parametrize('shape', [(272, 513), (2560, 16480)])
def test_bf16_shared_projection_fixed_reduction_graph(count, shape):
    from freetoken.kernel.triton.bf16_shared_linear import bf16_shared_linear

    torch.manual_seed(947)
    k, n = shape
    weight = torch.randn(n + 3, k, device='cuda', dtype=torch.bfloat16)[1:n+1]
    x = torch.randn(count, k * 2, device='cuda', dtype=torch.bfloat16)[:, ::2]

    def check(actual):
        singles = torch.cat([bf16_shared_linear(row, weight) for row in x.split(1)])
        torch.testing.assert_close(actual, singles, rtol=0, atol=0)
        reference = x.double() @ weight.double().t()
        torch.testing.assert_close(actual.double(), reference, rtol=.004, atol=.0005)

    check(bf16_shared_linear(x, weight))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = bf16_shared_linear(x, weight)
    for _ in range(3):
        x.add_(.125)
        graph.replay()
        check(actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('phase', ['decode', 'verify', 'prefill'])
@pytest.mark.parametrize('enabled', [False, True])
def test_shared_projection_dispatch(monkeypatch, phase, enabled):
    from freetoken.kernel.triton.bf16_shared_linear import bf16_shared_linear
    from freetoken.layers.linear import rowwise_linear

    monkeypatch.setenv('FREETOKEN_GDN_SHARED_INPUT', '1' if enabled else '0')
    batch = SimpleNamespace(is_decode=phase == 'decode', use_decode_moe=phase == 'verify')
    monkeypatch.setattr('freetoken.core.get_global_ctx', lambda: SimpleNamespace(batch=batch))
    op = LinearReplicated(272, 513, has_bias=False)
    op.weight = torch.randn(513, 272, device='cuda', dtype=torch.bfloat16)
    op._shared_decode = op._mtp_rowwise = True
    x = torch.randn(3, 272, device='cuda', dtype=torch.bfloat16)
    fn = (bf16_shared_linear if enabled and phase != 'prefill' else
          rowwise_linear if phase == 'verify' else F.linear)
    torch.testing.assert_close(op.forward(x), fn(x, op.weight), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("count", [2, 5])
@pytest.mark.parametrize("shape", [(2560, 512), (2560, 32768), (272, 513)])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("shared_rows", [False, True])
def test_mtp_nvfp4_linear_matches_single_row_reduction(
    monkeypatch, transposed, count, shape, has_bias, shared_rows,
):
    from freetoken.kernel.triton.nvfp4_linear import (
        Nvfp4DenseLinear, nvfp4_dense_linear, nvfp4_dense_linear_t, nvfp4_transpose_resident,
    )

    monkeypatch.setattr("freetoken.core.get_global_ctx", lambda: SimpleNamespace(
        batch=SimpleNamespace(use_decode_moe=True)))
    monkeypatch.setenv("FREETOKEN_MTP_NVFP4_SHARED_ROWS", "1" if shared_rows else "0")
    torch.manual_seed(31)
    width, out = shape
    op = Nvfp4DenseLinear(width, out, has_bias=False)
    op.weight = torch.randint(0, 256, (out, width // 2), device="cuda", dtype=torch.uint8)
    op.weight_scale = (torch.rand(out, width // 16, device="cuda") + .1).to(torch.float8_e4m3fn)
    op.weight_global = torch.full((out,), .1, dtype=torch.float16, device="cuda")
    op.bias = torch.randn(out, device="cuda", dtype=torch.bfloat16) if has_bias else None
    if transposed:
        op.weight, op.weight_scale = nvfp4_transpose_resident(op.weight, op.weight_scale)
        op._transposed = True
    x = torch.randn(count, width * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    op._mtp_rowwise = True
    fn = nvfp4_dense_linear_t if transposed else nvfp4_dense_linear
    expected = torch.cat([fn(row.contiguous(), op.weight, op.weight_scale, op.weight_global, op.bias)
                          for row in x.split(1)])
    torch.testing.assert_close(op.forward(x), expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op.forward(x)
    for _ in range(3):
        x.add_(.25)
        expected = torch.cat([fn(row.contiguous(), op.weight, op.weight_scale, op.weight_global, op.bias)
                              for row in x.split(1)])
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("shape", [(2560, 512), (2560, 32768), (272, 513)])
def test_mtp_nvfp4_shared_rows_keeps_fp32_reduction(transposed, shape):
    from freetoken.kernel.triton.nvfp4_linear import (
        _gemv, nvfp4_dense_linear, nvfp4_dense_linear_t, nvfp4_transpose_resident,
    )

    torch.manual_seed(53)
    width, out = shape
    weight = torch.randint(0, 256, (out, width // 2), device="cuda", dtype=torch.uint8)
    scale = (torch.rand(out, width // 16, device="cuda") + .1).to(torch.float8_e4m3fn)
    global_scale = (torch.rand(out, device="cuda") + .1).to(torch.float16)
    if transposed:
        weight, scale = nvfp4_transpose_resident(weight, scale)
    x = torch.randn(3, width * 2, device="cuda", dtype=torch.float32)[:, :width]
    fn = nvfp4_dense_linear_t if transposed else nvfp4_dense_linear
    expected = torch.cat([fn(row, weight, scale, global_scale) for row in x.split(1)])
    packed = weight.t() if transposed else weight.view(torch.int32)
    scales = scale.t() if transposed else scale
    actual = _gemv(x, packed, scales, global_scale, x.dtype, transposed, shared_rows=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("count", [2, 5])
@pytest.mark.parametrize("shape", [(2560, 336), (10240, 336), (320, 10240), (2560, 10240)])
def test_mtp_linear_matches_single_row_reduction(monkeypatch, has_bias, shape, count):
    monkeypatch.setattr("freetoken.core.get_global_ctx", lambda: SimpleNamespace(
        batch=SimpleNamespace(use_decode_moe=True)))
    with torch.device("cuda"):
        width, out = shape
        op = LinearReplicated(width, out, has_bias=has_bias)
        op.weight = torch.randn(out, width, dtype=torch.bfloat16)
        op.bias = torch.randn(out, dtype=torch.bfloat16) if has_bias else None
        x = torch.randn(count, width, dtype=torch.bfloat16)
    op._mtp_rowwise = True
    expected = torch.cat([F.linear(row, op.weight, op.bias) for row in x.split(1)])
    torch.testing.assert_close(op.forward(x), expected, rtol=0, atol=0)
    op._mtp_rowwise = False
    torch.testing.assert_close(op.forward(x), F.linear(x, op.weight, op.bias), rtol=0, atol=0)


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))])
def test_mtp_batched_linear_uses_standard_gemm_when_enabled(monkeypatch, has_bias, batched, device):
    from freetoken.core import Batch

    batch = Batch([], "prefill")
    assert not batch.mtp_batched_linear
    batch.use_decode_moe = True
    batch.mtp_batched_linear = batched
    monkeypatch.setattr("freetoken.core.get_global_ctx", lambda: SimpleNamespace(batch=batch))
    torch.manual_seed(702)
    op = LinearReplicated(33, 17, has_bias=has_bias)
    op.weight = torch.randn(17, 33, dtype=torch.bfloat16, device=device)
    op.bias = torch.randn(17, dtype=torch.bfloat16, device=device) if has_bias else None
    op._mtp_rowwise = True
    x = torch.randn(3, 66, dtype=torch.bfloat16, device=device)[:, ::2]
    expected = (F.linear(x, op.weight, op.bias) if batched else
                torch.cat([F.linear(row, op.weight, op.bias) for row in x.split(1)]))
    linear = F.linear
    calls = []

    def record_linear(values, weight, bias=None):
        calls.append(values.shape)
        return linear(values, weight, bias)

    monkeypatch.setattr(F, "linear", record_linear)
    torch.testing.assert_close(op.forward(x), expected, rtol=0, atol=0)
    assert calls == ([x.shape] if batched else [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('count,width,out', [(2, 2560, 1280), (3, 10240, 336),
                                           (4, 6144, 2560), (4, 2560, 512)])
def test_parallel_rows_match_single_row_graphs(monkeypatch, count, width, out):
    from freetoken.layers.linear import rowwise_linear

    torch.manual_seed(715)
    x = torch.randn(count, width * 2, device='cuda', dtype=torch.bfloat16)[:, ::2]
    weight = torch.randn(out, width, device='cuda', dtype=torch.bfloat16)
    monkeypatch.setenv('FREETOKEN_MTP_PARALLEL_LINEAR', '0')
    expected = rowwise_linear(x, weight)
    monkeypatch.setenv('FREETOKEN_MTP_PARALLEL_LINEAR', '1')
    torch.testing.assert_close(rowwise_linear(x, weight), expected, rtol=0, atol=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = rowwise_linear(x, weight)
    for _ in range(3):
        graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
