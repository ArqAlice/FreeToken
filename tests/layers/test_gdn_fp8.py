import pytest
import torch

from freetoken.layers.gdn_fp8 import GDNFP8Input

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason='needs NVIDIA SM89 or newer')


@pytest.mark.parametrize('rows', [0, 1, 4, 16, 17, 80])
@pytest.mark.parametrize('shape', [(272, 513, 17), (2560, 16480, 96)])
def test_quantized_projection_and_graph(rows, shape):
    torch.manual_seed(953)
    k, n, gates = shape
    w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    w[0].zero_()
    op = GDNFP8Input(w, gates)
    q = n - gates
    torch.testing.assert_close(op.weight_tail, w[q:], rtol=0, atol=0)
    assert op.weight_tail.untyped_storage().nbytes() == gates*k*2
    assert op.weight_tail.untyped_storage().data_ptr() != w.untyped_storage().data_ptr()
    assert op._saved_bytes == q*(k-4)
    dequant = op.weight.float()*op.weight_scale[:, None]
    assert (dequant-w[:q].float()).norm()/w[:q].float().norm() < .03
    x = torch.randn(rows, k*2, device='cuda', dtype=torch.bfloat16)[:, ::2]

    def check(actual):
        reference = torch.cat((x.double() @ op.weight.double().t()*op.weight_scale.double(),
                               x.double() @ op.weight_tail.double().t()), -1)
        torch.testing.assert_close(actual.double(), reference, rtol=.004, atol=.001)
        if 1 <= rows <= 16:
            singles = torch.cat([op.forward(row) for row in x.split(1)])
            torch.testing.assert_close(actual, singles, rtol=0, atol=0)

    check(op.forward(x))
    if rows == 0:
        return
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op.forward(x)
    x.add_(.25)
    graph.replay()
    check(actual)


def test_invalid_weights_and_inputs(monkeypatch):
    w = torch.zeros(33, 48, device='cuda', dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='gate_rows'):
        GDNFP8Input(w, 33)
    with pytest.raises(ValueError, match='CUDA BF16'):
        GDNFP8Input(w.float(), 1)
    op = GDNFP8Input(w, 1)
    assert torch.isfinite(op.weight_scale).all()
    assert op.forward(torch.ones(2, 48, device='cuda', dtype=torch.bfloat16)).count_nonzero() == 0
    with pytest.raises(ValueError, match='BF16 matrix'):
        op.forward(torch.zeros(1, 48, device='cuda'))
    with pytest.raises(ValueError, match='width'):
        op.forward(torch.zeros(1, 47, device='cuda', dtype=torch.bfloat16))
    w[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        GDNFP8Input(w, 1)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (8, 6))
    with pytest.raises(ValueError, match='SM89'):
        GDNFP8Input(w, 1)


def test_model_post_load_quantization(monkeypatch):
    from freetoken.layers.base import BaseOP, OPList
    from freetoken.layers.linear import LinearReplicated
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model.model = BaseOP()
    layer = BaseOP()
    layer.linear_attn = BaseOP()
    layer.linear_attn.num_v_heads = 4
    layer.linear_attn.in_proj = LinearReplicated(48, 136, False)
    layer.linear_attn.in_proj.weight = torch.empty(136, 48, device='cuda', dtype=torch.bfloat16)
    model.model.layers = OPList([layer])
    w = torch.randn_like(layer.linear_attn.in_proj.weight)
    monkeypatch.setenv('FREETOKEN_GDN_FP8_INPUT', '0')
    model.load_state_dict({'model.layers.0.linear_attn.in_proj.weight': w})
    assert layer.linear_attn.in_proj.weight is w
    monkeypatch.setenv('FREETOKEN_GDN_FP8_INPUT', '1')
    model.load_state_dict({'model.layers.0.linear_attn.in_proj.weight': w})
    op = layer.linear_attn.in_proj
    assert isinstance(op, GDNFP8Input)
    assert model._gdn_fp8_saved_bytes == 128*44
    model._requantize_gdn_inputs()
    assert layer.linear_attn.in_proj is op
    assert model._gdn_fp8_saved_bytes == 128*44
