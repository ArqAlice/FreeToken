"""Runtime-only FP8 GDN input weights with BF16 recurrence gates."""
import torch

from freetoken.layers.base import BaseOP


class GDNFP8Input(BaseOP):
    def __init__(self, weight: torch.Tensor, gate_rows: int):
        if weight.ndim != 2:
            raise ValueError('GDN FP8 input weights must be a matrix')
        if weight.dtype != torch.bfloat16 or not weight.is_cuda:
            raise ValueError('GDN FP8 requantization requires CUDA BF16 input weights')
        if torch.cuda.get_device_capability(weight.device) < (8, 9):
            raise ValueError('GDN FP8 input requires NVIDIA SM89 or newer')
        if not 0 < gate_rows < weight.shape[0]:
            raise ValueError('GDN FP8 gate_rows must leave nonempty projection and gate weights')
        if not torch.isfinite(weight).all().item():
            raise ValueError('GDN FP8 input weights must be finite')
        q = weight.shape[0] - gate_rows
        self.weight = torch.empty((q, weight.shape[1]), dtype=torch.float8_e4m3fn, device=weight.device)
        self.weight_scale = torch.empty(q, dtype=torch.float32, device=weight.device)
        # A contiguous slice can retain the entire BF16 allocation without clone.
        self.weight_tail = weight[q:].contiguous().clone()
        for start in range(0, q, 1024):
            chunk = weight[start:min(start+1024, q)].float()
            scale = (chunk.abs().amax(1) / 448.).clamp_min(1.e-12)
            self.weight[start:start+chunk.shape[0]] = (chunk / scale[:, None]).clamp(-448., 448.).to(self.weight.dtype)
            self.weight_scale[start:start+chunk.shape[0]] = scale
        self._saved_bytes = weight.numel()*weight.element_size() - sum(
            t.numel()*t.element_size() for t in (self.weight, self.weight_scale, self.weight_tail))

    def forward(self, x):
        from freetoken.kernel.triton.bf16_shared_linear import fp8_shared_linear

        if x.ndim != 2 or x.dtype != torch.bfloat16 or x.device != self.weight.device:
            raise ValueError('GDN FP8 input requires a BF16 matrix on the weight device')
        m, k = x.shape
        if k != self.weight.shape[1]:
            raise ValueError('GDN FP8 input width does not match the weight')
        if m == 0:
            return x.new_empty((0, self.weight.shape[0] + self.weight_tail.shape[0]))
        return fp8_shared_linear(x, self.weight, self.weight_scale, self.weight_tail)
