from __future__ import annotations

from typing import List
from functools import lru_cache
import os

import torch
import torch.nn.functional as F
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.utils import div_even

from .base import BaseOP


@lru_cache(maxsize=None)
def _row_streams(device):
    return tuple(torch.cuda.Stream(device=device) for _ in range(4))


def _parallel_rows(x, weight, bias):
    if (bias is not None or not x.is_cuda or x.dtype != torch.bfloat16
            or not 2 <= x.shape[0] <= 4
            or os.environ.get('FREETOKEN_MTP_PARALLEL_LINEAR', '0') != '1'):
        return False
    count, width = x.shape
    out = weight.shape[0]
    return ((width, out) == (2560, 1280)
            or count >= 3 and (width, out) in ((2560, 336), (10240, 336))
            or count == 4 and (width, out) in ((6144, 2560), (2560, 2560), (2560, 512)))


def rowwise_linear(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    """Preserve single-row GEMMs without launching a concatenation kernel."""
    output = x.new_empty((x.shape[0], weight.shape[0]))
    transposed = weight.t()
    if _parallel_rows(x, weight, bias):
        compute = torch.cuda.current_stream(x.device)
        streams = _row_streams(x.device)[:x.shape[0]]
        for stream in streams:
            stream.wait_stream(compute)
        try:
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    torch.mm(x[i:i + 1], transposed, out=output[i:i + 1])
        finally:
            # Join every reader/writer before callers reuse inputs or consume the result.
            for stream in streams:
                compute.wait_stream(stream)
        return output
    for i in range(x.shape[0]):
        if bias is None:
            torch.mm(x[i:i + 1], transposed, out=output[i:i + 1])
        else:
            torch.addmm(bias, x[i:i + 1], transposed, out=output[i:i + 1])
    return output


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None
        self._mtp_rowwise = False
        self._shared_decode = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self._shared_decode and x.is_cuda and x.ndim == 2 and 1 <= x.shape[0] <= 5
                and x.dtype == self.weight.dtype == torch.bfloat16 and self.bias is None
                and self.weight.stride(1) == 1
                and os.environ.get("FREETOKEN_GDN_SHARED_INPUT", "0") == "1"):
            from freetoken.core import get_global_ctx
            from freetoken.kernel.triton.bf16_shared_linear import bf16_shared_linear

            batch = get_global_ctx().batch
            if batch.is_decode or batch.use_decode_moe:
                # The same reduction serves ordinary decode and MTP verification.
                return bf16_shared_linear(x, self.weight)
        if self._mtp_rowwise and x.ndim == 2 and x.shape[0] > 1:
            from freetoken.core import get_global_ctx

            batch = get_global_ctx().batch
            if batch.use_decode_moe and not getattr(batch, "mtp_batched_linear", False):
                # Keep the GEMM reduction order used by one-token target decoding.
                return rowwise_linear(x, self.weight, self.bias)
        # Opt-in batched verification uses the ordinary prefill GEMM; its
        # floating-point reduction can differ from single-token decoding.
        return F.linear(x, self.weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
