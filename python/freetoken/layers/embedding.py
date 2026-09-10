from __future__ import annotations

from typing import Dict
import os

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.utils import div_ceil, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float | None = None,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        # Gemma scales embeddings by sqrt(hidden_size). The scale is materialized in
        # the weight dtype (bf16) to match HF, which downcasts the scalar. The GPU
        # scalar is built lazily (model __init__ runs on the meta device) and cached
        # so it is not reallocated inside a captured CUDA graph.
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel import indexing

        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        if self.tp_size > 1:
            y = self._comm.all_reduce(y)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(
                    self._embed_scale, dtype=y.dtype, device=y.device
                )
            y = y * self._embed_scale_t
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor, *, all_tokens: bool = False) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill and not all_tokens:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        module = self.tied_embedding or self
        fast_bf16 = (x.is_cuda and x.dtype == module.weight.dtype == torch.bfloat16
                     and os.environ.get("FREETOKEN_FAST_LINEAR", "0") == "1")
        if all_tokens and x.shape[0] > 1 and not fast_bf16:
            from .linear import rowwise_linear

            logits = rowwise_linear(x, module.weight, self.bias)
        else:
            logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]


class DraftFP8LMHead(BaseOP):
    """Runtime-only FP8 copy; the target head and checkpoint remain unchanged."""

    def __init__(self, source: ParallelLMHead):
        weight = (source.tied_embedding or source).weight
        if not weight.is_cuda or weight.dtype != torch.bfloat16:
            raise ValueError("Draft FP8 LM head requires CUDA BF16 weights")
        if torch.cuda.get_device_capability(weight.device) < (8, 9):
            raise ValueError("Draft FP8 LM head requires NVIDIA SM89 or newer")
        self._source = source
        self.weight = torch.empty(weight.shape, device=weight.device, dtype=torch.float8_e4m3fn)
        self.weight_scale = torch.empty(weight.shape[0], device=weight.device, dtype=torch.float32)
        self._tail = weight.new_empty((0, weight.shape[1]))
        for start in range(0, weight.shape[0], 1024):
            chunk = weight[start:start + 1024].float()
            if not torch.isfinite(chunk).all().item():
                raise ValueError("Draft FP8 LM head weights must be finite")
            scale = (chunk.abs().amax(1) / 448.).clamp_min(1.e-12)
            self.weight[start:start + chunk.shape[0]] = (chunk / scale[:, None]).clamp(-448., 448.).to(self.weight.dtype)
            self.weight_scale[start:start + chunk.shape[0]] = scale
        self._weight_bytes = self.weight.numel() + self.weight_scale.numel() * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.bf16_shared_linear import fp8_shared_linear

        batch = get_global_ctx().batch
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        logits = fp8_shared_linear(x, self.weight, self.weight_scale, self._tail)
        source = self._source
        if source.bias is not None:
            logits = logits + source.bias
        if source.tp_size == 1:
            return logits
        gathered = source._comm.all_gather(logits).view(source.tp_size, *logits.shape)
        return gathered.permute(1, 0, 2).reshape(logits.shape[0], -1)[:, :source.num_embeddings]
