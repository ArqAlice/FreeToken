from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.layers.moe import OffloadMoELayer, make_moe_layer
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE
from freetoken.moe.fused import fused_topk

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None) -> None:
        self._mtp_expert_overlap = os.environ.get("FREETOKEN_MTP_EXPERT_OVERLAP", "1") == "1"
        self._resident_mtp = (getattr(config.qwen4_args, "mtp_bf16_experts", False)
                              and layer_id is not None and layer_id >= config.num_layers)
        if getattr(config, "expert_quant", "none") != "fp8_block":
            super().__init__(config, layer_id=layer_id)
            return
        # Qwen3.8's block-fp8 checkpoint quantizes only the routed experts; the shared
        # expert stays bf16, so hide expert_quant from _SharedExpert's fp8 branch and
        # rebuild the routed experts with the fp8_block bank layout.
        super().__init__(replace(config, expert_quant="none"), layer_id=layer_id)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            weight_format="fp8_block",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        pending = (self.experts.prefetch_decode(hidden_states, router_logits)
                   if self._mtp_expert_overlap and isinstance(self.experts, OffloadMoELayer)
                   else nullcontext(None))
        with pending as routing:
            shared = self.shared_expert.forward(hidden_states)
            gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        if self._resident_mtp:
            # The global backend remains NVFP4 offload; this head owns BF16 resident weights.
            weights, ids = fused_topk(hidden_states, router_logits, self.experts.top_k,
                                     self.experts.renormalize)
            routed = self.experts.routed_forward(hidden_states, weights, ids)
        else:
            routed = (self.experts.forward_prefetched(hidden_states, routing) if routing is not None else
                      self.experts.forward(hidden_states=hidden_states, router_logits=router_logits))
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)


__all__ = ["Qwen4ExpMoE"]
