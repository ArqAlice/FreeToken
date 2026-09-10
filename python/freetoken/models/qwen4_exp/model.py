"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

from dataclasses import replace
import os
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaPlusOneRMSNorm,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import PLELayer

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


def build_linear_mixer(config: ModelConfig, layer_id: int) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        # Qwen3.8's block-fp8 checkpoint keeps the GDN projections bf16 (only the routed
        # experts are quantized), so do not let expert_quant flip them to Fp8Block.
        expert_quant="none" if config.expert_quant == "fp8_block" else config.expert_quant,
        attn_quant=config.attn_quant,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id)
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMoE(config, layer_id)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)
        self.ple = (
            PLELayer(config, layer_id) if layer_id in config.qwen4_args.ple_layer_ids else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.hc_count = config.qwen4_args.hc_count
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def forward(
        self, input_ids: torch.Tensor, batch: Batch, *, return_residual: bool = False,
        compute_output: bool = True,
    ) -> torch.Tensor | None | tuple[torch.Tensor | None, torch.Tensor]:
        hidden = self.embed_tokens.forward(input_ids).repeat(1, self.hc_count)
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
                ple.start_prefetch(batch, meta)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        output = self.hyper_connection_mixer.mix(hidden)[0] if compute_output else None
        return (output, hidden) if return_residual else output


class Qwen4ExpMTP(BaseOP):
    """The Qwen3.8 MTP head over the target model's hyper-connection residual state."""

    def __init__(self, config: ModelConfig) -> None:
        args = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaPlusOneRMSNorm(args.ple_state_width, config.rms_norm_eps)
        self.fc_embedding = LinearReplicated(self.hidden_size, self.hidden_size, has_bias=False)
        self.fc_hidden = LinearReplicated(self.hidden_size, self.hidden_size, has_bias=False)
        first_layer = config.num_layers
        mtp_config = replace(config, dense_quant=args.mtp_dense_quant)
        self.layers = OPList(
            [
                Qwen4ExpDecoderLayer(mtp_config, first_layer + i)
                for i in range(args.mtp_num_hidden_layers)
            ]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

    def forward(
        self, embedding: torch.Tensor, target_residual: torch.Tensor, batch: Batch,
        *, return_residual: bool = False, compute_output: bool = True,
    ) -> torch.Tensor | None | tuple[torch.Tensor | None, torch.Tensor]:
        embed = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(embedding))
        hidden = self.pre_fc_norm_hidden.forward(target_residual)
        hidden = self.fc_hidden.forward(hidden.view(-1, self.hidden_size)).view_as(hidden)
        hidden = hidden.view(-1, self.hc_count, self.hidden_size)
        hidden = (hidden + embed.unsqueeze(1)).flatten(1)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        output = self.hyper_connection_mixer.mix(hidden)[0] if compute_output else None
        return (output, hidden) if return_residual else output


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        self.mtp = (
            Qwen4ExpMTP(config)
            if config.qwen4_args.mtp_num_hidden_layers
            else None
        )
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

        if self.mtp is not None:
            from freetoken.layers.linear import _LinearTPImpl
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear

            def mark(op):
                if isinstance(op, (_LinearTPImpl, Nvfp4DenseLinear)):
                    op._mtp_rowwise = True
                for value in vars(op).values():
                    if isinstance(value, BaseOP):
                        mark(value)
                    elif isinstance(value, list):
                        for child in value:
                            if isinstance(child, BaseOP):
                                mark(child)

            mark(self.model)

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        # Convert after checkpoint materialization, before the engine budgets VRAM.
        self._requantize_gdn_inputs()
        self._draft_lm_head = None
        if self.mtp is not None and os.environ.get("FREETOKEN_MTP_DRAFT_FP8_HEAD", "0") == "1":
            from freetoken.layers.embedding import DraftFP8LMHead
            from freetoken.utils import init_logger

            if not isinstance(self.lm_head, ParallelLMHead):
                raise ValueError("Draft FP8 LM head requires an unquantized target LM head")
            self._draft_lm_head = DraftFP8LMHead(self.lm_head)
            init_logger(__name__).info_rank0(
                f"Draft FP8 LM head: added {self._draft_lm_head._weight_bytes} resident weight bytes")

    def _requantize_gdn_inputs(self):
        self._gdn_fp8_saved_bytes = 0
        if os.environ.get("FREETOKEN_GDN_FP8_INPUT", "0") != "1":
            return
        from freetoken.layers.gdn_fp8 import GDNFP8Input
        from freetoken.utils import init_logger

        for layer in self.model.layers.op_list:
            gdn = layer.linear_attn if hasattr(layer, 'linear_attn') else None
            if gdn is None or not hasattr(gdn, 'in_proj'):
                continue
            if isinstance(gdn.in_proj, GDNFP8Input):
                self._gdn_fp8_saved_bytes += gdn.in_proj._saved_bytes
                continue
            old = gdn.in_proj
            gates = 2 * gdn.num_v_heads * old.local_output_size // old.full_output_size
            gdn.in_proj = GDNFP8Input(old.weight, gates)
            self._gdn_fp8_saved_bytes += gdn.in_proj._saved_bytes
        init_logger(__name__).info_rank0(
            f"GDN FP8 input: released {self._gdn_fp8_saved_bytes} resident weight bytes; b/a gates stay BF16")

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE n-gram table (pinned checkpoint bank, or zeros for dummy weights); returns the pinned host bytes the engine reserves from its pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        if engine_config.ple_backend == "disk":
            from freetoken.utils import download_hf_weight

            from .ple_disk import DiskRowTable, resolve_row_source

            folder = download_hf_weight(engine_config.model_path)
            # one WAIT node per captured graph: the flag protocol supports a single consume
            assert len(ple_layers) == 1, "disk PLE backend expects exactly one PLE layer"
            emb, args = ple_layers[0].ple_embedding, ple_layers[0].args
            # hash with the state-dict-loaded constants, the same source the pinned path reads
            constants = {
                "num_ngram_heads": args.num_ngram_heads,
                "layer_multipliers": emb.layer_multipliers.tolist(),
                "per_head_vocab_sizes": emb.ngram_heads_vocab_sizes.tolist(),
                "per_head_offsets": emb.ngram_heads_offsets.tolist(),
                "eos_token_id": args.ngram_boundary_token_id,
            }
            disk_table = DiskRowTable(
                resolve_row_source(folder),
                constants,
                max_graph_rows=max(256, engine_config.cuda_graph_max_bs or 0),
                max_extend_tokens=engine_config.max_extend_tokens,
            )
            self._ple_table = disk_table
            for ple in ple_layers:
                ple.ple_embedding.attach_table(disk_table)
            # engine enters this around every dispatch; the graph itself never waits on the disk
            self.forward_host_ctx = disk_table.forward_host_ctx
            return 0

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale))
            )
        return table.bank.nbytes

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        return self.lm_head.forward(self.model.forward(batch.input_ids, batch))

    def forward_with_target_residual(
        self, *, all_tokens: bool = False, compute_logits: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        batch = get_global_ctx().batch
        output, residual = self.model.forward(batch.input_ids, batch, return_residual=True,
                                              compute_output=compute_logits)
        return (self.lm_head.forward(output, all_tokens=all_tokens) if compute_logits else None), residual

    def forward_mtp(
        self, input_ids: torch.Tensor, target_residual: torch.Tensor, batch: Batch,
        *, return_residual: bool = False, compute_logits: bool = True,
    ) -> torch.Tensor | None | tuple[torch.Tensor | None, torch.Tensor]:
        assert self.mtp is not None, "checkpoint has no MTP head"
        embedding = self.model.embed_tokens.forward(input_ids)
        output, residual = self.mtp.forward(embedding, target_residual, batch, return_residual=True,
                                           compute_output=compute_logits)
        logits = None
        if compute_logits:
            head = getattr(self, "_draft_lm_head", None) or self.lm_head
            logits = head.forward(output)
        return (logits, residual) if return_residual else logits


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel", "Qwen4ExpMTP", "build_linear_mixer"]
