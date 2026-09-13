"""Start the registered V4.1 engine in a fresh CUDA process, then serve an image request."""

import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import torch


def _run_engine_smoke(folder, port):
    import json
    from dataclasses import asdict

    from freetoken.core import Batch, Req, SamplingParams
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import Engine
    from freetoken.models.deepseek_v41.args import DeepseekV41Args
    from freetoken.models.deepseek_v41.image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START

    args = DeepseekV41Args(
        n_layers=3, n_mtp_layers=0, compress_ratios=(0, 2, 2),
        kv_source_layers=(1,), index_source_layers=(1,), candidate_source_layer=-1,
        dim=64, n_heads=2, head_dim=32, rope_head_dim=16, q_lora_rank=32,
        o_lora_rank=32, o_groups=2, window_size=8, index_n_heads=2, index_head_dim=32,
        index_topk=3, moe_inter_dim=64, n_routed_experts=4, n_activated_experts=2,
        vocab_size=128, hc_mult=2, engram_layer_ids=(1,), engram_num_embeddings=(17,),
        engram_max_ngram_size=2, engram_vocab_size=17, engram_n_heads=1,
        engram_head_dim=32, engram_compressed_vocab_size=128,
        vision_n_layers=1, vision_dim=32, vision_n_heads=2, vision_inter_dim=32,
        vision_patch_size=2, vision_downsample_ratio=2, image_token_id=127,
    )
    raw = asdict(args) | {
        "architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
        "quantization_config": {"moe_quant_algo": "NVFP4"},
        "vision_config": {"num_hidden_layers": 1, "hidden_size": 32, "num_attention_heads": 2,
                          "intermediate_size": 32, "patch_size": 2, "downsample_ratio": 2},
    }
    Path(folder, "config.json").write_text(json.dumps(raw))

    class LocalEngineConfig(EngineConfig):
        @property
        def distributed_addr(self):
            return f"tcp://127.0.0.1:{port}"

    config = LocalEngineConfig(
        model_path=folder, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=1, moe_backend="offload", moe_cpu_layers="", moe_cache_size=8,
        moe_prefill_overlap=True, use_dummy_weight=True, use_pynccl=False,
        max_seq_len_override=64, num_page_override=40, cuda_graph_max_bs=0,
    )
    engine = Engine(config)
    assert engine.model._engram_runtime is not None
    assert config.attention_backend == "dsv41_sparse"
    assert config.model_config.expert_quant == "nvfp4" and config.page_size == 8
    assert config.model_config.is_multimodal
    assert engine.model._transformer.vision.patch_embed.proj.weight.dtype == torch.bfloat16
    runtime = engine.model._engram_runtime
    resident_bytes = sum(p.numel() * p.element_size() for p in engine.model.state_dict().values())
    staged_bytes = sum(module._values.numel() * module._values.element_size() for module in runtime.modules)
    staged_bytes += runtime.device_mask.numel() * runtime.device_mask.element_size()
    assert engine._weights_bytes >= resident_bytes + staged_bytes
    sample = engine.sampler.sample

    def checked_sample(logits, sampling_args):
        assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
        return sample(logits, sampling_args)

    engine.sampler.sample = checked_sample
    engine.page_table[0, :16] = torch.arange(16, device="cuda")
    for start in (0, 8):
        engine.kv_cache.bind_window_pages(start, start)
    media = [{"start": 1, "types": torch.tensor([IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END]),
              "patches": torch.randn(4, 3, 2, 2), "n_vit_h": 2, "n_vit_w": 2}]
    req = Req(torch.tensor([5, 127, 127, 127, 127, 6, 7, 8], dtype=torch.int32),
              0, 0, 3, 0, SamplingParams(temperature=0), None, media=media)
    for phase in ("prefill", "decode"):
        batch = Batch([req], phase)
        batch.padded_reqs = batch.reqs
        batch.input_ids = req.input_ids[req.cached_len:].cuda()
        batch.positions = torch.arange(req.cached_len, req.device_len, device="cuda")
        batch.active_table_idx = torch.tensor([0], dtype=torch.long, device="cuda")
        batch.out_loc = engine.page_table[0, req.cached_len:req.device_len]
        engine.attn_backend.prepare_metadata(batch)
        with torch.inference_mode():
            output = engine.forward_batch(batch, engine.sampler.prepare(batch))
        output.copy_done_event.synchronize()
        assert output.next_tokens_cpu.shape == (1,)
        assert 0 <= output.next_tokens_cpu.item() < 128
        req.append_host(output.next_tokens_cpu)
    assert "embeddings" in media[0]
    assert req.cached_len == 9 and req.device_len == 10
    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()
    print("V41_ENGINE_PREFILL_DECODE_IMAGE_OK")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_engine_initialization_and_image_generation(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path), str(port)],
                            capture_output=True, text=True, timeout=180, env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "V41_ENGINE_PREFILL_DECODE_IMAGE_OK" in result.stdout


if __name__ == "__main__":
    _run_engine_smoke(sys.argv[1], int(sys.argv[2]))
