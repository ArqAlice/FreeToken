"""Small native NVFP4 shards exercise loading without the released weight payloads."""

import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from freetoken.models.deepseek_v41 import weight


@pytest.fixture(params=[True, False], ids=["modelopt_input_scale", "native_no_input_scale"])
def tiny_shards(tmp_path, request):
    config = SimpleNamespace(num_layers=1, num_experts=2, hidden_size=32, moe_intermediate_size=32)
    tensors, globals_ = {}, {}
    for expert in range(2):
        for proj_idx, proj in enumerate(("w1", "w2", "w3"), 1):
            name = f"layers.0.ffn.experts.{expert}.{proj}"
            tensors[name + ".weight"] = torch.full((32, 16), expert * 16 + proj_idx, dtype=torch.uint8)
            tensors[name + ".weight_scale"] = torch.full((32, 2), proj_idx, dtype=torch.float32).to(torch.float8_e4m3fn)
            globals_[name + ".weight_scale_2"] = torch.tensor((expert + 1) * proj_idx / 8)
            if request.param:
                globals_[name + ".input_scale"] = torch.tensor(99.0)
    globals_["mtp.0.ffn.experts.0.w1.weight"] = torch.ones(1, 1, dtype=torch.uint8)
    globals_["layers.0.engram.embed.weight"] = (torch.ones(1, 32).to(torch.float8_e4m3fn) if request.param
                                               else torch.ones(1, 16, dtype=torch.uint8))
    globals_["layers.0.engram.embed.scale"] = torch.ones(1, 1).to(torch.float8_e8m0fnu)
    safetensors.torch.save_file(tensors, tmp_path / "bulk.safetensors")
    safetensors.torch.save_file(globals_, tmp_path / "global.safetensors")
    index = {key: "bulk.safetensors" for key in tensors} | {key: "global.safetensors" for key in globals_}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return tmp_path, config


def test_validate_and_load_split_global_scales(tiny_shards):
    folder, config = tiny_shards
    completed = []
    banks = weight.load_nvfp4_expert_sources(str(folder), config,
                                           layer_sink=lambda layer, banks: completed.append(layer))
    assert completed == [0]
    for expert in range(2):
        assert torch.all(banks["gate_up_packed"][0][expert, :32] == expert * 16 + 1)
        assert torch.all(banks["gate_up_packed"][0][expert, 32:] == expert * 16 + 3)
        assert torch.all(banks["down_packed"][0][expert] == expert * 16 + 2)
        assert torch.all(banks["gate_up_scale"][0][expert, :32].float() == 1)
        assert torch.all(banks["gate_up_scale"][0][expert, 32:].float() == 3)
        assert torch.all(banks["down_scale"][0][expert].float() == 2)
        assert torch.all(banks["gate_up_global"][0][expert, :32] == (expert + 1) / 8)
        assert torch.all(banks["gate_up_global"][0][expert, 32:] == (expert + 1) * 3 / 8)
        assert torch.all(banks["down_global"][0][expert] == (expert + 1) / 4)


def test_serial_parallel_bank_bytes_match(tiny_shards):
    folder, config = tiny_shards
    sink = lambda layer, banks: None
    serial = weight.load_nvfp4_expert_sources(str(folder), config, layer_sink=sink)
    parallel = weight.load_nvfp4_expert_sources_parallel(str(folder), config, workers=2,
                                                        chunk=4096, layer_sink=sink)
    for name in serial:
        assert torch.equal(serial[name][0].view(torch.uint8), parallel[name][0].view(torch.uint8)), name


@pytest.mark.parametrize("mutation,match", [
    ("missing", "Missing NVFP4 tensor"), ("dtype", "Malformed NVFP4 tensor"),
    ("shape", "Malformed NVFP4 tensor"), ("expert", "outside configured backbone"),
])
def test_bad_expert_headers_fail_before_allocation(tiny_shards, mutation, match):
    folder, config = tiny_shards
    headers = weight.read_checkpoint_headers(folder)
    key = "layers.0.ffn.experts.0.w1.weight_scale"
    if mutation == "missing":
        del headers[key]
    elif mutation == "dtype":
        headers[key]["dtype"] = "F8_E8M0"
    elif mutation == "shape":
        headers[key]["shape"] = [32, 1]
    else:
        headers[key.replace("experts.0", "experts.2")] = headers.pop(key)
    with pytest.raises(ValueError, match=match):
        weight.validate_expert_headers(headers, config)


def test_resident_loader_preserves_native_keys_and_excludes_external_tables(tiny_shards):
    folder, _ = tiny_shards
    config = {"n_layers": 1, "compress_ratios": [0], "kv_source_layers": [],
              "index_source_layers": [], "candidate_source_layer": -1,
              "engram_layer_ids": [], "engram_num_embeddings": []}
    (folder / "config.json").write_text(json.dumps(config))
    tensors = {"head.weight": torch.arange(1024).view(32, 32).to(torch.bfloat16),
               "layers.0.attn.wo_a.weight": torch.ones(32, 32).to(torch.float8_e4m3fn),
               "layers.0.attn.wo_a.scale": torch.tensor([[128]], dtype=torch.uint8).view(torch.float8_e8m0fnu)}
    safetensors.torch.save_file(tensors, folder / "resident.safetensors")
    index = weight._weight_map(folder)
    index.update({name: "resident.safetensors" for name in tensors})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    resident = dict(weight.iter_weights(str(folder), "cpu", include_moe_experts=False))
    assert set(resident) == {"head.weight", "layers.0.attn.wo_a"}
    assert torch.all(resident["layers.0.attn.wo_a"] == 2)
    assert torch.equal(resident["head.weight"], tensors["head.weight"])
