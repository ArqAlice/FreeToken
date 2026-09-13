from types import SimpleNamespace
import base64
import io

import pytest
import torch
from PIL import Image

from freetoken.models.deepseek_v41.image_processor import (
    IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START, image_token_types,
    load_image, load_image_bytes, num_image_tokens, plan_image_grid,
)
from freetoken.models.deepseek_v41.vision import Aligner, ViT, merge_image_embeddings


def _args(**overrides):
    values = dict(vision_patch_size=2, vision_dim=8, vision_n_heads=2,
                  vision_inter_dim=12, vision_n_layers=2, vision_rope_theta=10000.0,
                  vision_downsample_ratio=2, dim=6, vision_min_pixels=16,
                  vision_max_n_token=24, vision_max_wh_ratio=None)
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("width,height", [(1, 10000), (10000, 1), (1, 1), (37, 53), (2000, 3000)])
def test_image_resize_respects_token_budget(width, height):
    args = _args()
    h, w, pixels_h, pixels_w = plan_image_grid(width, height, args)
    assert h > 0 and w > 0
    assert pixels_h % args.vision_patch_size == pixels_w % args.vision_patch_size == 0
    assert num_image_tokens(h, w) <= args.vision_max_n_token


def test_image_patches_normalization_and_layout():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 4), (255, 0, 0)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    patches, nh, nw, lh, lw = load_image({"url": "data:image/png;base64," + encoded}, _args())
    assert (nh, nw, lh, lw) == (2, 4, 1, 2)
    assert patches.shape == (8, 3, 2, 2)
    assert patches.dtype == torch.float32
    torch.testing.assert_close(patches[:, 0], torch.ones_like(patches[:, 0]))
    torch.testing.assert_close(patches[:, 1:], -torch.ones_like(patches[:, 1:]))
    assert image_token_types(lh, lw).tolist() == [IMAGE_START, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE_END]


@pytest.mark.parametrize("url", ["/etc/passwd", "file:///etc/passwd", "http://127.0.0.1/test", "http://[::1]/test"])
def test_api_image_loader_rejects_local_sources(url):
    with pytest.raises(ValueError):
        load_image_bytes({"url": url})


def test_aligner_channel_and_pixel_order():
    torch.manual_seed(12)
    args = _args()
    aligner = Aligner(args).float()
    values = torch.randn(3, 5, args.vision_dim)
    rows = []
    for h in range(0, 3, 2):
        for w in range(0, 5, 2):
            block = torch.zeros(args.vision_dim, 2, 2)
            crop = values[h:h + 2, w:w + 2].permute(2, 0, 1)
            block[:, :crop.shape[1], :crop.shape[2]] = crop
            rows.append(block.flatten())
    reference = aligner.w2(torch.nn.functional.gelu(aligner.w1(torch.stack(rows))))
    torch.testing.assert_close(aligner(values.reshape(-1, args.vision_dim), 3, 5), reference)


def test_vit_attention_matches_explicit_bidirectional_reference():
    torch.manual_seed(20)
    args = _args()
    model = ViT(args).float()
    patches = torch.randn(6, 3, 2, 2)
    from freetoken.models.deepseek_v41.vision import apply_rotary, get_vision_cos_sin

    x = model.patch_embed(patches)
    cos, sin = get_vision_cos_sin(2, 3, model.rope_dim, model.rope_theta)
    for block in model.blocks:
        q, k, v = [t.reshape(6, args.vision_n_heads, -1).transpose(0, 1)
                   for t in block.attn.wqkv(block.norm1(x)).chunk(3, -1)]
        q = apply_rotary(q.transpose(0, 1), cos, sin).transpose(0, 1)
        k = apply_rotary(k.transpose(0, 1), cos, sin).transpose(0, 1)
        scores = q @ k.transpose(-1, -2) / block.attn.head_dim ** 0.5
        attention = (scores.softmax(-1) @ v).transpose(0, 1).reshape(6, -1)
        x = x + block.attn.wo(attention)
        x = x + block.mlp(block.norm2(x))
    torch.testing.assert_close(model(patches, 2, 3), model.norm(x), atol=1e-6, rtol=1e-5)


def test_image_embedding_scatter_across_every_chunk_boundary():
    class Tower:
        def __init__(self):
            self.calls = 0
            self.vision = SimpleNamespace(patch_embed=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(1))))
            self.image_start = torch.full((3,), 10.0)
            self.image_newline = torch.full((3,), 20.0)
            self.image_end = torch.full((3,), 30.0)

        def encode_image(self, patches, nh, nw):
            self.calls += 1
            return torch.arange(12, dtype=torch.float32).reshape(4, 3)

    types = image_token_types(2, 2)
    span = torch.stack([torch.full((3,), 10.0), *torch.arange(6.).reshape(2, 3),
                        torch.full((3,), 20.0), *torch.arange(6., 12.).reshape(2, 3),
                        torch.full((3,), 20.0), torch.full((3,), 30.0)])
    expected = torch.cat((torch.zeros(2, 3), span, torch.zeros(2, 3)))
    for boundary in range(1, expected.shape[0]):
        model = Tower()
        media = [dict(start=2, patches=torch.zeros(4, 3, 2, 2), n_vit_h=2, n_vit_w=2, types=types)]
        outputs, masks = [], []
        for start, stop in ((0, boundary), (boundary, expected.shape[0])):
            req = SimpleNamespace(cached_len=start, extend_len=stop-start, media=media)
            batch = SimpleNamespace(is_prefill=True, reqs=[req])
            h, mask = merge_image_embeddings(model, batch, torch.zeros(stop-start, 3))
            outputs.append(h)
            masks.append(mask)
        torch.testing.assert_close(torch.cat(outputs), expected)
        assert torch.cat(masks).tolist() == [False] * 2 + [True] * types.numel() + [False] * 2
        assert model.calls == 1
        assert media[0]["patches"] is None
        assert media[0]["types"] is types


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vit_bf16_cuda_matches_cpu():
    torch.manual_seed(31)
    cpu = ViT(_args()).float().eval()
    gpu = ViT(_args()).to(device="cuda", dtype=torch.bfloat16).eval()
    gpu.load_state_dict(cpu.state_dict())
    patches = torch.randn(6, 3, 2, 2)
    with torch.no_grad():
        expected = cpu(patches, 2, 3)
        actual = gpu(patches.to(device="cuda", dtype=torch.bfloat16), 2, 3)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0.04, atol=0.025)


def test_vision_parameters_preserve_checkpoint_dtypes():
    vision = ViT(_args())
    aligner = Aligner(_args())
    for name, parameter in vision.named_parameters():
        assert parameter.dtype == (torch.float32 if "norm" in name else torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in aligner.parameters())
