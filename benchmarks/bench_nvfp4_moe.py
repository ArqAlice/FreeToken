"""Compare exact arithmetic E2M1 dequant with LUT gathers in decode MoE GEMVs."""

import argparse
import json
import os
from pathlib import Path

import torch
from triton.testing import do_bench_cudagraph

from freetoken.moe.fused_nvfp4 import _decode_gemm_marlin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare-layout", action="store_true")
    args = parser.parse_args()
    if args.compare_layout:
        import inspect
        from freetoken.kernel.triton.nvfp4_fused_moe import _decode_nvfp4_marlin_kernel
        if "N_MAJOR" not in inspect.signature(_decode_nvfp4_marlin_kernel.fn).parameters:
            parser.error("--compare-layout requires results/mtp-auto/nvfp4-layout-experiment.patch")
    torch.manual_seed(29)
    results = []
    shapes = ((2560, 1280, False), (640, 2560, True)) if args.compare_layout else (
        (2560, 1024, False), (512, 2560, True), (2560, 3072, False), (1536, 2560, True))
    for width, out, route_input in shapes:
        experts, top_k = (64, 10) if args.compare_layout else (16, 8)
        packed = torch.randint(0, 256, (experts, out, width // 2), device="cuda", dtype=torch.uint8)
        scale = (torch.rand(experts, out, width // 16, device="cuda") + .1).to(torch.float8_e4m3fn)
        glob = (torch.rand(experts, out, device="cuda") + .1).to(torch.float16)
        for count in (1, 2, 3, 4):
            x = torch.randn(count * top_k if route_input else count, width,
                            device="cuda", dtype=torch.bfloat16)
            weights = torch.rand(count, top_k, device="cuda")
            for routing in ("shared", "spread"):
                ids = (torch.arange(top_k, device="cuda", dtype=torch.int32).repeat(count, 1)
                       if routing == "shared" else
                       torch.arange(count * top_k, device="cuda", dtype=torch.int32)
                       .reshape(count, top_k).remainder(experts))
                expected = torch.empty(count, top_k, out, device="cuda", dtype=x.dtype)
                actual = torch.empty_like(expected)

                def run(output):
                    _decode_gemm_marlin(x, packed, scale, glob, output, weights, ids,
                                       route_input, route_input)

                control = "FREETOKEN_NVFP4_MOE_N_MAJOR" if args.compare_layout else "FREETOKEN_NVFP4_MOE_ARITHMETIC"
                os.environ[control] = "0"
                run(expected)
                lut_ms = do_bench_cudagraph(lambda: run(expected))
                os.environ[control] = "1"
                run(actual)
                arithmetic_ms = do_bench_cudagraph(lambda: run(actual))
                results.append(dict(shape=[count, top_k, width, out], routing=routing,
                    control=control, equal=torch.equal(expected, actual),
                    before_ms=lut_ms, after_ms=arithmetic_ms))
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    assert all(row["equal"] for row in results)


if __name__ == "__main__":
    main()
