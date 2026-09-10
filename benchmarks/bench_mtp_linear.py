"""Compare MTP multi-row linears with sequential single-row reductions on CUDA."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

from freetoken.kernel.triton.nvfp4_linear import (
    _gemv, nvfp4_dense_linear_t, nvfp4_transpose_resident,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nvfp4-only", action="store_true")
    parser.add_argument("--bf16-streams-only", action="store_true")
    parser.add_argument("--row-stream-count", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    torch.manual_seed(31)
    results = []
    if args.bf16_streams_only:
        from freetoken.layers.linear import rowwise_linear
        streams = [torch.cuda.Stream() for _ in range(args.row_stream_count)]
        for width, out in ((2560, 336), (10240, 336), (320, 10240),
                           (2560, 10240), (10240, 2560), (2560, 16480),
                           (6144, 2560), (2560, 1280), (640, 2560),
                           (2560, 13312), (2560, 2560), (2560, 512)):
            weight = torch.randn(out, width, device='cuda', dtype=torch.bfloat16)
            for count in (2, 3, 4):
                x = torch.randn(count, width, device='cuda', dtype=torch.bfloat16)

                def sequential():
                    return rowwise_linear(x, weight)

                def parallel():
                    y = x.new_empty((count, out))
                    compute = torch.cuda.current_stream()
                    for stream in streams[:count]:
                        stream.wait_stream(compute)
                    for i in range(count):
                        stream = streams[i % len(streams)]
                        with torch.cuda.stream(stream):
                            torch.mm(x[i:i + 1], weight.t(), out=y[i:i + 1])
                    for stream in streams[:count]:
                        compute.wait_stream(stream)
                    return y

                expected, actual = sequential(), parallel()
                equal = torch.equal(expected, actual)
                row = dict(shape=[count, width, out], streams=args.row_stream_count, equal=equal,
                           sequential_ms=do_bench_cudagraph(sequential),
                           parallel_ms=do_bench_cudagraph(parallel))
                results.append(row)
                print(json.dumps(row), flush=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        assert all(row['equal'] for row in results)
        return
    for width, out in ((2560, 512), (2560, 10240), (2560, 248320)):
        weight = torch.randint(0, 256, (out, width // 2), device="cuda", dtype=torch.uint8)
        scale = (torch.rand(out, width // 16, device="cuda") + .1).to(torch.float8_e4m3fn)
        global_scale = torch.full((out,), .1, device="cuda", dtype=torch.float16)
        weight, scale = nvfp4_transpose_resident(weight, scale)
        for count in (2, 3, 4, 5):
            x = torch.randn(count, width, device="cuda", dtype=torch.bfloat16)

            def sequential():
                return torch.cat([nvfp4_dense_linear_t(row, weight, scale, global_scale)
                                  for row in x.split(1)])

            def batched():
                return _gemv(x, weight.t(), scale.t(), global_scale, x.dtype, True)

            def shared():
                return _gemv(x, weight.t(), scale.t(), global_scale, x.dtype, True,
                             shared_rows=True)

            expected = sequential()
            shared_output = shared()
            equal = torch.equal(expected, batched()) and torch.equal(expected, shared_output)
            results.append(dict(kind="nvfp4", shape=[count, width, out], equal=equal,
                shared_mismatches=int(torch.count_nonzero(expected != shared_output)),
                sequential_ms=do_bench_cudagraph(sequential), batched_ms=do_bench_cudagraph(batched),
                shared_ms=do_bench_cudagraph(shared)))
    bf16_shapes = () if args.nvfp4_only else ((2560, 336), (10240, 336), (320, 10240), (2560, 10240))
    for width, out in bf16_shapes:
        weight = torch.randn(out, width, device="cuda", dtype=torch.bfloat16)
        for count in (2, 5):
            x = torch.randn(count, width, device="cuda", dtype=torch.bfloat16)

            def sequential():
                return torch.cat([F.linear(row, weight) for row in x.split(1)])

            def batched():
                return torch.bmm(x.unsqueeze(1), weight.t().expand(count, -1, -1)).squeeze(1)

            results.append(dict(kind="bf16_bmm_candidate", shape=[count, width, out],
                equal=torch.equal(sequential(), batched()),
                sequential_ms=do_bench_cudagraph(sequential), batched_ms=do_bench_cudagraph(batched)))
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    assert all(result["equal"] for result in results if result["kind"] == "nvfp4")


if __name__ == "__main__":
    main()
