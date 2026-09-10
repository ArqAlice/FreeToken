"""Sweep fused expert-copy launch geometry with actual Qwen3.8 NVFP4 bank sizes."""

import argparse
import json
from pathlib import Path

import torch
from triton.testing import do_bench_cudagraph

from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
from freetoken.kernel.pinned import device_ptr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sizes = [1638400, 204800, 2560, 819200, 102400, 5120]
    torch.manual_seed(42)
    sources = [torch.randint(0, 256, (40, size), dtype=torch.uint8, pin_memory=True)
               for size in sizes]
    destinations = [torch.empty_like(t, device='cuda') for t in sources]
    src_ptrs = torch.tensor([device_ptr(t) for t in sources], device='cuda')
    dst_ptrs = torch.tensor([t.data_ptr() for t in destinations], device='cuda')
    feats = torch.tensor(sizes, device='cuda')
    src_ids = torch.randperm(40, device='cuda').to(torch.int32)
    dst_ids = torch.randperm(40, device='cuda').to(torch.int32)
    count = torch.zeros(1, dtype=torch.int64, device='cuda')
    configs = [(1024, 8), (1024, 4), (512, 8), (256, 16), (512, 16), (1024, 16)]
    rows = []
    for n in (0, 1, 4, 8, 16, 24, 40):
        count.fill_(n)
        for threads, blocks in configs:
            def run():
                fast_index_copy_multi_jit(dst_ptrs, src_ptrs, feats, dst_ids, src_ids, count,
                                          num_threads=threads, blocks_per_bank=blocks)
            for t in destinations:
                t.fill_(17)
            run()
            torch.cuda.synchronize()
            for source, target in zip(sources, destinations):
                expected = torch.full_like(source, 17)
                expected[dst_ids[:n].cpu().long()] = source[src_ids[:n].cpu().long()]
                assert torch.equal(target.cpu(), expected)
            samples = [do_bench_cudagraph(run, rep=100) for _ in range(3)]
            row = dict(misses=n, threads=threads, blocks_per_bank=blocks,
                       samples_ms=samples, equal=True, bytes=n * sum(sizes))
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.output).write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
