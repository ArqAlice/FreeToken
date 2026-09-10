"""Rotate a 338 MB weight working set to avoid a single warm projection cache."""
import json
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from freetoken.layers.linear import rowwise_linear
from freetoken.kernel.triton.bf16_shared_linear import bf16_shared_linear

torch.manual_seed(951)
weights = [torch.randn(16480, 2560, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
results = []
for m in (1, 2, 3, 4, 5):
    inputs = [torch.randn(m, 2560, device='cuda', dtype=torch.bfloat16) for _ in weights]
    def run(fn):
        return [fn(x, w) for x, w in zip(inputs, weights)]
    row = dict(rows=m, weight_working_set_bytes=sum(w.numel()*w.element_size() for w in weights),
               sequential_ms=[do_bench_cudagraph(lambda: run(rowwise_linear))/4 for _ in range(3)],
               shared_ms=[do_bench_cudagraph(lambda: run(bf16_shared_linear))/4 for _ in range(3)])
    results.append(row)
    print(json.dumps(row), flush=True)
Path('/tmp/mtp-reuse-cold-probe.json').write_text(json.dumps(results, indent=2))
