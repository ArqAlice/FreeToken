"""Measure fixed-reduction shared-weight projection and numerical error."""
import json
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from freetoken.layers.linear import rowwise_linear
from freetoken.kernel.triton.bf16_shared_linear import bf16_shared_linear

torch.manual_seed(935)
w = torch.randn(16480, 2560, device='cuda', dtype=torch.bfloat16)
results = []
for m in (1, 2, 3, 4):
    x = torch.randn(m, 2560, device='cuda', dtype=torch.bfloat16)
    expected = rowwise_linear(x, w)
    accurate = x.float() @ w.float().t()
    baseline_ms = do_bench_cudagraph(lambda: rowwise_linear(x, w))
    for splits, bn in ((4, 32), (4, 64), (8, 32), (8, 64), (8, 128), (16, 32), (16, 64)):
        def run(v=x):
            return bf16_shared_linear(v, w, splits=splits, block_n=bn)
        actual = run()
        singles = torch.cat([run(row) for row in x.split(1)])
        row = dict(rows=m, splits=splits, block_n=bn,
                   single_equal=torch.equal(actual, singles),
                   cublas_mismatches=int((actual != expected).sum()),
                   relative_rms=float((actual.float()-accurate).square().mean().sqrt()/accurate.square().mean().sqrt()),
                   baseline_ms=baseline_ms, ms=do_bench_cudagraph(run))
        results.append(row)
        print(json.dumps(row), flush=True)
Path('/tmp/mtp-reuse-shared-probe.json').write_text(json.dumps(results, indent=2))
assert all(r['single_equal'] for r in results)
