"""Screen output tiling with the checkpoint GDN projection geometry."""
import json
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from freetoken.layers.linear import rowwise_linear

torch.manual_seed(931)
w = torch.randn(16480, 2560, device='cuda', dtype=torch.bfloat16)
results = []
for count in (2, 3, 4):
    x = torch.randn(count, 2560, device='cuda', dtype=torch.bfloat16)
    expected = rowwise_linear(x, w)
    for tile in (0, 512, 1024, 2048, 4096, 8192):
        def run():
            if tile == 0:
                return rowwise_linear(x, w)
            y = x.new_empty((count, w.shape[0]))
            for start in range(0, w.shape[0], tile):
                end = min(start + tile, w.shape[0])
                for i in range(count):
                    torch.mm(x[i:i+1], w[start:end].t(), out=y[i:i+1, start:end])
            return y
        actual = run()
        row = dict(rows=count, tile=tile, equal=torch.equal(actual, expected),
                   mismatches=int((actual != expected).sum()),
                   max_error=float((actual.float()-expected.float()).abs().max()),
                   ms=[do_bench_cudagraph(run) for _ in range(3)])
        results.append(row)
        print(json.dumps(row), flush=True)
Path('/tmp/mtp-reuse-linear-probe.json').write_text(json.dumps(results, indent=2))
