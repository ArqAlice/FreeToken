# MTP expert fetch overlap

Run date: 2026-09-10 JST (2026-09-09 UTC). Hardware: RTX 5090 32 GB,
Threadripper PRO 5965WX, driver 591.86, CUDA 13, WSL2/Docker.
Checkpoint: `aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP` in Docker volume
`freetoken_hf-cache`.

## Change

Qwen4 MTP continuations compute routing, then start cache lookup, eviction and
expert fetch on a separate CUDA stream. The current layer's shared expert and
shared gate run on the compute stream in parallel. The routed GEMM starts only
after the fetch joins the compute stream, and retains its existing all-reduce.

The fetch waits for previous compute before evicting cache slots, so the previous
layer cannot read overwritten weights. The context manager joins on exceptional
exit as well, before temporary routing tensors can be recycled. CUDA graphs
capture both the fork and join. Cache rebuild already synchronizes the GPU and
releases old graphs; the copy stream holds no bank views across rebuilds.

The feature applies to target and draft continuation batches marked
`use_decode_moe`, on the GPU expert-offload path. Ordinary prefill/decode, resident
experts and CPU/hybrid experts keep their existing execution path. The target
math and speculative sampling rule are unchanged.

`FREETOKEN_MTP_EXPERT_OVERLAP` now defaults to **1**. Set it to **0** before process
startup to disable the feature for comparison. Existing MTP launch commands need
no additional option. Production compose configuration was not edited.

## Same-checkpoint A/B

The baseline is this working branch immediately before this change, copied to
`/tmp/mtp-overlap-before-python` in the development container. It includes the
previous state-copy/history improvement. It is not upstream `main`, which lacks
this branch's MTP/checkpoint support; these results isolate this implementation
step rather than comparing against upstream main.

Both processes use the strict BF16 path, 3073 expert slots, 15629 KV pages,
1000128 sequence limit, naive cache, and two request slots. The configured maximum
MTP depth is 3 even when the benchmark selects mode 1. Input is the checkpoint's
61-token chat template for `猫の魅力について熱くたくさん語って。`. Sampling uses
temperature 1, top-k 20 and top-p .95. Each mode warms up for 128 output tokens,
then generates 512 tokens five times with seeds 1000-1004 and alternating mode
order. First-to-last token delivery defines decode throughput, excluding TTFT.

[before.json](before.json), [after.json](after.json): median decode tokens/s,
with min-max across five runs.

| Mode | Before | Overlap enabled | Median change |
|---|---:|---:|---:|
| Off | 47.75 (46.19-48.40) | 47.47 (46.52-49.01) | -0.60% |
| Depth 1 | 37.02 (35.80-38.04) | 38.19 (36.67-38.93) | +3.14% |
| Depth 3 | 23.93 (23.15-24.98) | 24.51 (23.78-25.60) | +2.41% |

All 15 matched `(mode, seed)` output token sequences are exactly equal, as are
proposal/acceptance counts. CUDA allocated memory at measurement boundaries is
unchanged at 28574390784 bytes. History remains 456015876 bytes. This is a modest
improvement; neither MTP mode overtakes MTP off. It is an offline measurement,
without HTTP/OpenWebUI, and not a prediction for other hardware or contexts.

Inside `freetoken-mtp-dev`, with `/opt/venv/bin/python`:

```sh
PYTHONPATH=/tmp/mtp-overlap-before-python FREETOKEN_MTP_BATCHED_LINEAR=0 python \
  /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --modes 0,1,3 --tokens 512 --repeats 5 --output /tmp/mtp-overlap-before.json

PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_BATCHED_LINEAR=0 \
FREETOKEN_MTP_EXPERT_OVERLAP=1 python /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --modes 0,1,3 --tokens 512 --repeats 5 --profile \
  --profile-trace /tmp/mtp-overlap-trace.json --output /tmp/mtp-overlap-after.json
```

The explicit flag enabled overlap during A/B. The final default was switched to
1 after these results; regression tests below use that default without the flag.

## Observed GPU overlap

The separate 128-token profiled generation records 3413 expert gather kernel
calls, totaling 2.625 s of device intervals. Their interval union overlaps other
CUDA kernels for 0.702 s, including shared-expert BF16 GEMV, activation and shared
gate kernels. Multiple stream IDs appear because CUDA graph replay uses internal
execution streams.

[trace-summary.json](trace-summary.json) contains the overlap breakdown;
[trace.json.gz](trace.json.gz) contains the compressed Chrome trace. Overlap is
computed by intersecting the union of `fast_index_copy_multi` kernel intervals
with the union of other kernel intervals. Memcpy events are excluded. The
profile includes instrumentation overhead and a different continuation from the
primary samples; 0.702 s is not a claimed end-to-end saving. Primary throughput
comes only from the uninstrumented five-run comparison above.

## Validation

Related tests with the final default: **187 passed** in 9.62 s, one existing
FlashInfer deprecation warning. [tests-default.log](tests-default.log).

Coverage includes dynamic routes, hits and evictions across layers, CUDA graph
replay, waiting on exceptional exit, CPU/hybrid fallback, shared-input reads
before routed in-place writes, existing MTP state handling and cache rebuild.

```sh
PYTHONPATH=/opt/freetoken/python python -m pytest -p no:cacheprovider -q \
  tests/moe/test_offload.py tests/models/qwen4_exp/test_skeleton.py \
  tests/scheduler/test_mtp.py tests/scheduler/test_cache_rebuild.py
```

[validation.json](validation.json) and [validation.log](validation.log) record a
successful real-model regression with overlap enabled by default, depth 3,
8192 sequence limit, 132 KV pages and 6000 expert slots. This separate geometry
checks correctness; its timings are not part of the A/B table.

- Both greedy prompts produce the same 256 output tokens as MTP off.
- Forced rejection accepts 0 of 759 proposals and preserves exact output parity.
- Chunked prefill and two concurrent requests preserve exact output parity.
- Sampling at `(temperature, top_k, top_p)` of `(0.7, -1, 1.0)`, `(0.8, 20, 0.9)`
  and `(1.0, 1, 0.8)` completes two 64-token outputs each. These are execution
  checks, not a statistical proof of distribution equality.

```sh
PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_BATCHED_LINEAR=0 python \
  /opt/freetoken/benchmarks/bench_mtp.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --speculative-tokens 3 --tokens 256 --warmup-tokens 128 --repeats 1 \
  --validate --sampling-smoke --sampling-tokens 64 \
  --output /tmp/mtp-overlap-validation.json
```

No new HTTP/OpenWebUI, tensor-parallel, or long-context performance evaluation
was run. The isolated runtime diff is in [runtime-change.patch](runtime-change.patch),
with baseline/final source hashes in [source.json](source.json). Existing unrelated
working-tree changes are outside that patch. There was no commit or production
configuration change.
