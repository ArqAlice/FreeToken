# MTP performance plan: first implementation tranche

Run date: 2026-09-10 JST (2026-09-09 UTC).

This tranche implements reduced state snapshots, history capacity based on the
initial maximum draft depth, and an instrumented benchmark for cycle timing and
expert cache traffic. Greedy proposals, draft QSA reuse, expert transfer overlap
and adaptive depth remain subsequent work.

Hardware: RTX 5090 32 GB, Threadripper PRO 5965WX, WSL2/Docker, CUDA 13,
NVIDIA driver 591.86. Checkpoint:
`aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP`, stored in Docker volume
`freetoken_hf-cache`.

The baseline is a copy of this working branch's Python source taken before this
tranche, at `/tmp/mtp-plan-before-python` in `freetoken-mtp-dev`. It is not upstream
`main`, which does not contain this branch's MTP implementation and checkpoint
support. The comparison isolates this tranche; it does not establish a speedup
against upstream main.

Both timing processes load the same checkpoint, retain the strict BF16 path, and
use the same 3073 expert slots, 15629 KV pages, 1000128 sequence limit, naive cache,
and maximum two request slots. MTP-off runs share the loaded model and memory
reservation with MTP-on runs. The configured maximum depth is 3 in both processes,
including when the benchmark temporarily selects depth 1.

The prompt is `猫の魅力について熱くたくさん語って。`, formatted with the checkpoint's
chat template (61 input tokens). Sampling uses temperature 1, top-k 20, top-p .95.
Each mode warms up for 128 output tokens, then generates 512 tokens five times;
mode order alternates. Seeds are 1000 through 1004. Decode throughput uses the
first-to-last token delivery interval and excludes TTFT. These are offline
measurements, without HTTP or OpenWebUI.

## Commands

Inside the development container, using `/opt/venv/bin/python`:

```sh
PYTHONPATH=/tmp/mtp-plan-before-python FREETOKEN_MTP_BATCHED_LINEAR=0 python \
  /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --modes 0,1,3 --tokens 512 --repeats 5 --output /tmp/mtp-plan-before.json

PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_BATCHED_LINEAR=0 python \
  /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --modes 0,1,3 --tokens 512 --repeats 5 --output /tmp/mtp-plan-after.json
```

## Results

Uninstrumented results: [before.json](before.json), [after.json](after.json).
Five runs per cell; values are median decode tokens/s, with min-max in parentheses.

| Mode | Before | After | Median change |
|---|---:|---:|---:|
| Off | 47.74 (46.24-48.90) | 47.87 (46.55-49.00) | +0.28% |
| Depth 1 | 37.29 (35.83-37.99) | 37.60 (35.94-37.95) | +0.84% |
| Depth 3 | 23.81 (23.07-24.76) | 23.92 (23.21-25.01) | +0.46% |

These differences are small relative to run-to-run variation. This is not evidence
of a substantial throughput improvement. Both MTP modes remain slower than MTP
off. Median TTFT changes from about 2.51-2.52 s to 2.50 s.

History allocations, including convolution and index buffers, decrease from
570019844 to 456015876 bytes (543.61 to 434.89 MiB). CUDA allocated memory at the
measurement boundaries decreases by the same 108.72 MiB. The depth-1 runs in this
table retain the depth-3 maximum's history capacity; a process initially configured
only for depth 1 reserves two history steps instead of four.

All 15 matched `(mode, seed)` cases have exactly equal generated token IDs before
and after. Acceptance counts match as well. This checks this implementation change;
equal seeds are not expected to give equal outputs between MTP-on and MTP-off.

An initial after-run was discarded: lazy runner construction under the first
positive benchmark mode (1) reserved only two history steps, so mode 3 correctly
fell back to replay. The harness now constructs the runner under the configured
maximum before switching modes. The baseline implementation always allocated five
steps and was unaffected by this initialization detail.

Related GPU/unit tests: **175 passed, 4 skipped** in 15.39 s, with one existing
FlashInfer deprecation warning. [Full test output](tests.log).

```sh
PYTHONPATH=/opt/freetoken/python python -m pytest -p no:cacheprovider -q \
  tests/scheduler/test_mtp.py tests/models/qwen4_exp/test_gdn.py \
  tests/models/qwen4_exp/test_ple.py tests/scheduler/test_cache_rebuild.py
```

## Instrumented diagnosis

[diagnostics.json](diagnostics.json) contains a separate 128-token generation per
depth, seed 20260910, with expert statistics enabled before graph capture. Both
depths reuse the four-step maximum-depth allocation. All decode cycles in these
two diagnostic runs used existing graphs; prefill is separated.

| Steady decode metric | Depth 1 | Depth 3 |
|---|---:|---:|
| Cycles / emitted tokens after prefill | 75 / 127 | 52 / 127 |
| `_one` CUDA stream interval | 3.697 s | 4.489 s |
| Target CUDA stream interval | 3.348 s | 3.991 s |
| Draft repair + next draft interval | 0.177 s | 0.114 s |
| Additional lookahead interval | 0 | 0.236 s |
| Snapshot + restore + commit intervals | 0.024 s | 0.028 s |
| Target unique expert visits | 59716 | 64958 |
| Target missing expert visits | 18055 | 21634 |
| Estimated target packed expert fetch | 50.06 GB | 59.98 GB |

Stream intervals include idle time and nested operations overlap: do not sum the
table into an exclusive kernel breakdown. These instrumented runs are not the
throughput A/B above. Their short prompt continuation and seed also have different
acceptance rates from the 512-token primary measurements.

The target accounts for about 89-91% of cycle stream time. Snapshot/restore/commit
account for less than 1%. The packed expert size is 2772480 bytes; estimated fetch
bytes multiply per-layer LRU misses by actual packed bank sizes. This is not a
hardware PCIe counter. The library's ACTIVE statistic counts distinct queried IDs
per invocation; existing routing already deduplicates repeated expert IDs.

A separate depth-3 PyTorch profile in the same artifact records 2.906 s in the
`fast_index_copy_multi` expert gather kernel and about 1.499 s across the two main
BF16 GEMV entries. The 2.456 s pinned HtoD transfer entry includes full-layer
prefill transfer and must not be added to decode transfer cost. This profile is a
different generated continuation from the phase-timing runs.

```sh
PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_BATCHED_LINEAR=0 python \
  /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --modes 1,3 --tokens 128 --repeats 0 --diagnostics --profile \
  --output /tmp/mtp-plan-diagnostics.json
```

## Existing FlashInfer backend trial

`--nvfp4-backend flashinfer` selects `b12x` and finishes repacking weights, but fails
during engine graph warmup before any timed generation. No throughput comparison
or quality result is available for this backend. [Full failure log](flashinfer.log).

```text
ValueError: force_tile_config fc2 tile (tile_k=32, tile_n=512) does not fit
problem N/K=2560/640 at moe_block_size=8
```

The actual model config has hidden size 2560, MoE intermediate size 640, 512
experts and top-10 routing. FreeToken passes those dimensions correctly. The
installed FlashInfer tile validator rejects tile K below 64, including the selected
K=32 candidate. Fixing that backend compatibility issue remains separate work;
this tranche does not change third-party libraries or the default backend.

```sh
PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_BATCHED_LINEAR=0 python \
  /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --nvfp4-backend flashinfer --modes 0,1,3 --tokens 512 --repeats 2 \
  --output /tmp/mtp-plan-flashinfer.json
```

## Next priority

Prioritize overlapping target expert fetch with the same layer's shared expert
computation. Routing already precedes shared-expert computation in
`Qwen4ExpMoE.forward`; the current cache ensure/copy happens after it. Separating
expert preparation from the routed GEMM provides a concrete overlap boundary.
Measure copy-stream contention and graph correctness before enabling it.

Greedy proposals and draft QSA reuse remain useful secondary experiments, but
eliminating all measured draft/lookahead time would still leave the current target
verification cost. Preserve target sampling and measure actual MTP-on throughput.
Adaptive depth remains a separate way to avoid slowdowns, not a measured
acceleration of the underlying speculative cycle.

## Real-model regression

[validation.json](validation.json) and [validation.log](validation.log) record a
successful depth-3 run with four history steps. This harness uses its separate
8192 sequence limit, 132 pages and 6000 expert slots; its timings are not part of
the server-capacity A/B above.

- Greedy outputs equal MTP off for both prompts, 256 tokens each.
- Forced rejection accepts 0 of 759 proposals and retains exact output parity.
- Chunked prefill and two concurrent requests retain exact output parity.
- Sampling smoke tests complete two 64-token outputs each at `(temperature,
  top_k, top_p)` of `(0.7, -1, 1.0)`, `(0.8, 20, 0.9)`, and `(1.0, 1, 0.8)`.
  These are execution checks, not a statistical proof of distribution equality.

```sh
PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_BATCHED_LINEAR=0 python \
  /opt/freetoken/benchmarks/bench_mtp.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --speculative-tokens 3 --tokens 256 --warmup-tokens 128 --repeats 1 \
  --validate --sampling-smoke --sampling-tokens 64 \
  --output /tmp/mtp-plan-validation.json
```

GPU recurrence tests also cover exact-size history buffers of 2, 3, 4 and 5
steps, padded buffers, untouched slots, and graph replay. Scheduler tests cover
compact/full snapshots, QSA ring boundaries, every accepted prefix, runtime depth
changes and fallback. No new HTTP/OpenWebUI, long-context performance, or broad
quality evaluation was run in this tranche.

The isolated runtime change is saved in [runtime-change.patch](runtime-change.patch),
with compared source hashes in [source.json](source.json). Existing unrelated
working-tree changes are outside this patch. Production configuration was not
modified.
