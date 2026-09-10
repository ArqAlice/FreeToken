# MTP performance follow-up

## Implementation

- Target replay and intermediate prompt chunks skip the final residual mixer and
  LM head when only state is needed. Intermediate draft chunks also skip logits,
  and requests at their output limit no longer generate another draft.
- Draft continuations use CUDA Graphs. Each replay stages tokens, target/head
  residuals, request slots and attention metadata. Graphs are released alongside
  target graphs before runtime cache rebuilding. `FREETOKEN_MTP_DRAFT_GRAPH=0`
  selects eager draft execution independently of the target graph switch.
- QSA's derived write plan is rebuilt inside capture. The MTP head starts after
  QSA slot zero, so retaining a warmup plan would capture old position buffers.
  A same-state head comparison and a replay/slot-switch regression protect this.
- Sampling tensors are reused per request, with parameter changes invalidating
  the cache. MTP metadata preparation omits redundant sampling preparation.
  Rejection constructs a residual distribution for the first rejected row only.
- Disk PLE stages the current token block and two history tokens, avoiding a
  full-prefix copy. The small token readback still synchronizes with the GPU.
- NVFP4 verification rows share one kernel launch, with independent rows in the
  launch grid. Split-K partials and arrival counters are isolated per input row.
  The single-row FP32 accumulation and reduction order is preserved. This does
  not yet share a weight load between input rows within a kernel program.

## Reproduction

Hardware: RTX 5090 (32607 MiB), NVIDIA driver 591.86, Threadripper PRO 5965WX
(24 cores / 48 threads), WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`.
Docker image: `freetokenfp8-local:cu130`; checkpoint:
`aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP`, cached in
`freetoken_hf-cache`.

The benchmark uses 6000 MoE expert slots, 132 KV pages, NVFP4 KV, naive cache,
no overlap, two request slots, an 8192-token sequence limit and 128-token prefill
chunks. These are benchmark settings, not the repository's current 1M-context
Compose settings.

```sh
docker exec freetoken-mtp-dev /opt/venv/bin/python \
  /opt/freetoken/benchmarks/bench_mtp.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --speculative-tokens 2 --warmup-tokens 256 --tokens 256 --repeats 3 \
  --validate --sampling-smoke --output /tmp/mtp-perf-final2-plain.json
```

Repeat with depths 1 and 4. Before-change runs used the same benchmark script
with `docker exec -e PYTHONPATH=/tmp/mtp-before-python ...`; that directory is a
copy of this branch's Python package from before the performance changes.
These are before/after comparisons on the user's MTP branch, not upstream-main
benchmarks. No upstream-main performance claim is made.

Each mode warms both prompts with 256 generated tokens, then measures three
256-token greedy generations per prompt with `ignore_eos=True`. Model loading
and warmup are excluded. Total time includes prompt processing and final
detokenization. `ttft_seconds` measures offline first-token delivery, including
tokenization and scheduling; it is not HTTP TTFT. `decode_seconds` is the time
between first and last token delivery. Phase fields were added after the
depth-1/2 before runs; depth-4 before and all final runs include them.

Primary speed comparisons do not use `--timings`. That switch records nested
host/CUDA intervals for diagnosis, adds overhead, and its intervals overlap.
Do not sum nested intervals or compare an instrumented run directly with a
plain run to claim a speedup.

The two prompts are `The capital of France is` and
`Write a Python function that computes Fibonacci numbers.` The results are
limited to these short contexts, this checkpoint and this offload/cache setup.
Long-context performance and tensor parallelism remain unmeasured.

## End-to-end results

Mean seconds per 256-token request over both prompts and three repeats (six
requests per depth). The linked JSON files contain individual samples and tokens.

| Draft tokens | Before | After | Time reduction | After TTFT | After decode |
|---|---:|---:|---:|---:|---:|
| 1 | [8.081](mtp-perf-before1-plain.json) | [7.661](mtp-perf-final1-plain.json) | 5.2% | 2.486 | 5.175 |
| 2 | [8.621](mtp-perf-before2-plain.json) | [8.032](mtp-perf-final2-plain.json) | 6.8% | 2.486 | 5.545 |
| 4 | [9.839](mtp-perf-before4-plain.json) | [9.078](mtp-perf-final4-plain.json) | 7.7% | 2.487 | 6.591 |

At depth 4, decode time fell from 7.352 to 6.591 seconds (10.4%), while TTFT
remained about 2.487 seconds. Prompt processing is not the source of this gain.
MTP-disabled generation in the final runs averaged 7.215-7.232 seconds. Thus this
change improves MTP, but MTP still loses to ordinary decoding in this workload.
Depth 1 remains the fastest tested MTP setting.

Greedy token sequences match for all 36 primary before/after pairs (18 MTP and
18 MTP-disabled). Draft accepted /
proposed counts also stay unchanged for each prompt: depth 1 = 124/130 and
120/135; depth 2 = 160/190 and 157/196; depth 4 = 188/264 and 183/284.

## Kernel experiment

`bench_mtp_linear.py --output /tmp/mtp-perf-linear.json` runs on synthetic
resident weights, independently of model timing. CUDA Graph times below are
microseconds; the sequential reference uses the single-row kernel plus a cat.
Every NVFP4 result is exactly equal. See [raw results](mtp-perf-linear.json).

| Rows | Input width | Output width | Sequential | Multi-row |
|---|---:|---:|---:|---:|
| 2 | 2560 | 512 | 5.17 | 2.49 |
| 5 | 2560 | 512 | 11.85 | 3.58 |
| 2 | 2560 | 10240 | 13.27 | 10.18 |
| 5 | 2560 | 10240 | 31.96 | 22.68 |
| 2 | 2560 | 248320 | 436.60 | 434.76 |
| 5 | 2560 | 248320 | 1094.85 | 1111.09 |

Small projections benefit; the large vocabulary projection does not improve in
this microbenchmark, and its five-row sample is 1.5% slower. A single launch
does not eliminate repeated weight traffic.

BF16 `torch.bmm` was faster in this experiment but failed exact equality in
three of eight geometries: `(rows, input, output)` = `(5, 10240, 336)`,
`(2, 2560, 10240)` and `(5, 2560, 10240)`. It was not adopted. The existing BF16
single-row reduction remains in production.

## Validation

All three final depths pass 256-token greedy comparisons for both ordinary
prompts, forced rejection, a 145-token prompt spanning prefill chunks, and
concurrent versus individual requests. The three non-greedy settings
`(temperature, top_k, top_p)` = `(0.7, -1, 1)`, `(0.8, 20, 0.9)` and
`(1, 1, 0.8)` each complete two concurrent 64-token requests at every depth.
These are execution smoke tests, not an empirical proof of distributional
equivalence or equal-seed output equality.

[Regression log](mtp-perf-tests.log): **395 passed, 4 skipped**, on the GPU
environment above. Coverage includes scheduler/cache rebuilding, speculative
sampling, BF16/NVFP4 linears, Qwen4 config/weights/GDN/PLE/QSA, quantized QSA pools
and kernels, and detokenization. Added cases cover skipped logits, finished
requests, sampling-cache invalidation, graph replay with changing slots and
positions, compact PLE context, and NVFP4 bias, tails, noncontiguous inputs,
single-pass/split-K and repeated graph replay.

The [post-fix same-state trace](mtp-perf-fixed-diagnostic.json) compares eager
and graph head execution for 20 calls with changing positions and 1-3 input rows:
both logits and residuals are exactly equal in every call. The
[earlier failing trace](mtp-perf-diagnostic2-plain.json) predates the QSA plan
capture fix and is retained as diagnostic evidence, not a passing result.
To reproduce the final diagnostic, add `--timings --draft-diagnostic` to the
depth-2 command above and use `--repeats 1`.

Exact regression command inside the container:

```sh
/opt/venv/bin/python -m pytest -p no:cacheprovider -q \
  tests/scheduler tests/engine/test_speculative.py tests/engine/test_kv_quant_config.py \
  tests/layers/test_linear_mtp.py tests/models/qwen4_exp/test_config.py \
  tests/models/qwen4_exp/test_gdn.py tests/models/qwen4_exp/test_ple.py \
  tests/models/qwen4_exp/test_ple_disk.py tests/models/qwen4_exp/test_skeleton.py \
  tests/models/qwen4_exp/test_weight.py tests/models/qwen4_exp/test_qsa_backend.py \
  tests/models/qwen4_exp/test_qsa_kernels.py tests/kvcache/test_qsa_pool.py \
  tests/kvcache/test_qsa_pool_fp8.py tests/kernels/test_qsa_fp8.py \
  tests/kernels/test_qsa_nvfp4.py tests/tokenizer/test_detokenize.py
```

## Remaining cost

In the final instrumented depth-2 run, target verification/replay occupies 9.86
of the 10.99 seconds recorded around decode `_one` calls (two requests).
These are nested stream intervals, including transfers and waits, not exclusive
kernel times. Target work remains the dominant cost. Larger gains require
reducing repeated target weight reads across input rows or avoiding accepted-prefix
replay after rejection while preserving recurrent/QSA/PLE state. The current
row grid does not provide that weight reuse, and BF16 batched GEMM cannot simply
replace the exact single-row path.
