# MTP CUDA Graph performance follow-up

Historical one-token measurements. The later PLE rounding fix, stochastic sampling
and multi-token implementation are described in [the follow-up](sampling-multi.md).

Warmed 64-token MTP generation now takes about 4.0 seconds, down from 7.0-7.3 seconds
for the optimized eager path. Same-process A/B measurement shows about 44% less elapsed
time. Ordinary decoding on this branch remains faster at about 3.8 seconds.

## Changes

- Write single-row target GEMM results directly into the output buffer, eliminating
  concatenation kernels while retaining the single-row reduction order.
- Capture one- and two-token target continuations, including rejection replay, in
  CUDA graphs. Restage token IDs, positions, page mappings, sequence lengths and
  recurrent-state slots before every replay.
- Restore recurrent state after capture warmup, stage disk PLE rows into graph
  buffers, and release graphs before scheduler cache rebuilding or shutdown.
- Retain the fixed one- and two-token PLE convolution indices on the GPU so captured
  execution does not depend on a temporary pinned-host copy source.

Graphs are enabled by default when engine CUDA graphs are enabled. Set
`FREETOKEN_MTP_CUDA_GRAPH=0` for eager verification. No additional Compose command
arguments are required beyond `--mtp-speculative-tokens 1`. Graphs are captured lazily;
the first request pays capture and warmup costs, excluded from the warmed timings here.
Requests still execute serially and the draft head remains eager.

## Hardware and reproduction

RTX 5090, 32607 MiB, driver 591.86; Docker image `freetokenfp8-local:cu130`.
Checkpoint `aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP` in `freetoken_hf-cache`.
The same-process harness uses greedy sampling, 64 fixed output tokens (`ignore_eos=True`),
6000 MoE expert slots, 132 KV pages, NVFP4 KV, naive cache, disabled overlap, maximum
sequence length 8192, and prefill chunks of 128 tokens. Model loading and warmup are
excluded; prompt processing and generation are included.

```sh
docker exec -e FREETOKEN_MTP_CUDA_GRAPH=1 freetoken-mtp-dev \
  /opt/venv/bin/python /opt/freetoken/benchmarks/bench_mtp.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --timings --validate --compare-graphs --repeats 2 --tokens 64 \
  --output /tmp/mtp-graph-final.json
```

Both graph settings run in the same loaded process; their order is reversed for the
second repeat. This isolates the graph change. The non-MTP baseline is the current
branch with MTP disabled, not upstream main; these are not upstream-main A/B numbers.

## Measurements

[mtp-graph-final.json](mtp-graph-final.json) contains full token IDs, timings and checks.
Mean seconds across two repeats:

| Prompt | MTP disabled | MTP eager | MTP graph | Graph vs eager time reduction |
| --- | ---: | ---: | ---: | ---: |
| The capital of France is | 3.821 | 7.020 | 3.957 | 43.6% |
| Write a Python function that computes Fibonacci numbers. | 3.764 | 7.257 | 3.988 | 45.0% |

These correspond to about 16.2 and 16.0 output tokens/s including prefill with graphs,
versus about 16.7 and 17.0 tokens/s with MTP disabled. The graph A/B samples above are
uninstrumented; the report's primary runs additionally include method timing events.
All four primary comparisons and all eight graph A/B comparisons match baseline token
IDs exactly. Forced rejection (0 accepted of 62), a chunked 145-token prompt, and two
simultaneous requests versus separate MTP requests also pass.

Final-source confirmation after retaining PLE index buffers is in
[mtp-graph-release.json](mtp-graph-release.json): 3.979/3.977 seconds with MTP versus
3.839/3.753 seconds without it. Both 64-token outputs and all three additional checks
pass with default graph settings.

For four 64-token MTP requests, target decode intervals fall from 17.680 seconds
([mtp-rowout.json](mtp-rowout.json)) to 5.063 seconds. Their host dispatch time falls
from 17.461 to 0.202 seconds. CUDA event intervals include host launch gaps and stalls;
they are not a measurement of GPU busy time alone. Nested method timings must not be
summed. Snapshot/restore costs were small and were left intact.

Earlier intermediate results: [mtp-timing-before.json](mtp-timing-before.json),
[mtp-rowout.json](mtp-rowout.json), [mtp-graph.json](mtp-graph.json) (two-token graph only).

## Extended validation and remaining output difference

[mtp-graph-256.json](mtp-graph-256.json) extends generation to 256 tokens using default
graph settings. Both short prompts match baseline token IDs (7.210/7.280 seconds without
MTP, 8.124/8.252 seconds with MTP). Forced rejection (0 accepted of 254) and concurrent
versus individual MTP requests pass. The chunked long-prompt check fails: output index
99 (the 100th token) is 314 without MTP and 494 with MTP.

The same difference reproduces with graphs disabled, and graph/eager outputs match
each other exactly: [mtp-long-ab.json](mtp-long-ab.json). Restoring the previous
single-row `F.linear` plus concatenation implementation also yields exactly the same
MTP output: [mtp-long-legacy.json](mtp-long-legacy.json). This is a pre-existing MTP
parity limitation, not a regression from graph replay or removal of concatenation.
These extended benchmark commands intentionally exit nonzero on the known mismatch.

```sh
docker exec freetoken-mtp-dev /opt/venv/bin/python /opt/freetoken/benchmarks/bench_mtp.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --long-prompt --compare-graphs --compare-legacy-linear --repeats 1 --tokens 128 \
  --output /tmp/mtp-long.json
```

The changed subsystems' regression tests pass: **129 passed, 4 skipped**. This includes
graph state rollback and slot restaging, graph-disable controls, PLE continuation
replay, disk PLE graph staging, cache rebuild ordering and exact single-row GEMM tests.
Three optional HF-reference tests and the standalone real PLE table test are skipped
because their environment variables are unset; this is not a full-suite run.

```sh
docker exec freetoken-mtp-dev /opt/venv/bin/python -m pytest \
  tests/scheduler tests/layers/test_linear_mtp.py \
  tests/models/qwen4_exp/test_ple.py tests/models/qwen4_exp/test_ple_disk.py \
  tests/tokenizer/test_detokenize.py -q -p no:cacheprovider
```

## Limits

The measured improvements apply to this hardware, checkpoint, cache budget and short
prompt workload. They do not establish universal greedy equivalence or a throughput
gain over non-MTP decoding. Expert transfers, serial requests, eager draft work and
rejection replay remain performance costs. Stochastic speculative acceptance and
multi-token drafts remain outside this implementation.
