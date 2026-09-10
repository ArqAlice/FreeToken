# Automatic MTP depth and asynchronous disk PLE

Test date: 2026-09-10 JST. Branch `feat/mtp-support`, base HEAD
`3381015664c3acc5003b04fcad9a29742429792c`, with the existing uncommitted MTP,
QSA and quantization changes retained. Hardware: RTX 5090 32 GB, Threadripper PRO
5965WX, driver 591.86, CUDA 13, Linux inside Docker/WSL2.

Checkpoint: `aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP`, snapshot
`18bd7d69491d4709ac1933e95343e15f045ea21d`, in Docker volume `freetoken_hf-cache`.
Tests use `/opt/venv/bin/python` and `PYTHONPATH=/opt/freetoken/python` in
`freetoken-mtp-dev`. The model argument below denotes this snapshot's local path.

The throughput reference is this branch with MTP disabled, not upstream `main`.
Upstream main was not benchmarked; these are local-fork measurements, not a claim
of compliance with upstream's main-versus-PR performance requirement. No commit,
push or PR was made. Compose and Dockerfile were not changed.

## Implementation

- `--mtp-auto` chooses 0/1/2/3, bounded by `--mtp-speculative-tokens`, on TP=1.
  Four completed measurements follow 16 initial ordinary cycles to avoid a cold
  expert-cache baseline. EMA elapsed time divided by EMA emitted tokens scores
  candidates. Capture cycles are recorded but excluded from policy updates.
- A depth-1 loss stops speculation for that request. Otherwise deeper candidates
  are explored. Active operation refreshes the ordinary baseline every 32 updates
  and recalibrates at 8192-token context boundaries. Profitable-depth changes use
  10% hysteresis; continuation requires a 3% estimated gain over ordinary decode.
  This conservative search can miss a profitable deeper depth after a depth-1 loss.
- Ordinary CUDA graphs optionally retain the target residual, so baseline decode
  uses the ordinary path while keeping draft KV current. Calibration omits draft
  vocabulary projection/sampling until a proposal is needed. After stopping, draft
  work disappears; all-zero sampling requests use normal batched decode. Greedy
  requests retain single-row arithmetic across depth changes. Other requests in
  a mixed batch can keep speculating. Completed-cycle decisions do not inspect and
  discard newly sampled proposals.
- Disk PLE verification now reads GPU draft IDs into pinned memory asynchronously,
  launches the target graph, then fills/signals disk rows before the graph consumes
  them. This extends the existing ordinary-decode wait-sync protocol to verification.
  Eager/gate-sync paths still stage synchronously. `FREETOKEN_MTP_ASYNC_PLE=0`
  restores the previous verification readback for comparison. Non-disk PLE avoids
  the unnecessary host copy entirely.

No cache capacity or numerical approximation was changed. BF16 batched linears,
deterministic drafts for stochastic targets, and draft QSA reuse remain off by
default.

## Actual-time automatic selection

[auto.json](auto.json) uses five repetitions per mode, 512 output tokens, a
128-token warmup and the same capacity/sampling settings described below.

| Mode | Median decode tok/s | Range |
|---|---:|---:|
| MTP off | 46.82 | 45.53-47.66 |
| Auto, ceiling 3 | 45.03 | 44.23-48.12 |

The median difference is -3.81%. All five measured requests tried depth 1 and
stopped at 0. Of 2555 decode tokens, 2525 (98.83%) used depth 0, including 2425
(94.91%) after speculation was permanently stopped. This is protection against
unprofitable speculation, not a claim that active MTP generates at 45 tok/s.
Each request made four proposals. Calibration's extra draft maintenance averaged
about 27 ms per request; graph captures occurred during warmup, not these runs.
The five-request median meets a 5% slowdown target in this offline workload;
it is not a bound on individual requests or on HTTP/UI performance.

```sh
python /opt/freetoken/benchmarks/bench_mtp_chat.py --model "$MODEL" \
  --modes 0,3 --mtp-auto --tokens 512 --warmup-tokens 128 --repeats 5 \
  --output /tmp/mtp-auto.json
```

## Fixed-depth optimization

[async-ple.json](async-ple.json) contains five paired repetitions per depth, with
alternating order and identical seeds within each pair. Chat prompt:
`猫の魅力について熱くたくさん語って。`, 61 input tokens, 256 output tokens,
temperature 1, top-k 20, top-p .95. Capacity stays at 1,000,128 tokens, 15,629 pages,
3,073 expert slots and two request slots, with naive cache and disk PLE wait-sync.

| Fixed depth | Synchronous PLE, median tok/s | Asynchronous PLE, median tok/s | Change |
|---|---:|---:|---:|
| 1 | 34.65 | 35.35 | +2.04% |
| 3 | 24.05 | 24.73 | +2.84% |

All ten paired output token sequences match exactly. This is a small improvement
to active MTP, not evidence that active MTP overtakes ordinary decode. The output
budget differs from the 512-token automatic-policy comparison; compare paired
columns here, not absolute rates across the two experiments.

```sh
python /opt/freetoken/benchmarks/bench_mtp_chat.py --model "$MODEL" \
  --modes 1,3 --compare-ple --tokens 256 --warmup-tokens 128 --repeats 5 \
  --output /tmp/mtp-async-ple.json
```

## Remaining bottleneck and rejected experiments

The separate instrumented [diagnostics.json](diagnostics.json), taken before this
turn's changes, reports 75 steady depth-1 cycles, 127 emitted tokens and 52/75
accepted drafts. The target interval accounts for 3.273 s of the 3.606 s cycle
stream interval (about 91%). Target expert-cache counters estimate 18,071 misses
and 50.10 GB of fetched expert payload. These are estimated bytes, not measured
PCIe traffic; nested interval times must not be summed. Draft time is 0.175 s,
snapshots 0.0105 s and commit 0.0102 s. The target/expert path remains the main
optimization area; state-copy or sampling changes alone cannot remove this cost.

A route/N grid-layout experiment used the actual hidden 2560, intermediate 640,
top-10 shapes at M=1/2/3/4, with shared and spread routing. All 16 outputs were
bitwise equal, but timing gains were inconsistent and mostly negative. It was
removed from production. [layout.json](layout.json) and
[nvfp4-layout-experiment.patch](nvfp4-layout-experiment.patch) preserve the result
and reproduction; `bench_nvfp4_moe.py --compare-layout` explicitly requires that
patch, rather than silently benchmarking identical implementations.

The earlier [host-ids-control.json](host-ids-control.json) tested avoiding copies
only for non-disk PLE. This checkpoint uses disk PLE, so both sides took the same
path. It is a negative control, not evidence of a speedup. The subsequent
asynchronous disk PLE implementation is the measured optimization above.

Early policy prototypes are retained separately as `initial.json`,
`ordinary-baseline.json`, and `before-probe-skip.json`. They exposed a biased
baseline and excess probe work and do not describe the final implementation.

## Correctness and lifecycle verification

[validation.json](validation.json) uses the real checkpoint and the same 1M-token,
3073-expert-slot geometry. `--force-auto-depths` supplies synthetic policy costs
to force every transition through 0/1/2/3, including periodic return to 0 and
resumption. Model execution, sampling and state restoration remain real. This is
a correctness test; its timing fields must not be treated as automatic-policy
performance results. The separate automatic throughput run uses actual timings.

- Both ordinary greedy prompts match MTP off for all 128 output tokens.
- Forced rejection accepts 0 of 279 proposals and matches MTP off.
- The continuation test and two concurrent greedy requests match their references;
  concurrent outputs also match the separate runs.
- Synthetic inputs of 8,190 and 32,760 tokens match MTP off for all 128 output
  tokens, crossing the context-bucket boundaries. Prefill chunks are at most 8192.
  These are repeated cat-themed source sentences plus a summary instruction,
  not a model-quality evaluation or near-million-token context test.
- Sampling settings `(.7, -1, 1)`, `(.8, 20, .9)` and `(1, 1, .8)` each complete
  two 64-token outputs. Tuple order is temperature, top-k, top-p.

```sh
python /opt/freetoken/benchmarks/bench_mtp.py --model "$MODEL" \
  --mtp-auto --force-auto-depths --trace-stalls --speculative-tokens 3 \
  --tokens 128 --warmup-tokens 64 --repeats 1 --validate \
  --sampling-smoke --sampling-tokens 64 --max-extend-tokens 8192 \
  --max-seq-len 1000128 --pages 15629 --moe-slots 3073 \
  --validation-contexts 8190,32760 --output /tmp/mtp-auto-forced-validation.json
```

This broader validation exposed two issues that fixed-depth microbenchmarks missed:

1. The first replay of a newly captured depth-2 verification graph could block
   before deferred PLE fill was called. First replay now stages PLE synchronously;
   subsequent replays retain asynchronous staging. The original blocked replay
   stack is in `forced-validation-stall.log`. A CUDA regression also asserts the
   first/subsequent staging distinction for different graph shapes.
2. Batching greedy depth-0 requests changed their arithmetic relative to separate
   single-row execution. Greedy auto now keeps the single-row path at every depth.
   The initial differing comparison is explicitly saved as
   `before-greedy-batch-guard.json`; the final comparison passes in `validation.json`.

The focused suite passed **369 tests**, including policy warmup, token-weighted
costs, baseline refresh, hysteresis, asynchronous event consumption, graph-capture
exclusion, draft suppression, batch/greedy dispatch, disk PLE byte fidelity,
multi-row deferred reads, state rollback, model residuals and cache rebuilds.
This is not a full repository test run.

```sh
python -m pytest -p no:cacheprovider -q \
  /opt/freetoken/tests/models/qwen4_exp/test_ple_disk.py \
  /opt/freetoken/tests/scheduler/test_mtp.py \
  /opt/freetoken/tests/scheduler/test_cache_rebuild.py \
  /opt/freetoken/tests/engine/test_kv_quant_config.py \
  /opt/freetoken/tests/engine/test_speculative.py \
  /opt/freetoken/tests/models/qwen4_exp/test_skeleton.py
```

## HTTP and OpenWebUI

Separate servers used the same checkpoint, capacity and two request slots.
Overlap scheduling was disabled for both with
`FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`. Each HTTP mode used one 128-token warmup
followed by three 512-token streams, with sampling parameters omitted.
The server resolves these to temperature 1, top-k 20, top-p .95.

| Mode | Median streamed delivery tok/s | Median TTFT, seconds |
|---|---:|---:|
| MTP off | 47.62 | 2.520 |
| Auto, ceiling 3 | 45.09 | 2.525 |

The HTTP median difference is -5.32%; the offline 5% target does not hold for
this small HTTP sample. Delivery rate is `(completion_tokens - 1)` divided by
time from first to last nonempty content/reasoning event. It is not a UI FPS
measurement. All six streams reported 61 input / 512 output tokens and one
`length` finish event followed by `[DONE]`.
[http-off.json](http-off.json) and [http-auto.json](http-auto.json) retain results.
Auto additionally passed a greedy stop-marker check, connection closure after
32 nonempty stream events, a subsequent 64-token request, and two simultaneous
128-token requests with completion and usage checks.

```sh
FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 ft serve --model "$MODEL" \
  --served-model-name Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --host 0.0.0.0 --port 1919 --max-running-requests 2 \
  --max-seq-len-override 1000128 --num-pages 15629 --moe-cache-size 3073 \
  --kv-cache-dtype nvfp4 --max-prefill-length 8192 --max-output-tokens 512 \
  --cache-type naive --mtp-speculative-tokens 3 --mtp-auto
python /opt/freetoken/benchmarks/bench_mtp_http.py \
  --url http://host.docker.internal:1921 \
  --model Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --tokens 512 --warmup-tokens 128 --repeats 3 --validate \
  --output /tmp/mtp-http-auto.json
```

For the off server, replace the final MTP options with
`--mtp-speculative-tokens 0`; the off HTTP run omits `--validate`.
Docker maps host port 1921 to container port 1919.

OpenWebUI v0.11.3 was tested through its authenticated browser UI using a
temporary chat and the exact cat prompt. Sampling controls were left unchanged.
The displayed answer grew across successive observations, finished, and exposed
the regeneration controls and follow-up suggestions. No UI error was displayed.
The server logged 6162 input tokens for this request, rather than the 61 tokens
in the bare API test; the UI environment adds context. Its exact request settings
were not captured, so this is a functional UI check, not a matched throughput A/B.
The UI supplied its own output budget: the server's 512 setting is only the
default for requests that omit a budget. Generation continued beyond 512 tokens
and completed naturally in the UI.

Server request 10 switched 0 -> 1 -> 0 (baseline 22.82 ms/token, depth 1
26.01 ms/token), followed by ordinary decode generally in the 40s tok/s in
periodic server logs. These interval rates are not an OpenWebUI displayed speed.
Request 11 then generated the UI's follow-up suggestions. See
[server-auto.log](server-auto.log). The temporary API connection was restored to
its original value after verification; user sampling settings were not edited.

## Source and usage

[runtime-change.patch](runtime-change.patch) records this turn's runtime delta
against the pre-turn source snapshot, which already includes earlier uncommitted
MTP work. It is not a patch against HEAD. [source.json](source.json) records the
changed runtime paths, hashes of tested sources, package versions and geometry.
The capture script is retained alongside it.

Enable automatic depth with these entries in the existing Compose command list,
replacing an existing speculative-token value rather than duplicating it:

```yaml
- --mtp-speculative-tokens
- "3"
- --mtp-auto
```

Server shutdown completed and all model workers exited. Python emitted one
`resource_tracker` leaked-semaphore warning during shutdown, retained in the log;
this shutdown warning remains unresolved. The development container was then
stopped, while OpenWebUI and the user's other previously running service remained
running.

The service must run the updated source or be rebuilt. This work does not edit
Compose, commit changes or start the user's normal service. The temporary test
server is stopped after verification. Active MTP still trails ordinary decode on
this hardware/model; target verification and expert movement remain the primary
future optimization area. Long-context correctness was tested around 8K and 32K,
not near the full reserved million-token capacity. Auto remains opt-in, TP=1,
and inherits MTP's disabled prefix caching and overlap scheduling.
