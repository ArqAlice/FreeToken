# Target verification bottleneck investigation

Date: 2026-09-10 JST. Local branch `feat/mtp-support`, base HEAD
`3381015664c3acc5003b04fcad9a29742429792c`, retaining prior uncommitted work.
RTX 5090 32 GB, Threadripper PRO 5965WX, driver 591.86, CUDA 13, Docker/WSL2.
Checkpoint `aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP`, revision
`18bd7d69491d4709ac1933e95343e15f045ea21d`, in `freetoken_hf-cache`.
Runtime: `/opt/venv/bin/python`, `PYTHONPATH=/opt/freetoken/python`.
The `$MODEL` argument in commands denotes
`/models/huggingface/hub/models--aday777--Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP/snapshots/18bd7d69491d4709ac1933e95343e15f045ea21d`.

All model tests use capacity 1,000,128, 15,629 KV pages, 3,073 expert slots,
two request slots, NVFP4 KV, Triton NVFP4, naive cache and disk PLE wait-sync.
Prompt: `猫の魅力について熱くたくさん語って。`, 61 chat-template input tokens,
temperature 1, top-k 20, top-p .95 unless a test specifies otherwise.
The reference is the branch before this additional optimization, not upstream
main. No upstream-main benchmark, commit, push, PR or Compose edit was performed.

## Diagnosis

[before.json](before.json) contains a separate instrumented 128-token generation.
The 75 steady depth-1 cycles emitted 127 tokens in 3.622 s of CUDA event intervals;
target verification accounts for 3.274 s (90.4%). It visited 59,716 unique experts
per layer invocation, with 18,071 misses and an estimated 50.10 GB of bank payload
fetched. The estimate is not a PCIe traffic-counter measurement.

The separate PyTorch profiler run includes prefill. [kernels.json](kernels.json)
excludes kernels before the final prefill expert kernel and reports target plus
draft decode kernel durations: expert copies 1.974 s, BF16 GEMV 1.036 s, NVFP4
expert GEMV 0.128 s, LRU bookkeeping 0.0365 s, other kernels 0.683 s. Durations
can overlap across streams; these are not exclusive wall-time shares. The full
profiler trace is `/tmp/mtp-target-trace.json` in the development container.

```sh
python /opt/freetoken/benchmarks/bench_mtp_chat.py --model "$MODEL" \
  --modes 1 --tokens 128 --warmup-tokens 128 --repeats 0 --diagnostics \
  --profile --profile-trace /tmp/mtp-target-trace.json \
  --output /tmp/mtp-target-before.json
```

## Expert-copy launch geometry

[copy-sweep.json](copy-sweep.json) sweeps six launch geometries at 0/1/4/8/16/24/40
misses using the checkpoint's six bank sizes. Each configuration copies randomly
permuted rows from pinned host memory and checks every destination byte, including
untouched rows. All 42 cases match. Each timing has three CUDA-graph measurements.
For nonzero misses the best configuration improves at most 0.4%, with no consistent
winner. The production copy kernel and launch defaults were retained.

```sh
python /opt/freetoken/benchmarks/bench_mtp_copy.py --output /tmp/mtp-copy-sweep.json
```

The PCIe link was read with `nvidia-smi` during generation: generation 4, width 16
(current and maximum reported values). Copy-only tests sustain about 25.5 GB/s.
This supports focusing on transferred bytes and computation overlap rather than
increasing copy-kernel parallelism.

## Cache simulation

[routes.json](routes.json) records the actual eager route sequence for a separate
256-token generation. Graphs are deliberately disabled to capture every invocation;
its elapsed time is not used as a throughput benchmark. The simulation replays
the initial slot contents and routes without running the model.

| Policy | Fetch count |
|---|---:|
| LRU | 33,885 |
| LRU2 | 36,377 |
| LFU | 41,548 |
| Decayed frequency, half-life 256 layer calls | 32,978 |
| Decayed frequency, half-life 512 layer calls | 33,953 |
| Decayed frequency, half-life 1024 layer calls | 35,889 |

The best simulated candidate reduces fetches by 2.68%. Giving tentative-only
expert routes lower LRU priority also failed to improve this trace. These are
single-trace simulations, not evidence of general cache-policy superiority.

```sh
python /opt/freetoken/benchmarks/bench_mtp_chat.py --model "$MODEL" \
  --modes 1 --tokens 256 --warmup-tokens 128 --repeats 0 \
  --diagnostics --routing-trace --output /tmp/mtp-target-routes.json
python /opt/freetoken/benchmarks/results/mtp-target/simulate_cache.py \
  /tmp/mtp-target-routes.json /tmp/mtp-cache-simulation.json
```

## Decayed-frequency implementation and comparison protocol

The candidate tracks a float frequency and the last-use clock for each expert.
A unique route updates `frequency = frequency * 2**(-age / half_life) + 1`.
Slot priorities are ranked in log space, with a negative offset so the existing
LRU kernel can still protect current hits using its positive clock. It performs
the same deduplication, slot mapping and exact-byte copies as before. Reset and
cache rebuild clear the extra history. No sampling or weight arithmetic changes.
Log ranks are quantized to 1/256 of a clock step; the simulation is a policy
screening tool, not a bit-exact oracle for every victim tie.

The candidate is opt-in with `FREETOKEN_EXPERT_LRFU_HALF_LIFE=256`; zero is ordinary
LRU. The experimental implementation accepts 0 through 4096 and fewer than 40,000
slots, keeping the negative priority representation out of the streaming selector.

The paired comparison uses separate captured graph dictionaries for each policy,
resets the cache before each measurement and performs the same seeded 128-token
warmup. Five 256-token measurements per policy and depth alternate order and use
matching seeds. Reset and warmup are outside the measured interval. The preliminary
`lrfu.log` was interrupted when this protocol was tightened; it is not performance
evidence. The final run is `lrfu-fair.json` / `lrfu-fair.log`.

```sh
python /opt/freetoken/benchmarks/bench_mtp_chat.py --model "$MODEL" \
  --modes 1,3 --tokens 256 --warmup-tokens 128 --repeats 5 \
  --compare-lrfu --output /tmp/mtp-target-lrfu-fair.json
```

| Fixed depth | LRU median tok/s | Decayed frequency median tok/s | Change |
|---|---:|---:|---:|
| 1 | 35.72 | 36.12 | +1.13% |
| 3 | 24.78 | 24.98 | +0.82% |

All ten paired output token sequences match exactly. Individual timing pairs
include regressions, especially at depth 3, so these small median gains do not
establish a general speedup. The candidate remains disabled by default.

## Independent BF16 rows

The other large kernel category is BF16 matrix-vector multiplication. Sequential
single-row operations preserve the target's arithmetic but serialize independent
rows. A separate experiment runs those same `torch.mm` calls on different CUDA
streams and joins them before the caller consumes the output.

The microbenchmarks cover the actual GDN, attention, shared-expert and
hyper-connection shapes. All tested outputs are bitwise equal. Some large shapes
show large, unstable timing swings, while the fused GDN input projection does
not improve. The candidate therefore targets only the consistently useful shapes:
2560 -> 1280 at 2-4 rows; 2560/10240 -> 336 at 3-4 rows; and 6144/2560 -> 2560
or 2560 -> 512 at four rows. Bias, other dtypes, CPU inputs and other shapes keep
the sequential path. See [row-streams-shapes.json](row-streams-shapes.json) and
[row-streams-two.json](row-streams-two.json).

The runtime switch is `FREETOKEN_MTP_PARALLEL_LINEAR=1`, disabled by default.
It retains the per-row GEMM, unlike `FREETOKEN_MTP_BATCHED_LINEAR=1`, which changes
the GEMM shape and can change numerical results. Four persistent streams per
device are allocated lazily. Both successful calls and exceptions join the streams
before returning control to the caller.

```sh
python /opt/freetoken/benchmarks/bench_mtp_linear.py --bf16-streams-only \
  --row-stream-count 4 --output /tmp/mtp-row-streams-shapes.json
python /opt/freetoken/benchmarks/bench_mtp_chat.py --model "$MODEL" \
  --modes 1,3 --tokens 256 --warmup-tokens 128 --repeats 3 \
  --compare-row-streams --output /tmp/mtp-row-streams-model.json
```

| Fixed depth | Sequential median tok/s | Parallel median tok/s | Change |
|---|---:|---:|---:|
| 1 | 34.85 | 34.80 | -0.15% |
| 3 | 24.80 | 25.19 | +1.58% |

All six paired output token sequences match exactly. All three depth-3 timing
pairs improve (1.93%, 1.37%, 1.58%), but this is still a small, single-workload
sample. Depth 1 does not improve. Parallel rows remain off by default, and active
MTP still trails the previous MTP-off measurements. Different experiment tables
use separate runs and should not be treated as a combined speedup. The new two
options are tested together for correctness, not for combined throughput.

## Verification and delivery

The focused suites passed 303 cache/MTP tests and 82 linear tests (385 total),
including frequency decay, duplicate routes, bidirectional slot-map consistency,
reset, invalid geometry, CUDA graph replay and bitwise single-row linear parity
with noncontiguous inputs. This is not a full repository test run.

```sh
python -m pytest -p no:cacheprovider -q \
  /opt/freetoken/tests/moe/test_offload.py \
  /opt/freetoken/tests/moe/test_fused_copy.py \
  /opt/freetoken/tests/scheduler/test_cache_rebuild.py \
  /opt/freetoken/tests/scheduler/test_mtp.py
python -m pytest -p no:cacheprovider -q \
  /opt/freetoken/tests/layers/test_linear_mtp.py
```

[validation.json](validation.json) enables both new settings on the real model.
It passes both short greedy references, forced rejection (0/279 accepted),
continuation, two concurrent requests matching separate runs, and 8190 input
tokens crossing the 8192 boundary, with 128 output tokens. Three sampling
settings also complete two 64-token requests each. `--force-auto-depths` uses
synthetic policy costs to force 0/1/2/3 transitions; weights, computations and
state restoration remain real. Its times are not automatic-policy performance
measurements. This turn did not repeat the earlier 32K or OpenWebUI tests.

```sh
FREETOKEN_EXPERT_LRFU_HALF_LIFE=256 FREETOKEN_MTP_PARALLEL_LINEAR=1 \
python /opt/freetoken/benchmarks/bench_mtp.py --model "$MODEL" \
  --mtp-auto --force-auto-depths --trace-stalls --speculative-tokens 3 \
  --tokens 128 --warmup-tokens 64 --repeats 1 --validate \
  --sampling-smoke --sampling-tokens 64 --max-extend-tokens 8192 \
  --max-seq-len 1000128 --pages 15629 --moe-slots 3073 \
  --validation-contexts 8190 --output /tmp/mtp-target-validation.json
```

[runtime-change.patch](runtime-change.patch) records the runtime delta against
the source snapshot taken before these changes, including the earlier uncommitted
MTP implementation. It is not a patch against HEAD. [source.json](source.json)
records source hashes and runtime versions; the source-capture and trace-analysis
scripts are included. After the measurements, the benchmark's environment report
was extended to explicitly include the LRFU environment variable; this reporting
addition does not change the measured operations.

No production default was changed. For fixed-depth-3 experiments, enable
`FREETOKEN_MTP_PARALLEL_LINEAR=1` in the service environment and restart the
updated service. The separate cache experiment uses
`FREETOKEN_EXPERT_LRFU_HALF_LIFE=256`. The standard automatic-depth option remains
useful for avoiding unprofitable MTP. The fused GDN input projection and expert
traffic still dominate; the measured changes do not close the gap to MTP off.

The temporary model process exited, and the development container was returned
to its original stopped state. OpenWebUI and the other initially running service
were left running. The user's normal FreeToken service was not started or changed.
