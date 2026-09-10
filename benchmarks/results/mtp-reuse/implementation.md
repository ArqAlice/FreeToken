# Expert transfer reduction and shared GDN weights

Date: 2026-09-10 JST. Local `feat/mtp-support` branch at HEAD
`3381015664c3acc5003b04fcad9a29742429792c`, with prior uncommitted MTP work retained.
RTX 5090 32 GB, Threadripper PRO 5965WX, PCIe Gen4 x16, driver 591.86,
CUDA 13, Docker/WSL2. Checkpoint
`aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP`, revision
`18bd7d69491d4709ac1933e95343e15f045ea21d`, in `freetoken_hf-cache`.

The reference is the previous state of this local branch, not upstream main.
The checkpoint/MTP branch has not been benchmarked against upstream main.
No commit, push, PR, Compose edit, or production service change is part of this work.

## Implementation

Expert eviction optionally combines decayed frequency with cyclic layer distance.
Set `FREETOKEN_EXPERT_LRFU_HALF_LIFE=256` and
`FREETOKEN_EXPERT_LAYER_DISTANCE=2`. The existing log priority subtracts
`256 * distance_penalty * ((owner_layer - current_layer - 1) mod num_layers)`.
The same preparation kernel computes this adjustment without another launch,
host synchronization, cache slots, or expert-weight copies. Active experts remain
protected by the existing LRU ensure kernel. Repeated draft layers and concurrent
requests mean distance is a heuristic. Zero penalty retains the previous policy.

`FREETOKEN_GDN_SHARED_INPUT=1` dispatches unquantized Qwen4 GDN input projections
with 1-5 decode/verification rows to `bf16_shared_linear.py`. A fixed 16-row tile
shares BF16 weights among input rows, using BF16 dot products with FP32 accumulation.
Four independent K partitions store FP32 partial results. A second kernel combines
them in a deterministic tree and casts once to BF16. There are no atomic reductions.
Ordinary single-token decode uses the same kernel/reduction, even with MTP off.
Prefill, other linears, quantized GDN inputs and unsupported layouts/dtypes keep
their prior paths. Both options are disabled by default and require a restart.

This changes GDN arithmetic relative to cuBLAS. Agreement between the new
single-row and multi-row paths is different from agreement with the old backend.
The old and new backends are not promised to generate identical text.

## Screening

`probe_cache.py` replays the previous recorded depth-1 route sequence. Plain LRU
requires 33,885 fetches; half-life 256 alone requires 32,978; adding distance penalty
2 requires 31,920 (-5.8% versus LRU). Penalties 4 and 8 yield 31,897 and 31,870:
the smaller penalty was selected to avoid relying on that trace's marginal gains.
These are simulations; production log scores are quantized and tie handling differs.

`probe_linear.py` tried output tiling around single-row cuBLAS calls. The only
fully matching tile size (512) was slower, so this approach was not integrated.
`probe_shared.py` screens shared-kernel split/tile choices. Four splits with
64 output columns give about 0.017 ms for four rows versus 0.211 ms for sequential
cuBLAS. This is a repeated CUDA-graph microbenchmark with warm weights/cache,
not an end-to-end speedup. All 28 configurations match their own single-row
execution bitwise; some elements differ from cuBLAS. Relative RMS error against
FP32 reference is about 0.00165, including the final BF16 rounding.

## Model protocol

`run_model.py` first runs correctness validation, then starts fresh processes for
baseline, expert-only, GDN-only and combined settings. Each process uses separate
startup CUDA graphs for its environment. Sampling uses the cat chat prompt
`猫の魅力について熱くたくさん語って。` (61 input tokens), temperature 1, top-k 20,
top-p .95, 256 output tokens, three repetitions per depth 0/1/3. Each measurement
resets the expert cache and runs the same 128-token seeded warmup outside timing.
Measurement seeds match between settings; depth order alternates per repetition.
The process order is fixed, so between-process drift is not eliminated.

All model runs retain max sequence capacity 1,000,128, 15,629 KV pages, 3,073
expert slots, two request slots, NVFP4 KV, Triton NVFP4 and naive prefix cache.
Device expert counters are enabled for transfer comparison. Reported bytes are
miss counts times packed bank row bytes, not hardware PCIe traffic counters;
the run includes the generation's routed expert invocations. Timing is from
first to last delivered token and excludes prefill/TTFT. These instrumented
measurements should not be mixed with earlier uninstrumented results.

## Verification

408 focused tests passed: linear/kernel numerical checks, cache routing/mapping,
CUDA graph replay, resets/rebuild, invalid settings, fused copy and MTP scheduler.
Shared projection tests include noncontiguous input, unaligned output tail,
one through five rows, changing graph inputs, and a float64 reference. The initial
test invocation had six test-only missing-import failures, fixed before this pass.

```sh
PYTHONPATH=/opt/freetoken/python /opt/venv/bin/python -m pytest -p no:cacheprovider -q \
  /opt/freetoken/tests/layers/test_linear_mtp.py \
  /opt/freetoken/tests/moe/test_offload.py /opt/freetoken/tests/moe/test_fused_copy.py \
  /opt/freetoken/tests/scheduler/test_cache_rebuild.py \
  /opt/freetoken/tests/scheduler/test_mtp.py
/opt/venv/bin/python /opt/freetoken/benchmarks/results/mtp-reuse/run_model.py
```

The combined settings pass short greedy parity (English factual completion and
Python generation), forced rejection (0/375 proposals accepted), continuation,
two concurrent requests matching separate runs, and 8,190 input tokens plus
128 output tokens across the 8,192 boundary. Three sampling configurations
(.7/-1/1, .8/20/.9, 1/1/.8) complete two 64-token outputs each. These validate
MTP against ordinary decoding with the new projection enabled, not against
the previous cuBLAS arithmetic. See `validation.json` and `validation.log`.

## Expert-only model results

| Depth | Baseline median tok/s | Expert median tok/s | Fetches before | Fetches after | Payload reduction |
|---|---:|---:|---:|---:|---:|
| 0 | 44.86 | 45.01 | 74,579 | 72,554 | 2.72% |
| 1 | 34.78 | 35.67 | 101,648 | 94,227 | 7.30% |
| 3 | 24.72 | 25.01 | 156,790 | 147,415 | 5.98% |

Fetch totals cover the three 256-token generations per setting/depth. All nine
matched-seed output token pairs are identical, including proposal/acceptance
counts for active MTP. The extra cache policy work offsets part of the transfer
savings: median throughput improves 2.57% at depth 1 and 1.14% at depth 3.
MTP-off throughput changes only 0.33%, within the observed timing spread.

## Shared-GDN-only model results

| Depth | Baseline median tok/s | GDN median tok/s | Change | GDN range |
|---|---:|---:|---:|---:|
| 0 | 44.86 | 43.31 | -3.45% | 42.12-43.76 |
| 1 | 34.78 | 36.55 | +5.10% | 35.38-37.69 |
| 3 | 24.72 | 27.29 | +10.37% | 26.26-29.88 |

All nine sampling token sequences differ from the previous backend. The routed
expert workload and proposal/acceptance counts also change, so these are workload
outcomes rather than an isolated kernel-speed comparison. For example, MTP-off
fetches increase from 74,579 to 81,366 across the three generations. The measured
GDN-only option is therefore not an across-the-board speedup. See `gdn.json`.

## Combined model results

| Depth | Baseline median tok/s | Combined median tok/s | Change | Combined range |
|---|---:|---:|---:|---:|
| 0 | 44.86 | 42.81 | -4.57% | 41.56-43.52 |
| 1 | 34.78 | 37.05 | +6.52% | 36.25-37.97 |
| 3 | 24.72 | 28.51 | +15.31% | 27.19-31.14 |

All nine GDN-only/combined token pairs match, while all nine baseline/combined
pairs differ. On the new GDN route sequence, the cache policy reduces fetches
from 104,057 to 97,319 at depth 1 and 134,772 to 122,254 at depth 3 (6.48% and
9.29%). Comparing combined transfer totals directly against the old backend
would conflate cache improvements with changed generated text and expert routes.

TTFT medians stay around 2.50-2.51 seconds. Reported allocated CUDA memory is
28,574,393,856 bytes for baseline/GDN and 28,574,694,912 for expert/combined.
This is allocated memory at the end of a run, not a peak-memory measurement.
Active MTP still trails MTP off, even after these changes. Neither optional
setting is enabled by default or inserted into Compose.

## Weight-working-set check

`probe_cold.py` rotates four different GDN matrices (337,510,400 bytes total).
Unlike the initial single-matrix microbenchmark, the working set does not stay
as one repeatedly reused matrix. Per-projection median times are:

| Rows | Sequential cuBLAS ms | Shared projection ms |
|---|---:|---:|
| 1 | 0.05246 | 0.05339 |
| 2 | 0.10358 | 0.05353 |
| 3 | 0.15094 | 0.05366 |
| 4 | 0.20253 | 0.05430 |
| 5 | 0.26154 | 0.05386 |

The four-row projection improves about 3.7x in this test, while a single row
does not improve. This explains why the warm-cache 12x figure should not be
extrapolated to end-to-end throughput or to MTP-off decoding.

## Same-state numerical comparison

`followup_model.py` runs `bench_mtp_chat.py --compare-gdn-input` separately from
timing. For 16 verification batches (64 rows), it snapshots target state,
evaluates the old GDN path eagerly, restores state, then evaluates the new path
on the same inputs. Other linear projections retain their strict path. CPU
FP32 softmax plus top-20/top-p .95 threshold filters summarize the distributions.

| Metric | Value |
|---|---:|
| Top-1 agreement | 61/64 (95.31%) |
| Mean unfiltered total variation distance | 0.014707 |
| Maximum unfiltered TV | 0.094954 |
| Mean filtered TV | 0.015116 |
| Maximum filtered TV | 0.094978 |
| Maximum absolute logit difference | 1.796875 |

Small projection rounding changes can propagate through later recurrent,
attention and expert-routing layers. These results rule out claiming equivalence
to the previous backend's output distribution. This is not a perplexity or model
quality evaluation. The default remains cuBLAS; the shared kernel is explicitly
experimental. With the shared kernel selected, separate MTP-on/off correctness
checks use its matching single-token reduction. See `numerics.json`.

```sh
/opt/venv/bin/python /opt/freetoken/benchmarks/results/mtp-reuse/followup_model.py
```

## Usage

Add these keys to the service's existing `environment` to try both options:

```yaml
FREETOKEN_EXPERT_LRFU_HALF_LIFE: "256"
FREETOKEN_EXPERT_LAYER_DISTANCE: "2"
FREETOKEN_GDN_SHARED_INPUT: "1"
```

The fixed-depth-3 numbers use `--mtp-speculative-tokens 3`. The checked-in local
Compose currently specifies zero; the environment keys alone do not enable MTP.
Rebuild the image containing the updated source and restart for new CUDA graphs.
To preserve the old GDN arithmetic, omit `FREETOKEN_GDN_SHARED_INPUT` and try
only the two cache keys. Existing default settings remain unchanged.

## Automatic depth and delivery

`auto.json` / `auto.log` pass short greedy parity, forced rejection, continuation
and concurrent requests while forcing depth 0/1/2/3 transitions. Policy costs are
synthetic to exercise those paths; model computations and state restoration are
real. This is a correctness test, not an automatic-policy speed benchmark. The
separate fixed-depth validation above covers the 8K boundary and sampling.
This work did not repeat the previous 32K or HTTP/OpenWebUI tests, nor does it
establish quality or performance on other checkpoints, GPUs or tensor parallelism.

`summary.json` summarizes all 36 instrumented throughput runs. `runtime-change.patch`
records only this turn's runtime changes against the saved local pre-reuse source;
`source.json` records the tested runtime and source hashes. `git diff --check`
passed. Both model-run drivers completed successfully; no GPU compute process
remained. The development container was stopped again, and OpenWebUI/open-terminal
were left running. The ordinary FreeToken service stayed stopped as it was initially.
