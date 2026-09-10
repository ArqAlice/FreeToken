# Experimental Qwen4 MTP decoding

See the [automatic-depth implementation and measurements](../benchmarks/results/mtp-auto/implementation.md)
for the latest real-model, HTTP and OpenWebUI validation.
The [target-bottleneck follow-up](../benchmarks/results/mtp-target/implementation.md)
adds two optional experiments; neither makes active MTP faster than MTP off on
the measured system.

Enable speculation with `--mtp-speculative-tokens N`, where `N` is 1 through 4.
Use 0 to disable MTP. This implementation targets
`aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP` on one GPU.
For a Compose service whose command is a YAML list, add the following, or replace
the value of an existing `--mtp-speculative-tokens` entry:

```yaml
  - --mtp-speculative-tokens
  - "1"
```

Both greedy decoding and sampling with temperature, top-k and top-p use MTP.
The sampling options belong to the generation request; for example:

```json
{"temperature": 0.8, "top_k": 20, "top_p": 0.9, "max_tokens": 256}
```

A larger lookahead does not guarantee faster generation. Start with 1 and measure
your workload before increasing it. In the measured Japanese chat workload, 1 is
the fastest MTP setting, but MTP-disabled generation is still faster.

Add `--mtp-auto` to select the lookahead from measured time per emitted token.
The positive `--mtp-speculative-tokens` value remains the ceiling; automatic
selection considers at most 3. For the existing Compose command list:

```yaml
  - --mtp-speculative-tokens
  - "3"
  - --mtp-auto
```

This mode requires one GPU (TP=1). Each request first uses ordinary decode while
maintaining the draft cache. It excludes the first 16 cycles, then measures four
cycles for the baseline and each candidate. CUDA graph capture is excluded from
selection, but remains part of user-visible latency. Timing events are read only
after completion; the policy adds no synchronization to wait for a measurement.
The cost is exponentially averaged elapsed time divided by exponentially averaged
emitted tokens, including verification, correction, state commit and draft work.

If depth 1 cannot improve on ordinary decode by at least 3%, speculation stops for
the remainder of that request. Otherwise depths 2 and 3 are explored up to the
ceiling. Active speculation remeasures the ordinary baseline every 32 updates and
recalibrates at 8192-token context boundaries. Switching between profitable depths
requires a 10% estimated improvement. An unprofitable current depth can switch
without that margin. This is a conservative heuristic, not a
guarantee of the globally fastest depth: an early depth-1 loss skips deeper probes.
A new request starts a new calibration. Decisions use completed cycles, never a
newly sampled candidate's probability, preserving the sampling acceptance rule.

Depth 0 uses the ordinary decode graph; all-zero sampling batches use normal
batched decode. Greedy requests retain one-row execution so a depth change does
not change the BF16 reduction order compared with separate greedy requests.
Draft cache maintenance during calibration omits vocabulary projection and
sampling until the next probe needs a proposal. After stopping, draft work is
removed entirely. Probe and lazy-capture costs can still make short responses
slower than `--mtp-speculative-tokens 0`; auto stopping is not a speedup of the MTP
verification kernels. Server logs report depth transitions and estimated ms/token.

To try the optional batched BF16 path measured below, add this entry to the
service's existing `environment` mapping:

```yaml
FREETOKEN_MTP_BATCHED_LINEAR: "1"
```

This option can change rounding and sampling probabilities; it is disabled by
default. Omit it or set it to `"0"` to retain the single-row calculation. When
using this repository's Compose service, rebuild the source and recreate the
server after editing the configuration:

```sh
docker compose up -d --build freetoken
```

The first request can incur lazy graph-capture overhead. Compare warmed requests
with the same prompt, sampling settings and generation length.

The first draft consumes the target's hyper-connection residual and the next token
embedding. Further drafts feed the head's pre-final-mixer residual back into the
same MTP head. The target verifies the known input and all drafts together.
Greedy decoding accepts the matching prefix. Stochastic decoding accepts a draft
with probability `min(1, p(token) / q(token))`, using the same temperature and
filters for target distribution `p` and draft distribution `q`. At the first
rejection, it samples a correction from normalized `max(p - q, 0)`. Full acceptance
adds a bonus token from the target. Equal random seeds do not imply identical
sampled sequences with MTP enabled and disabled: the random draws differ.

`FREETOKEN_MTP_GREEDY_DRAFT=1` selects deterministic argmax proposals while keeping
the target's temperature, top-k and top-p. The target samples each verification
row independently and accepts the matching prefix. The first mismatching sample
is the correction, or the final row supplies a bonus after full acceptance. This
is rejection sampling for a point-mass draft distribution: acceptance has mass
`p(draft)`, and a mismatch has distribution `p` conditioned on excluding the draft.
It removes draft softmax, filtering, RNG and stored vocabulary probabilities.
Acceptance can be lower than with stochastic proposals; this experiment is off
by default. Set the variable before starting the process. Greedy target requests
keep their existing behavior; stochastic output need not match previous seeds.
In the [real-model comparison](../benchmarks/results/mtp-20260910-greedy-draft/implementation.md),
acceptance fell and throughput decreased at depths 1 and 3; leave this disabled
for the tested checkpoint and default chat sampling settings.

`FREETOKEN_MTP_QSA_REUSE=1` enables an independent draft-only experiment for depths
2-4. A draft-extend saves the final row's selected compressed blocks for each MTP
QSA layer. Adjacent lookahead steps reuse those blocks while rebuilding the causal
tail for the current position. KV and raw index-key writes still run at every
step. Query normalization/rope, block scoring and top-k are skipped on reuse;
the index projection and key compression are retained.

Selection refreshes at compression closure, ring wrap, a nonadjacent position,
request change, configured depth change or block-table width change. Draft repair
always refreshes. Chunked prefill without logits, request cleanup and cache rebuild
invalidate ownership. CUDA graphs share fixed buffers and distinguish refresh,
reuse and uncached paths; target batches never receive these buffers.

Within the selection budget the same blocks remain selected. Beyond that budget,
the draft's selection and probability distribution can change; verification uses
the actual resulting draft probabilities. Target selection and sampling settings
are unchanged. Depth 1 has no reuse opportunity and allocates no selection buffer.
This experiment is disabled by default.
The [QSA reuse measurements](../benchmarks/results/mtp-20260910-qsa-reuse/implementation.md)
showed unchanged throughput on the short chat and about 6% improvement on one
3661-token synthetic prompt. The sparse-context comparison changes stochastic
continuations as well; neither case overtook MTP off.

After rejection, the target commits the saved GDN state after the accepted prefix.
GDN and PLE convolution histories are rebuilt from their saved inputs, and the QSA
index ring retains the accepted writes. This avoids running the target again for
the accepted prefix. The recurrent buffer reserves the configured maximum draft
depth plus one step: 216 MiB at depth 1, 432 MiB at depth 3, or 540 MiB at depth 4
with this checkpoint's default FP32 state dtype, plus small convolution buffers.
It is allocated on the first verification and reused across requests and graph
shapes. Runtime depth increases beyond that initial capacity use restore-and-replay
without replacing buffers referenced by existing graphs. For a QSA-only draft
head, lookahead restores only the index ring; target recurrence is committed from
verification history without first copying its old value.
`FREETOKEN_MTP_STATE_HISTORY=0` selects the earlier restore-and-replay path.
Unused whole KV pages are freed.
The committed target residual rebuilds the draft head state, replacing speculative
history. Lookahead shrinks near the generation limit or when KV pages run short.
Chunked prompts preserve the residual at each boundary. EOS, stop strings and
output limits are processed in token order.

Speculative requests are serialized within a batch; automatic depth-0 sampling batches
can use ordinary batched decode. Prefix-cache reuse and overlap scheduling
are disabled. Target continuations of one through five tokens use CUDA Graph replay
when engine CUDA graphs are enabled. Graphs are captured on first use, adding
initial latency, and released before runtime cache resizing. Set
`FREETOKEN_MTP_CUDA_GRAPH=0` for eager target forwards and
`FREETOKEN_MTP_DRAFT_GRAPH=0` for eager draft forwards. Draft continuations also use
graphs, with residual inputs and QSA write positions updated on each replay.
Small verification batches use the decode expert cache rather than transferring
all experts. Target BF16/NVFP4 linears and PLE convolution preserve the single-step
reduction order; GDN uses the decode recurrence, and QSA keeps the decode
attention split-K profile.

Target replay and intermediate prompt chunks omit the final mixer and vocabulary
projection when only residual state is needed. Draft generation stops at the
output limit. Sampling parameters are prepared once per request, and disk PLE
stages only the current tokens and two preceding tokens. With disk PLE and CUDA
wait-sync available, verification reads back the GPU draft IDs asynchronously,
launches the target graph, and stages/signals the disk rows before the graph's PLE
lookup consumes them. A newly captured verification graph's first replay stages
rows synchronously: its launch can block before deferred staging gets a chance
to signal the graph's PLE wait. Eager execution and gate-sync also stage synchronously.
`FREETOKEN_MTP_ASYNC_PLE=0` restores synchronous verification readback for A/B tests.
Non-disk PLE does not copy verification IDs to the CPU.
NVFP4 verification
linears share weight loads across rows for sufficiently wide projections while
preserving the single-token reduction order. Narrow projections retain independent
rows. Routed NVFP4 experts also use arithmetic unpacking that preserves the earlier
weight reconstruction and accumulation. BF16 linears retain the single-row path
by default.

QSA prefill and speculative metadata bound score work to the active context,
rounded up to a power-of-two page bucket with an 8192-token minimum. Graph keys
include this width so context growth selects the correct captured graph. Score
and top-k workspaces reuse views of existing buffers. The optional torch.topk
fallback retains the full width to preserve its tie behavior.

Acceptance-prefix reduction and correction-distribution selection run on the GPU.
The scheduler still reads the accepted length to commit state and manage pages.
Server generation-throughput logs count emitted tokens, including multiple accepted
tokens per verification; padding, aborted requests and tokens discarded after
termination do not contribute. OpenWebUI measures its own delivery interval.

For non-greedy requests, `FREETOKEN_MTP_BATCHED_LINEAR=1` enables ordinary batched
BF16 linear projections. Their accumulation and rounding can differ from
single-row decoding, changing logits and the filtered sampling distribution.
This is an opt-in throughput experiment, disabled by default; greedy requests
continue to use the strict path. The speculative acceptance rule remains the
same, but this option does not promise the default path's numerical equivalence.

MTP starts the current layer's routed-expert cache
lookup and fetch on a separate CUDA stream while computing its shared expert.
The compute stream waits before the routed GEMM; cache eviction also waits for
the previous layer's reads. This applies to Qwen4 MTP continuation batches
using GPU expert offload. Ordinary prefill, CPU/hybrid experts and resident
experts retain their existing path. Set `FREETOKEN_MTP_EXPERT_OVERLAP=0` before
starting the process to disable it; stream allocation and CUDA graph capture
depend on the setting.

| Environment variable | Default | Purpose |
|---|---:|---|
| `FREETOKEN_MTP_CUDA_GRAPH` | `1` | Target continuation graphs |
| `FREETOKEN_MTP_DRAFT_GRAPH` | `1` | Draft continuation graphs |
| `FREETOKEN_MTP_STATE_HISTORY` | `1` | Commit saved prefix state without target replay |
| `FREETOKEN_MTP_NVFP4_SHARED_ROWS` | `1` | Reuse NVFP4 weights across wide projection rows |
| `FREETOKEN_NVFP4_MOE_ARITHMETIC` | `1` | Arithmetic NVFP4 expert-weight unpacking |
| `FREETOKEN_MTP_BATCHED_LINEAR` | `0` | Experimental batched BF16 projections for sampling |
| `FREETOKEN_MTP_EXPERT_OVERLAP` | `1` | Overlap expert fetch with shared-expert compute in MTP continuations |
| `FREETOKEN_MTP_GREEDY_DRAFT` | `0` | Argmax proposals with unchanged target sampling settings |
| `FREETOKEN_MTP_QSA_REUSE` | `0` | Reuse draft QSA blocks between safe adjacent lookahead steps |
| `FREETOKEN_EXPERT_LRFU_HALF_LIFE` | `0` | Experimental decayed-frequency expert cache; tested at `256` layer calls |
| `FREETOKEN_EXPERT_LAYER_DISTANCE` | `0` | Penalize distant layers in decayed-frequency eviction; requires a positive half-life |
| `FREETOKEN_GDN_SHARED_INPUT` | `0` | Fixed-reduction BF16 GDN projection sharing weights across decode/verification rows |
| `FREETOKEN_MTP_PARALLEL_LINEAR` | `0` | Parallel single-row BF16 GEMMs for selected measured shapes |

The decayed-frequency cache changed fixed-depth-1 median throughput from 35.72 to
36.12 tok/s in five paired measurements; depth 3 changed from 24.78 to 24.98.
Pairs include regressions, so it remains experimental. It affects GPU expert
admission in ordinary decode as well as MTP, uses the same cache capacity, clears
history on reset/rebuild, and supports fewer than 40,000 slots. Nonzero half-lives
must be at most 4096. No weight precision or sampling distribution is changed.

Parallel BF16 rows changed fixed-depth-3 median throughput from 24.80 to 25.19
tok/s in three paired measurements, with identical output token sequences.
Depth 1 did not improve. This preserves separate single-row GEMMs and is distinct
from the numerical approximation allowed by `FREETOKEN_MTP_BATCHED_LINEAR`.
Both new options require a process restart because graphs capture their paths.
To try parallel rows for fixed depth 3, add to the service's existing environment:

```yaml
FREETOKEN_MTP_PARALLEL_LINEAR: "1"
```

The cache experiment can separately be enabled with
`FREETOKEN_EXPERT_LRFU_HALF_LIFE: "256"`. Their speedups have not been measured
in combination and must not be added together. Both remain off by default.

For the next expert-cache experiment, combine
`FREETOKEN_EXPERT_LRFU_HALF_LIFE=256` with `FREETOKEN_EXPERT_LAYER_DISTANCE=2`.
The latter subtracts a cyclic layer-distance penalty from the frequency rank,
favoring experts in upcoming layers when their reuse frequency is similar.
It changes eviction decisions, not routing, weight precision or cache capacity.
Distance penalties must be finite and between 0 and 16. Repeated draft layers
and concurrent requests make this a heuristic, not a prediction of future routes.

`FREETOKEN_GDN_SHARED_INPUT=1` selects a BF16 Tensor Core projection for Qwen4's
unquantized fused GDN input, at one through five decode/verification rows.
Each weight tile serves all input rows. Four deterministic FP32 partial sums
are combined before rounding to BF16; ordinary one-token decode uses the same
reduction as MTP. Ordinary prefill and quantized GDN inputs keep their existing
paths. This option also applies with MTP disabled.

The shared projection can differ from the original cuBLAS result and therefore
change generated text. It is distinct from `FREETOKEN_MTP_BATCHED_LINEAR`, which
only changes non-greedy verification: shared GDN is enabled for both ordinary
decode and greedy/non-greedy MTP. It remains opt-in. Set these variables before
starting the process; restart to rebuild captured graphs after changing them.
See the [weight-reuse measurements](../benchmarks/results/mtp-reuse/implementation.md)
for throughput, transfer counts and numerical validation.

The former long-prompt divergence at output token 100 was traced to PLE convolution
rounding and fixed. Regression checks use 256 output tokens, forced rejection,
chunked prefill and concurrent requests. Quantized multi-row linear rounding has a
separate exact regression test. Validation is limited to the tested prompts,
hardware and checkpoint; other checkpoints and tensor parallelism are unvalidated.
See [validation and timings](../benchmarks/results/mtp-20260909/sampling-multi.md).
See the [performance follow-up](../benchmarks/results/mtp-20260909/performance.md)
for the later before/after measurements and graph/kernel validation.
The [chat workload measurements](../benchmarks/results/mtp-20260909/chat-performance.md)
use a chat template, sampling and a configured 1M-token capacity; they distinguish
the default calculation path from the optional BF16 experiment.
Batched verification across requests and prefix-cache reuse remain future work.
