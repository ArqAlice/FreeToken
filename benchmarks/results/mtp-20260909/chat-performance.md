# MTP chat workload performance

This follow-up measures the Japanese chat workload with the server's configured
1M-token capacity. The earlier [performance report](performance.md) used short
plain-text prompts, greedy decoding, 132 KV pages and 6000 MoE expert slots.
Those measurements answer a different workload question.

## Changes

- Target verification saves the GDN recurrence after each input token using the
  existing FLA intermediate-state interface. Rejection commits the accepted
  prefix directly. GDN and PLE convolution tails are rebuilt from saved raw
  inputs; the QSA ring combines accepted writes with the preceding ring state.
  Accepted target residuals are reused without a second target forward.
- The checkpoint has 36 GDN layers with 48 value heads and 128-by-128 FP32 states:
  108 MiB per saved prefix. A single five-step buffer uses 540 MiB, is allocated
  lazily, and is shared by all sequential requests and graph shapes. Convolution
  input buffers add about 3.6 MiB. `FREETOKEN_MTP_STATE_HISTORY=0` selects the
  earlier replay path for comparison.
- QSA prefill and speculative score work uses a power-of-two active page bucket,
  starting at 128 pages / 8192 tokens. At this workload's short active prefix,
  scoring uses 2048 columns instead of the 250032 columns implied by the
  1000128-token capacity. Target and draft graph keys include the bucket width.
  Existing score buffers supply prefix views. The torch.topk fallback keeps its
  original width because padding can affect tie ordering.
- Wide NVFP4 projections reuse unpacked weights across input rows while retaining
  the previous single-row multiplication and reduction. Small projections keep
  independent row programs. The default switch is
  `FREETOKEN_MTP_NVFP4_SHARED_ROWS=1`.
- Routed NVFP4 experts use arithmetic weight unpacking with the same rounded
  scales and reconstructed products as the lookup-table path.
  `FREETOKEN_NVFP4_MOE_ARITHMETIC=1` enables it by default. It also applies to
  ordinary decoding, so final comparisons must remeasure the MTP-disabled mode.
- Acceptance-prefix reduction and selection of the correction distribution run
  on the device. The scheduler reads the accepted length once for state and page
  management. Probability inspection does not draw additional samples.
- Server decode logs count emitted tokens rather than one token per request per
  forward. Padding, aborted/previously finished requests, and tokens discarded
  after EOS or an output limit are excluded. This fixes the server log's MTP
  undercount; OpenWebUI has its own delivery-based measurement.

The default BF16 path preserves the single-row calculation. The optional
`FREETOKEN_MTP_BATCHED_LINEAR=1` uses ordinary batched BF16 projections for
non-greedy requests. Changed accumulation and rounding can change logits and
top-k/top-p membership. This experimental option remains off by default, and
greedy requests retain the strict path. Its performance results are separated
below rather than treated as an exact replacement for the default computation.

## Workload and measurement

Hardware: NVIDIA RTX 5090 (32607 MiB), driver 591.86, Threadripper PRO 5965WX
(24 cores / 48 threads), WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`.
Docker image: `freetokenfp8-local:cu130`. Checkpoint:
`aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP`, stored in `freetoken_hf-cache`.
FreeToken version 0.1.2, branch `feat/mtp-support`, base commit `3381015`, with
the uncommitted local MTP implementation. The workspace source is mounted over
the development image; these measurements do not imply a rebuilt production image.

- Prompt: `猫の魅力について熱くたくさん語って。`, with the checkpoint's chat template
  and generation prompt; 61 input tokens.
- Sampling: temperature 1, top-k 20, top-p 0.95, `ignore_eos=True`.
- 512 output tokens, 128 warmup tokens per mode, two primary repeats in the
  intermediate JSON files below. Mode order alternates between repeats.
- Maximum sequence capacity 1000128, 15629 KV pages, NVFP4 KV storage, 3073
  **expert cache slots**, two request slots, naive cache, no overlap scheduling,
  and an 8192-token prefill chunk limit. Expert slots are not MiB.

The configured capacity is 1M tokens; this is not a 1M-token input benchmark.
Model loading and warmup are excluded from primary times. `decode_tokens_per_second`
is `(output_tokens - 1) / (last_delivery - first_delivery)`. TTFT is measured at
offline token delivery and includes tokenization/scheduling. HTTP transport and
OpenWebUI rendering are outside this harness. The earlier JSON files also contain a
separate profiling generation after all primary runs; primary values do not
include profiler instrumentation.

These are before/after comparisons on the local MTP branch. They do not establish
an upstream-main speedup. Sampling acceptance and numerical differences can
change the generated continuation, so experimental-mode differences are workload
results rather than an isolated kernel A/B.

## Recorded intermediate results

Arithmetic mean decode tokens/s across two 512-token requests per mode. The
state-history column records an intermediate implementation before the final
combined kernel changes; it is not the final default-path measurement.

| Draft tokens | Before | State history, strict BF16 |
|---|---:|---:|
| 0 | 47.80 | 47.93 |
| 1 | Not run | Not run |
| 2 | Not run | Not run |
| 3 | 20.85 | 23.98 |

Sources: [before](chat-before.json), [state-history stage](chat-state-history.json).
For depth 3, mean total time fell from 27.023 to 23.833 seconds; mean TTFT remained
2.505-2.506 seconds. Both stages accepted 491 of 1586 proposed draft tokens
(31.0%). All four matching mode/repeat token sequences are identical between
these two saved files, including the MTP-disabled requests.

The separate optional batched BF16 experiment produced:

| Draft tokens | Decode tokens/s | Total seconds | Accepted / proposed |
|---|---:|---:|---:|
| 0 | 47.82 | 13.184 | 0 / 0 |
| 1 | 39.70 | 15.375 | 365 / 656 |
| 2 | 31.86 | 18.550 | 467 / 1105 |
| 3 | 29.84 | 19.640 | 489 / 1595 |

Source: [experimental batched BF16 stage](chat-batched.json). These samples use
`FREETOKEN_MTP_BATCHED_LINEAR=1`. They precede the final combined default kernel
run. The earlier harness did not record environment variables in these JSON
settings; the new harness records the relevant switches for subsequent runs.
At this stage, ordinary decoding remains faster than every measured MTP depth.

## Kernel evidence

The [NVFP4 shared-row microbenchmark](nvfp4-shared-rows.json) records zero output
mismatches in all 12 synthetic cases. Selected CUDA Graph times are microseconds:

| Rows | Input width | Output width | Independent row grid | Shared weights |
|---|---:|---:|---:|---:|
| 4 | 2560 | 10240 | 18.34 | 11.80 |
| 4 | 2560 | 248320 | 862.01 | 287.77 |
| 5 | 2560 | 248320 | 1096.00 | 323.02 |

Sharing was slower at output width 512, which is why the engine retains the
independent row grid for narrow projections. This benchmark uses resident
synthetic weights and does not include routed-expert transfers or model work.

The [NVFP4 expert arithmetic microbenchmark](nvfp4-moe-fp16-bits.json) records
exactly equal outputs in all 16 shape/routing cases. For four rows, eight experts,
and a 2560-to-3072 projection with shared routing, kernel time changes from
100.73 to 85.02 microseconds. An earlier
[direct arithmetic experiment](nvfp4-moe-direct-bits.json) is retained as an
intermediate artifact; the FP16-rounding-preserving variant supplies this claim.

## Reproduction

Final default calculation path, with modes 0, 1 and 3 compared against the same
loaded model:

```sh
docker exec -e PYTHONPATH=/opt/freetoken/python -e FREETOKEN_MTP_BATCHED_LINEAR=0 \
  freetoken-mtp-dev \
  /opt/venv/bin/python /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --modes 0,1,3 --max-seq-len 1000128 --pages 15629 --moe-slots 3073 \
  --temperature 1 --top-k 20 --top-p .95 --warmup-tokens 128 \
  --tokens 512 --repeats 2 --output /tmp/mtp-chat-final-strict.json
```

Optional batched BF16 primary run:

```sh
docker exec -e PYTHONPATH=/opt/freetoken/python -e FREETOKEN_MTP_BATCHED_LINEAR=1 \
  freetoken-mtp-dev \
  /opt/venv/bin/python /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --modes 0,1,3 --max-seq-len 1000128 --pages 15629 --moe-slots 3073 \
  --temperature 1 --top-k 20 --top-p .95 --warmup-tokens 128 \
  --tokens 512 --repeats 2 --output /tmp/mtp-chat-final.json
```

Adding `--compare-batched-linear` runs a separate 128-token generation at
temperature 1, top-k 20 and top-p
0.95 after primary timing. Its first 16 target verifications evaluate both paths
from the same saved state, then continue generation using the batched path.
`batched_linear_comparison` reports per-row and aggregate total variation for
full softmax and the filtered sampling distribution, top-1 agreement and maximum
logit difference. Probabilities are computed in CPU FP32 with threshold filters
that retain boundary ties; this isolates the comparison from device sampling
workspaces. Comparing probabilities makes no additional random draws.
The diagnostic restores its wrapper, environment, mode and random state afterward.
Use `--profile` for an additional instrumented generation after primary timing.

## Final combined result

The final runs enable the shared-row and arithmetic NVFP4 kernels and state
history. Both the [default BF16 path](chat-final-strict.json) and the
[optional batched BF16 path](chat-final.json) use two 512-token requests per mode
under the same 1M-capacity settings above. Values are arithmetic means.

| Draft tokens | Default decode tokens/s | Individual runs | Optional decode tokens/s | Individual runs |
|---|---:|---|---:|---|
| 0 | 47.68 | 48.20, 47.17 | 48.03 | 48.67, 47.39 |
| 1 | 36.29 | 35.40, 37.18 | 39.96 | 39.84, 40.09 |
| 3 | 24.17 | 24.52, 23.82 | 30.07 | 30.49, 29.64 |

At depth 3 the final default path is about 16% faster than the earlier 20.85
tokens/s, and the combined optional path is about 44% faster. Both remain slower
than MTP-disabled generation. Depth 1 is the fastest measured MTP setting.
The optional BF16 change is not enabled by default; do not attribute its gain to
the default implementation. At depth 0 the option has no effect, and the small
difference between columns is run-to-run variation.
All four matching mode/repeat token sequences in the before and final-default
files are identical. Within the optional path, all six matching sequences are
also identical before and after enabling the arithmetic MoE kernel. These checks
do not imply identical continuations between optional and default BF16 paths.

The remaining cost is principally target work and offloaded expert transfer.
In the saved 128-token profiles, removing accepted-prefix replay reduces target
calls from 86 to 51, while the routed-expert copy kernels still total about
2.25 seconds. The full prefill transfer adds about 2.45 seconds in both profiles.
These instrumented figures identify costs; they are not additional throughput
measurements. Low draft acceptance makes deeper speculative batches particularly
expensive in this workload.

## Validation

[GPU regression](chat-regression.txt): **546 passed, 8 skipped**. The command
covers scheduler status/termination/abort/rebuild, speculative sampling,
BF16/NVFP4 linears and experts, Qwen4 GDN/PLE/QSA, KV quantization, and
detokenization. New history tests check every accepted prefix, nonzero state
slots, ring wrapping, shared buffer reuse and graph replay.

[Real-model regression](chat-greedy.json): both ordinary 256-token greedy
requests match MTP-disabled output at depth 4. Forced first-proposal rejection
also matches (0 accepted / 1010 proposed), as do chunked prefill and concurrent
versus individual requests. Three sampling configurations each complete two
64-token requests: `(0.7, -1, 1)`, `(0.8, 20, 0.9)` and `(1, 1, 0.8)` for
temperature, top-k and top-p. This run uses the earlier regression geometry
(8192 context, 132 pages, 6000 expert slots, 128-token prefill chunks), not the
1M-capacity speed configuration.

```sh
docker exec -e PYTHONPATH=/opt/freetoken/python -e FREETOKEN_MTP_BATCHED_LINEAR=1 \
  freetoken-mtp-dev /opt/venv/bin/python /opt/freetoken/benchmarks/bench_mtp.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --speculative-tokens 4 --tokens 256 --warmup-tokens 128 --repeats 1 \
  --validate --sampling-smoke --sampling-tokens 64 --output /tmp/mtp-chat-greedy.json
```

The [same-state numerical diagnostic](chat-numerics.json) compares 16
verifications / 64 rows. Batched versus strict BF16 has mean softmax total
variation 0.01595 and mean CPU-reference filtered variation 0.01696
(maximum 0.12623).
Top-1 agrees in all 64 rows; maximum absolute logit difference is 1.78125.
This is a small numerical probe, not a quality evaluation or proof of identical
sampling distributions. The initial GPU-only diagnostic stalled at readback;
the recorded successful diagnostic copies each target result to CPU before
evaluating probabilities. Primary timing was completed and saved separately.

The [HTTP streaming run](chat-http.json) uses depth 1 with optional batched BF16
enabled. Requests omit temperature, top-k and top-p; the server resolves these
to 1, 20 and 0.95. After 128 warmup tokens, both measured requests return 512
completion tokens, 61 prompt tokens and `finish_reason: "length"`, with streamed
reasoning and Japanese answer text. Delivery throughput is 43.65 and 41.77
tokens/s, and TTFT is 2.515 and 2.507 seconds. The numerator includes all
completion tokens; first/last nonempty SSE text chunks define the interval.
This exercises the chat API used by OpenWebUI, without the UI itself.

The server automatically allocates **3475 expert slots**, compared with the
offline benchmark's fixed 3073; both use 15629 KV pages. These HTTP samples
validate serving and provide an additional observed speed, not a same-cache A/B
against the offline table or the user's earlier OpenWebUI reading.

The server and client ran inside the dedicated development container with the
workspace source mounted at `/opt/freetoken`:

```sh
PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_BATCHED_LINEAR=1 /opt/venv/bin/ft serve \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --host 127.0.0.1 --port 19191 --max-running-requests 2 \
  --max-seq-len-override 1000128 --kv-reserve-tokens 1000128 \
  --kv-cache-dtype nvfp4 --mtp-speculative-tokens 1 --max-prefill-length 8192

/opt/venv/bin/python /opt/freetoken/benchmarks/bench_mtp_http.py \
  --url http://127.0.0.1:19191 \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --tokens 512 --warmup-tokens 128 --repeats 2 --output /tmp/mtp-chat-http.json
```
