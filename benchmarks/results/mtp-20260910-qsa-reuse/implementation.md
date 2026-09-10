# Draft-only QSA selection reuse

Run date: 2026-09-10 JST (2026-09-09 UTC). Hardware: RTX 5090 32 GB,
Threadripper PRO 5965WX, driver 591.86, CUDA 13, WSL2/Docker.
Checkpoint: `aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP`, snapshot
`18bd7d69491d4709ac1933e95343e15f045ea21d`, in Docker volume `freetoken_hf-cache`.
Its QSA selection budget is 2048 tokens with compression ratio 4.

## Implementation

`FREETOKEN_MTP_QSA_REUSE=1` saves the last draft-extend row's selected compressed
blocks per MTP QSA layer, then reuses them for adjacent single-token lookahead.
The causal tail is expanded from the current position on every step. KV writes,
raw index-key writes and compression remain active. Reuse skips index-query
normalization/rope, compressed-block scores and block top-k. It retains the
combined index Q/K projection.

The scheduler refreshes selection at compression closure, ring wrap, request
change, configured depth change, block-table width change or nonadjacent position.
Draft repair always refreshes, and intermediate chunks without logits invalidate
ownership. Buffers have fixed addresses shared by refresh/reuse graphs; graph keys
distinguish those paths. Request cleanup clears ownership, and runner close/cache
rebuild releases graphs before buffers. Exceptions invalidate the cached owner.
Each MTP layer has a separate selection buffer; target batches do not receive it.

Only isolated QSA draft heads use this path, at configured depths 2-4. Depth 1
does not save selections. Interleaved requests refresh at ownership changes, so
one request cannot inherit another's block choices. Reduced remaining output
length can shorten lookahead without reusing any nonadjacent position.

For contexts within the budget all complete blocks are selected. Beyond it,
reusing a previous choice can change draft logits/probabilities; speculative
verification uses the actual proposal distribution. Target selection and sampling
parameters stay unchanged. Greedy target output remains checked against target
predictions, independently of draft accuracy.

The option remains an experiment, **disabled by default**. Eliminating draft
selection work alone did not improve the short chat's measured throughput.
The longer synthetic prompt improved by about 6%, but that comparison also
changes stochastic continuations and does not establish a general speedup.

## Short chat A/B

The baseline is the working branch before this step, saved at
`/tmp/mtp-qsa-reuse-before-python`. It includes the preceding overlap and optional
greedy-proposal work. It is not upstream `main`, which lacks this branch's
checkpoint/MTP support. [Source hashes](source.json) and the
[runtime delta](runtime-change.patch) identify the compared versions.

Both runs use strict BF16, Triton NVFP4, expert overlap, stochastic proposals,
3073 expert slots, 15629 KV pages, 1000128 sequence capacity, naive cache and two
request slots. Prompt: `猫の魅力について熱くたくさん語って。`, 61 input tokens after
the chat template. Temperature 1, top-k 20, top-p .95. Each mode warms for 128
tokens, then generates 512 tokens five times, seeds 1000-1004 with alternating
mode order. History capacity is set for maximum depth 3 before warmup.

[before.json](before.json), [after.json](after.json), [summary.json](summary.json):
median decode tokens/s excluding TTFT; parentheses show min-max of five runs.

| Mode | Before | Reuse enabled | Median change |
|---|---:|---:|---:|
| Off | 47.21 (46.38-48.77) | 47.76 (46.31-48.77) | +1.17% |
| Depth 1 | 38.12 (36.16-38.76) | 38.19 (36.40-38.78) | +0.17% |
| Depth 3 | 24.39 (23.65-25.50) | 24.37 (23.61-25.51) | -0.08% |

All 15 token sequences and proposal/acceptance counts match exactly. Depth 3
reuses selection for 1735 head calls and refreshes for 2415 calls. Depths 0/1 have
zero reuse/save calls. Aggregate acceptance stays 920/1633 at depth 1 and
1166/4146 at depth 3. There is no measured throughput improvement; control-mode
variation is larger than the depth-3 difference, and MTP off remains faster.

Allocated CUDA memory at run boundaries increases from 28574390784 to 28576069632
bytes (about 1.60 MiB), including the extra graph variants. The block buffer itself
is 2048 bytes for this checkpoint's single MTP layer. All modes share graphs warmed
at maximum depth 3; this memory delta does not describe a standalone depth-1 run.
History remains 456015876 bytes.

Inside `freetoken-mtp-dev`, using `/opt/venv/bin/python`:

```sh
PYTHONPATH=/tmp/mtp-qsa-reuse-before-python FREETOKEN_MTP_GREEDY_DRAFT=0 \
FREETOKEN_MTP_BATCHED_LINEAR=0 python /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --modes 0,1,3 --tokens 512 --warmup-tokens 128 --repeats 5 \
  --output /tmp/mtp-qsa-reuse-before.json

PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_QSA_REUSE=1 \
FREETOKEN_MTP_GREEDY_DRAFT=0 FREETOKEN_MTP_BATCHED_LINEAR=0 \
python /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --modes 0,1,3 --tokens 512 --warmup-tokens 128 --repeats 5 \
  --output /tmp/mtp-qsa-reuse-after.json
```

## Beyond the selection budget

[long-before.json](long-before.json), [long-after.json](long-after.json),
[long-summary.json](long-summary.json) use [long-prompt.txt](long-prompt.txt), a
synthetic repetition of a sentence about cats followed by the original question.
The chat template produces **3661 input tokens**, exceeding the 2048-token budget
from the start. Cache geometry, sampling and calculation order match the short
test. Each mode warms for 128 tokens then generates 256 tokens three times, seeds
1000-1002, with alternating mode order.

| Mode | Before | Reuse enabled | Median change |
|---|---:|---:|---:|
| Off | 41.16 tok/s | 40.81 tok/s | -0.86% |
| Depth 3 | 23.86 tok/s | 25.28 tok/s | +5.96% |

The three off sequences match. All three depth-3 continuations differ, as allowed
when changing draft probabilities and therefore rejection/sampling outcomes.
Aggregate acceptance is similar: 406/1070 (37.94%) before, 404/1073 (37.65%) after.
Reuse skips 468 selections, with 606 refreshed head calls. These measurements
include the resulting differences in expert routing and continuation content;
they do not isolate kernel time on identical tokens. MTP off is still faster.

The initial prompt was only 1861 tokens; that exploratory run was interrupted
and excluded. The reported runs use the longer prompt throughout.

```sh
PYTHONPATH=/tmp/mtp-qsa-reuse-before-python FREETOKEN_MTP_GREEDY_DRAFT=0 \
FREETOKEN_MTP_BATCHED_LINEAR=0 python /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --prompt-file /opt/freetoken/benchmarks/results/mtp-20260910-qsa-reuse/long-prompt.txt \
  --modes 0,3 --tokens 256 --warmup-tokens 128 --repeats 3 \
  --output /tmp/mtp-qsa-reuse-long-before.json

PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_QSA_REUSE=1 \
FREETOKEN_MTP_GREEDY_DRAFT=0 FREETOKEN_MTP_BATCHED_LINEAR=0 \
python /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --prompt-file /opt/freetoken/benchmarks/results/mtp-20260910-qsa-reuse/long-prompt.txt \
  --modes 0,3 --tokens 256 --warmup-tokens 128 --repeats 3 \
  --output /tmp/mtp-qsa-reuse-long-after.json
```

## Tests

The final code passes 303 tests with the option enabled. Tests cover all acceptance
lengths, request/depth/width changes, nonadjacent positions, compression closure,
ring wrap, repair, chunk invalidation, cleanup and cache rebuild. Actual QSA layer
tests cover BF16, FP8 and NVFP4 KV, dense and sparse selection budgets, index-key
updates during reuse, and CUDA graph replay with changed blocks, positions and
request slots. The single warning is an existing FlashInfer/CUTLASS deprecation.

```sh
PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_QSA_REUSE=1 python -m pytest \
  -p no:cacheprovider -q /opt/freetoken/tests/models/qwen4_exp/test_qsa_backend.py \
  /opt/freetoken/tests/scheduler/test_mtp.py \
  /opt/freetoken/tests/scheduler/test_cache_rebuild.py \
  /opt/freetoken/tests/engine/test_speculative.py
```

These measurements use the offline engine, not HTTP/OpenWebUI. Tensor parallelism,
other checkpoints and near-million-token contexts remain unvalidated.

## Real-model regression

[validation.json](validation.json) verifies the enabled path at depth 3:

- Both ordinary greedy prompts match MTP off exactly for 256 output tokens.
- Forced rejection accepts 0/759 proposals and matches MTP off exactly.
- The longer prompt, split into 512-token prefill chunks, matches MTP off for all
  256 output tokens. This comparison uses the raw prompt rather than the chat
  template; the exact input is the linked `long-prompt.txt`.
- Two concurrent requests match their separate runs, exercising owner changes.
- Each sampling setting `(.7, -1, 1)`, `(.8, 20, .9)` and `(1, 1, .8)` completes
  two 64-token outputs. These are execution checks, not a model-quality study.

Regression geometry is 8192 sequence capacity, 132 pages and 6000 expert slots.
Its timings should not be compared with the fixed-cache chat benchmark.

```sh
PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_QSA_REUSE=1 \
FREETOKEN_MTP_GREEDY_DRAFT=0 FREETOKEN_MTP_BATCHED_LINEAR=0 \
python /opt/freetoken/benchmarks/bench_mtp.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --speculative-tokens 3 --tokens 256 --warmup-tokens 128 --repeats 1 \
  --validate --sampling-smoke --sampling-tokens 64 --max-extend-tokens 512 \
  --validation-prompt-file /opt/freetoken/benchmarks/results/mtp-20260910-qsa-reuse/long-prompt.txt \
  --output /tmp/mtp-qsa-reuse-validation.json
```
