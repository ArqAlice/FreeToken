# Deterministic MTP proposals with stochastic target sampling

Run date: 2026-09-10 JST (2026-09-09 UTC). Hardware: RTX 5090 32 GB,
Threadripper PRO 5965WX, driver 591.86, CUDA 13, WSL2/Docker.
Checkpoint: `aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP`, snapshot
`18bd7d69491d4709ac1933e95343e15f045ea21d`, from `freetoken_hf-cache`.

## Implementation and decision

`FREETOKEN_MTP_GREEDY_DRAFT=1` chooses argmax proposals for all draft steps.
The target keeps its original temperature, top-k, top-p and numerical kernels.
Each verification row is sampled independently from the filtered target
distribution. The matching prefix is accepted, followed by the first mismatch
or, after full acceptance, a bonus sample. Later samples are discarded after
rejection. Existing state commit, page release and draft repair use this prefix.

This is distribution-preserving verification for a deterministic proposal:
acceptance probability is `p(draft)`, and the rejected target sample is distributed
as `p` conditioned on excluding the draft. No dense one-hot proposal distribution
is needed. Draft softmax, filters, random draws and stored vocabulary probabilities
are eliminated. Greedy target requests retain their previous behavior.

**Keep the default at 0.** This checkpoint and chat workload became slower because
proposal acceptance fell. The option remains available for controlled experiments;
it is not recommended as a speed improvement for the user's settings. This result
does not justify changing target temperature or filters to increase acceptance.

## Real-model A/B

The baseline is the working branch immediately before this step, copied to
`/tmp/mtp-greedy-draft-before-python`. It already includes expert-transfer overlap.
It is not upstream `main`, which lacks this branch's MTP/checkpoint support.
[Source hashes](source.json) and the [runtime delta](runtime-change.patch) identify
the compared implementations.

Both processes use strict BF16 calculation order, Triton NVFP4, expert overlap,
3073 expert slots, 15629 KV pages, 1000128 sequence capacity, naive cache and two
request slots. Capacity is not prompt length: the cat prompt
`猫の魅力について熱くたくさん語って。` is 61 tokens with the chat template.
Temperature is 1, top-k 20, top-p .95. Each mode warms up for 128 output tokens,
then produces 512 tokens five times, seeds 1000-1004, alternating mode order.
Both runners reserve history for maximum depth 3 before warmup.

[before.json](before.json), [after.json](after.json), [summary.json](summary.json):
median decode tokens/s, excluding TTFT; parentheses show min-max of five runs.

| Mode | Stochastic draft | Greedy draft | Median change |
|---|---:|---:|---:|
| Off | 47.67 (46.28-48.72) | 47.78 (46.04-48.82) | +0.22% |
| Depth 1 | 38.16 (36.25-38.30) | 34.96 (34.17-37.00) | -8.39% |
| Depth 3 | 24.33 (23.63-25.46) | 23.39 (22.99-24.94) | -3.87% |

Aggregate accepted/proposed tokens across five runs:

| Depth | Stochastic draft | Greedy draft |
|---|---:|---:|
| 1 | 920/1633 (56.34%) | 799/1752 (45.61%) |
| 3 | 1166/4146 (28.12%) | 1047/4505 (23.24%) |

The off control is stable and all five off token sequences match exactly.
Proposal-policy changes consume different random
draws, so stochastic continuations differ; this A/B includes that workload
variation and is not a same-token timing comparison. Neither MTP variant beats
MTP off. CUDA allocated memory at run boundaries remains 28574390784 bytes and
history remains 456015876 bytes in both processes; no persistent memory reduction
was observed. These are offline engine measurements, not HTTP/OpenWebUI results.

Inside `freetoken-mtp-dev`, using `/opt/venv/bin/python`:

```sh
PYTHONPATH=/tmp/mtp-greedy-draft-before-python FREETOKEN_MTP_BATCHED_LINEAR=0 \
FREETOKEN_MTP_EXPERT_OVERLAP=1 python /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --modes 0,1,3 --tokens 512 --warmup-tokens 128 --repeats 5 \
  --output /tmp/mtp-greedy-draft-before.json

PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_BATCHED_LINEAR=0 \
FREETOKEN_MTP_EXPERT_OVERLAP=1 FREETOKEN_MTP_GREEDY_DRAFT=1 \
python /opt/freetoken/benchmarks/bench_mtp_chat.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --modes 0,1,3 --tokens 512 --warmup-tokens 128 --repeats 5 \
  --output /tmp/mtp-greedy-draft-after.json
```

## Validation

[Real-model regression](validation.json), with greedy draft enabled and depth 3:

- Both greedy prompts match MTP off exactly for all 256 output tokens.
- Forced rejection accepts 0/759 proposals and matches MTP off exactly.
- Chunked prefill matches MTP off; two concurrent requests match their separate runs.
- Sampling smoke checks complete two 64-token outputs each for `(temperature,
  top_k, top_p)` of `(.7, -1, 1)`, `(.8, 20, .9)` and `(1, 1, .8)`.
  These are execution checks, not a statistical model-quality evaluation.

Regression geometry is 8192 sequence capacity, 132 pages and 6000 expert slots,
matching the existing regression harness. Its timings are not the chat A/B above.

```sh
PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_GREEDY_DRAFT=1 \
FREETOKEN_MTP_BATCHED_LINEAR=0 python /opt/freetoken/benchmarks/bench_mtp.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --speculative-tokens 3 --tokens 256 --warmup-tokens 128 --repeats 1 \
  --validate --sampling-smoke --sampling-tokens 64 \
  --output /tmp/mtp-greedy-draft-validation.json
```

309 tests passed, including CUDA, with the new mode enabled (individual tests
explicitly select old/new policies where needed). Coverage includes depths 1-4,
every rejection position and full acceptance, page boundaries, target draws
different from argmax, absence of draft probability storage, graph replay and
cache rebuild. An exhaustive small-vocabulary test checks the joint target
distribution including continuation after a correction. Target filter-order and
tie tests run against FlashInfer and Triton sampling. A separate initial CPU-only
run passed 234 tests with 20 CUDA skips.

```sh
PYTHONPATH=/opt/freetoken/python FREETOKEN_MTP_GREEDY_DRAFT=1 python -m pytest \
  -p no:cacheprovider -q /opt/freetoken/tests/scheduler/test_mtp.py \
  /opt/freetoken/tests/scheduler/test_cache_rebuild.py \
  /opt/freetoken/tests/engine/test_speculative.py \
  /opt/freetoken/tests/moe/test_offload.py \
  /opt/freetoken/tests/models/qwen4_exp/test_skeleton.py
```

## Sampling microbenchmark

[sampling.json](sampling.json) measures synthetic 248320-token logits at the
same temperature/top-k/top-p. Five groups of 100 calls follow warmup. Each call
includes draft sampling, target probability construction, verification and the
accepted-length CPU read, but no model forward. Median host elapsed time per call:

| Draft depth | Stochastic proposals | Greedy proposals | Saved |
|---|---:|---:|---:|
| 1 | 931.42 us | 619.87 us | 311.56 us |
| 3 | 1349.49 us | 644.39 us | 705.10 us |

Sampling does become cheaper in isolation. This saving is small relative to a
full target/draft cycle, and the real-model acceptance drop offsets it. The
synthetic benchmark does not measure the exact per-cycle saving on the cat prompt.

```sh
PYTHONPATH=/opt/freetoken/python python /opt/freetoken/benchmarks/bench_mtp_sampling.py \
  --output /tmp/mtp-greedy-draft-sampling.json
```

Tensor parallelism, other checkpoints and long-context serving remain unvalidated.
The next planned optimization is draft-only QSA index reuse, evaluated separately
from this proposal-policy experiment.
