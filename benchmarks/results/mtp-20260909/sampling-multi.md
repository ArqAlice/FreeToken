# MTP sampling and multi-token validation

## Changes

- Fixed the greedy long-prompt divergence at output token 100. PLE continuation
  convolution now uses the same FP32 tap reduction as single-token decode.
- Added stochastic speculative acceptance with temperature, top-k and top-p.
  Rejection uses the normalized positive difference between target and draft
  distributions; full acceptance samples a target bonus token.
- Added 1-4 draft tokens by feeding the MTP head's pre-final-mixer residual into
  the next draft step. Partial acceptance restores recurrent/index-ring state,
  replays the accepted input prefix, frees unused pages and rebuilds draft state
  from target residuals. Generation and KV capacity bound the lookahead.
- Extended target graphs to 1-5 input rows. BF16/NVFP4 target linears preserve
  single-row rounding. QSA verification keeps the single-token attention split-K
  profile, avoiding a different reduction tree when the row count increases.

The QSA regression reproduces an exact-equality failure with the old profile
(627/5120 and 5038/20480 differing output elements for two test geometries),
and passes with the decode profile. The NVFP4 linear regression also failed
before its rowwise fix and passes afterward.

## Hardware and commands

RTX 5090, 32607 MiB, driver 591.86, Docker image `freetokenfp8-local:cu130`.
Checkpoint `aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP` resides in the Docker
volume `freetoken_hf-cache`. The benchmark uses 6000 MoE expert slots, 132 KV pages,
NVFP4 KV, naive cache, no overlap, two request slots, maximum sequence length
8192 and prefill chunks of 128 tokens.

```sh
docker exec freetoken-mtp-dev /opt/venv/bin/python \
  /opt/freetoken/benchmarks/bench_mtp.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --speculative-tokens 4 --sampling-smoke --validate \
  --tokens 256 --repeats 1 --output /tmp/mtp-multi4-final.json
```

Use `--speculative-tokens 2` to repeat at depth 2. Greedy validation requests 256
output tokens with `ignore_eos=True`; each stochastic smoke setting requests 64.
The three stochastic settings are `(temperature, top_k, top_p)` = `(0.7, -1, 1)`,
`(0.8, 20, 0.9)` and `(1, 1, 0.8)`. All run both prompts concurrently.

These are same-branch MTP-enabled/disabled comparisons, not upstream-main A/B
numbers. Model loading and the initial 8-token warmup are excluded; prompt
processing is included. Lazy graph shapes not exercised by warmup can still add
capture cost. One timing sample per prompt is a validation measurement, not a
claim of a stable performance improvement.

## Earlier evidence

[mtp-ple-fixed.json](mtp-ple-fixed.json) records 256-token greedy parity for the
two ordinary prompts, forced rejection (0/254), the 145-token chunked prompt and
concurrent generation after the PLE fix.
[mtp-sampling1.json](mtp-sampling1.json) records the one-token stochastic smoke
runs with accepted/proposed counts 57/69, 58/67 and 57/67, and passing greedy checks.
These predate the subsequent multi-token changes.

[mtp-multi4-state.json](mtp-multi4-state.json) is a diagnostic failure from BEFORE
the QSA profile fix. It reproduces the difference with graphs disabled and first
finds target recurrent-state differences after cached position 33, at the layer
following the first QSA attention layer. [mtp-diag5.json](mtp-diag5.json) shows
that early five-row verification can still match all layer outputs exactly; it
is not evidence that all positions matched before the fix.

## Final depth-4 real-model results

[mtp-multi4-final.json](mtp-multi4-final.json): both ordinary prompts, forced
rejection, the 145-token chunked prompt and concurrent-versus-individual requests
match the greedy baseline for 256 generated tokens. Across this run, every prefix
length occurred: 0 accepted in 342 verifications, 1 in 117, 2 in 73, 3 in 106 and
4 in 239. This includes warmup, forced-rejection and stochastic runs, so it is not
an acceptance-rate measurement for a single workload.

| Prompt (256 output tokens) | MTP disabled | 4 drafts | Accepted / proposed |
| --- | ---: | ---: | ---: |
| The capital of France is | 7.162 s | 10.097 s | 188 / 264 |
| Write a Python function that computes Fibonacci numbers. | 7.290 s | 9.928 s | 183 / 284 |

The three stochastic settings completed two 64-token outputs each. Their accepted /
proposed counts were 89/143, 85/160 and 84/163. These are runtime smoke checks;
distribution preservation is tested analytically and statistically in unit tests.
Sampled token identity with the greedy baseline is not an acceptance criterion.

The depth-4 path is slower than ordinary decoding on these measurements. Serial
requests, eager draft calls, per-row dense arithmetic and rejection replay remain
costs. Increasing the lookahead should be evaluated per workload.

## Final depth-2 real-model results

[mtp-multi2-final.json](mtp-multi2-final.json) passes the same 256-token greedy
checks: both ordinary prompts, forced rejection, chunked prefill and concurrent
requests. All three stochastic settings complete both 64-token outputs, with
accepted/proposed counts 71/106, 70/107 and 73/102.

| Prompt (256 output tokens) | MTP disabled | 2 drafts | Accepted / proposed |
| --- | ---: | ---: | ---: |
| The capital of France is | 7.187 s | 8.615 s | 160 / 190 |
| Write a Python function that computes Fibonacci numbers. | 7.284 s | 8.725 s | 157 / 196 |

Depth 2 was faster than depth 4 in these runs but still slower than MTP-disabled
decoding. These timings do not establish a speedup.

## Regression command

```sh
docker exec freetoken-mtp-dev /opt/venv/bin/python -m pytest \
  tests/scheduler \
  tests/engine/test_speculative.py tests/engine/test_kv_quant_config.py \
  tests/layers/test_linear_mtp.py \
  tests/models/qwen4_exp/test_config.py tests/models/qwen4_exp/test_gdn.py \
  tests/models/qwen4_exp/test_ple.py tests/models/qwen4_exp/test_ple_disk.py \
  tests/models/qwen4_exp/test_skeleton.py tests/models/qwen4_exp/test_weight.py \
  tests/models/qwen4_exp/test_qsa_backend.py tests/models/qwen4_exp/test_qsa_kernels.py \
  tests/kvcache/test_qsa_pool.py tests/kvcache/test_qsa_pool_fp8.py \
  tests/kernels/test_qsa_fp8.py tests/kernels/test_qsa_nvfp4.py \
  tests/tokenizer/test_detokenize.py -q -p no:cacheprovider
```

Coverage includes acceptance and residual-distribution mathematics, empirical
sampling frequencies, both sampling backends' filter order/ties, all accepted
prefix lengths at depths 2 and 4, page-boundary rollback, no-free-page fallback,
head-residual alignment, graph replay, quantized QSA attention and host-side
EOS/output-budget handling.

Final regression result: **368 passed, 4 skipped** (20.55 seconds).
The optional reference/table checks were skipped because their required environment
variables were unset. This is the targeted regression selection above, not the
entire repository suite. The test-only QSA batch fixtures were updated to carry
the new Batch flag. `git diff --check` also passed.
