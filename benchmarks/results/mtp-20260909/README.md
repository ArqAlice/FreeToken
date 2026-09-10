# MTP validation, 2026-09-09

Historical initial results below. See [sampling and multi-token validation](sampling-multi.md)
for the current implementation: tested greedy output parity is fixed; a speedup is
still not established.

Hardware: NVIDIA GeForce RTX 5090, 32607 MiB, driver 591.86, Docker image
`freetokenfp8-local:cu130`. Checkpoint:
`aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP`, from `freetoken_hf-cache`.

Server command (container port 1919 published at host port 1921):

```sh
ft serve --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --host 0.0.0.0 --port 1919 --max-running-requests 2 \
  --max-seq-len-override 8192 --kv-reserve-tokens 8192 \
  --kv-cache-dtype nvfp4 --max-prefill-length 128 --mtp-speculative-tokens 1
```

The comparison uses the same branch and model with `--mtp-speculative-tokens 0
--cache-type naive`. Auto MoE cache sizing was enabled. This is a smoke comparison,
not a controlled benchmark against main. Timings include prefill and HTTP overhead;
generated content differs between the execution paths.

| Prompt | MTP tokens / seconds | Disabled tokens / seconds |
| --- | --- | --- |
| The capital of France is | 64 / 7.497 | 23 / 3.568 (EOS) |
| Write a Python function that computes Fibonacci numbers. | 64 / 7.174 | 64 / 3.835 |

Requests used `/v1/completions`, `temperature=0`, `top_p=1`, `top_k=-1`,
`max_tokens=64`. Final responses are in `mtp-validation-cached.json`; the disabled
responses are in `mtp-validation-off.json`. The final server log is
`mtp-validation-cached.log` (local, ignored by Git). Its first eight-token warmup
contains diagnostic replay; that diagnostic was removed from the final source and
was no longer active during the timed requests.

Same-state joint versus sequential forward diagnostics found near-tied leading
logits changing the greedy choice (one joint result had two leading logits both
15.5625). This is evidence of numerical path sensitivity, not proof that every
remaining output difference is benign. Strict parity needs further investigation.

Additional real-model checks:

- Both accepted and rejected drafts, and generation crossing a 64-token KV page.
- 145-token prompt split at 128 tokens; four output tokens streamed in order,
  followed by `finish_reason=length` and `[DONE]` (`mtp-validation-stream.txt`).
- `stop="."` terminates after ` Paris`, suppressing the speculative successor
  (`mtp-validation-stop.json`).

Regression checks: 206 passed with:

```sh
/opt/venv/bin/python -m pytest tests/scheduler tests/engine/test_kv_quant_config.py \
  tests/tokenizer/test_detokenize.py tests/models/qwen4_exp/test_config.py \
  tests/models/qwen4_exp/test_weight.py tests/models/qwen4_exp/test_ple_disk.py \
  tests/kvcache/test_qsa_pool.py tests/moe/test_offload.py -q -p no:cacheprovider
```

One additional GPU test passed:
`pytest tests/models/qwen4_exp/test_gdn.py -k two_token -q -p no:cacheprovider`.
It compares two-token GDN continuation against sequential decode at the existing
2e-2 tolerance. No main-branch benchmark or stochastic speculative validation ran.

Earlier `on`, `fixed`, `eager`, and `diagnostic` artifacts record intermediate
investigation, including failures before the host PLE input fix and the expert-cache
change. They are not results for the final implementation.
