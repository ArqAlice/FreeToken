# Greedy one-token MTP follow-up

Historical parity investigation. [The subsequent CUDA Graph work](graphs.md) reduces
the MTP latency reported below while retaining the tested output parity.

The tested output differences are fixed. End-to-end acceleration is not achieved.
Tests used an RTX 5090 (32607 MiB), driver 591.86, Docker image
`freetokenfp8-local:cu130`, and
`aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP` from `freetoken_hf-cache`.

## Changes and diagnosis

Same-state, teacher-forced comparison localized divergence to tiny numerical
differences accumulating through the target layers. Using the fused decode GDN
recurrence fixed the single-token prefill/decode difference. Running two-token
target linear projections row by row fixed the remaining joint/sequential GEMM
rounding differences. The measured two-row target logits then matched sequential
decode exactly (maximum absolute difference 0 for both rows).

The MTP hidden normalization now spans all residual streams, matching the
[reference implementation](https://docs.vllm.ai/en/latest/api/vllm/models/qwen4_exp/nvidia/mtp/).
Draft preparation reuses matching target metadata and avoids unnecessary CPU token
copies. Per-step logs are debug-level. Tiny GDN verification uses the fused recurrence
instead of chunk-prefill kernels. Row-wise target projections preserve tested parity
but add launches; these changes do not establish an overall speedup.

## Reproduction

The harness uses one loaded model for both paths, fixed 64-token output budgets
(`ignore_eos=True`), greedy sampling, naive KV cache, disabled overlap, 6000 MoE expert
slots, 132 pages, maximum sequence length 8192 and prefill chunks of 128 tokens.
The baseline is this branch with MTP disabled, not upstream main.

```sh
docker exec freetoken-mtp-dev /opt/venv/bin/python /opt/freetoken/benchmarks/bench_mtp.py \
  --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --compare-draft-norms --validate --repeats 1 --tokens 64 \
  --output /tmp/mtp-norm.json
```

Use `--diagnose --validate --repeats 2 --tokens 64` for layer diagnostics and repeated
parity tests; `--profile --repeats 1 --tokens 16` for CPU/CUDA profiling.

## Results

Final source results are in [mtp-norm-optimization.json](mtp-norm-optimization.json).
Times include prefill and generation, excluding model load and warmup.

| Prompt | MTP off, seconds | MTP on, seconds | Accepted / proposed | Token parity |
| --- | ---: | ---: | ---: | --- |
| The capital of France is | 3.836 | 7.374 | 30 / 32 | exact |
| Write a Python function that computes Fibonacci numbers. | 3.750 | 7.722 | 29 / 33 | exact |

Both output 64 tokens. Final-source forced rejection (0 accepted of 62), chunked
145-token prompt and simultaneous-versus-individual requests also passed.
[mtp-parity-64.json](mtp-parity-64.json) records the preceding row-wise fix with two
repeats per prompt and the same checks; it predates the draft normalization fix.

The same-process normalization comparison moved France acceptance from 29/33 to
30/32, but Python moved from 30/32 to 29/33. Aggregate speed did not improve, so this
is a correctness fix rather than a demonstrated performance gain.

The focused regression suite passed 211 tests (6 deselected); the subsequent global
hidden-normalization GPU regression passed separately. This is not a full-suite run.

## Remaining performance work

[mtp-profile.json](mtp-profile.json) contains a 16-token profile including prefill,
captured before the normalization change. CUDA copies dominate device time, while
host synchronization and thousands of kernel launches dominate CPU time. Prefill
expert transfers are included, so this profile cannot isolate decode-only costs.
The next measurement should separate prefill, draft, target verification, state
snapshot and rejection replay. Dedicated verification CUDA graphs and reduced
launch overhead are candidates; both require preserving the parity checks above.

Eager serial verification, state snapshots and rejection replay remain. Longer
contexts, broader prompts, other hardware/checkpoints and tensor parallelism remain
unvalidated. Non-greedy speculative acceptance and multi-token drafts are outside
this change.
