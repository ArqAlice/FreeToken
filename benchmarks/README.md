# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

**`bench_mtp.py`** compares MTP-disabled generation with 1-4 draft tokens using
the same loaded checkpoint. Use a long warmup to exercise lazy graph shapes.
JSON records total time, offline first-token latency and time between first and
last token delivery. `--timings` adds nested CPU/CUDA intervals for diagnosis;
use a separate run without it for speed comparisons. `--validate` checks greedy
parity through rejection, chunked prompts and concurrent requests;
`--sampling-smoke` exercises temperature, top-k and top-p.

```bash
python benchmarks/bench_mtp.py --model /path/to/model --speculative-tokens 2 \
  --warmup-tokens 256 --tokens 256 --repeats 3 --validate --sampling-smoke \
  --output /tmp/mtp.json
```

**`bench_mtp_chat.py`** compares MTP depths with the checkpoint's chat template
and a configured 1M-token capacity. It records first/last token delivery, decode
tokens/s, accepted/proposed counts and MTP environment settings. `--moe-slots`
counts expert cache entries. Warmup precedes the timed runs, and alternating the
mode order reduces ordering bias. These are offline measurements; HTTP delivery
and OpenWebUI are outside this harness.

```bash
python benchmarks/bench_mtp_chat.py --model /path/to/model --modes 0,1,2,3 \
  --max-seq-len 1000128 --pages 15629 --moe-slots 3073 \
  --temperature 1 --top-k 20 --top-p .95 --warmup-tokens 128 \
  --tokens 512 --repeats 3 --output /tmp/mtp-chat.json
```

`--profile` runs a separate instrumented generation after the primary samples.
`--profile-trace /tmp/mtp-trace.json` also exports its Chrome trace, including
CUDA stream events, for checking whether expert fetch overlaps shared-expert
computation. Compare `FREETOKEN_MTP_EXPERT_OVERLAP=0` against the default `1` in
separate processes with the same model, cache budget and sampling settings.
`--nvfp4-backend triton|flashinfer|auto` compares existing expert backends with
identical cache capacity. Backend changes can change rounding; they are not an
output-equivalence claim. Each timed run also records acceptance-length counts,
history buffer bytes and allocated CUDA memory.

`--diagnostics --repeats 0` enables expert statistics before CUDA graph capture
and measures a separate instrumented run for each positive mode. It records per-cycle target, draft,
lookahead, sampling and state-copy intervals, and estimates packed expert fetch
bytes from cache misses. Prefill and graph-capture cycles are separated from
steady decode. Nested intervals overlap; do not sum them or use diagnostic
throughput as the uninstrumented A/B result.

```bash
python benchmarks/bench_mtp_chat.py --model /path/to/model --modes 3 \
  --diagnostics --repeats 0 --tokens 128 --output /tmp/mtp-diagnostics.json
```

`--compare-batched-linear` adds a separate 128-token diagnostic at temperature 1,
top-k 20 and top-p .95. It compares the first 16 target verifications from identical
saved states, recording full-softmax and filtered-distribution total variation,
top-1 agreement and maximum logit differences. The diagnostic restores its
wrapper, environment settings and random state afterward.

```bash
FREETOKEN_MTP_BATCHED_LINEAR=1 python benchmarks/bench_mtp_chat.py \
  --model /path/to/model --modes 0,1,2,3 --tokens 512 --repeats 3 \
  --compare-batched-linear --output /tmp/mtp-chat-batched.json
```

The batched BF16 option is disabled by default because it can change rounding and
the sampling distribution. Greedy requests retain the strict path. See
[chat performance results](results/mtp-20260909/chat-performance.md) for the
separate default-path and experimental measurements.

`FREETOKEN_MTP_GREEDY_DRAFT=1` compares argmax draft proposals against the default
stochastic proposals without changing target sampling settings. Run each setting
in a fresh process with the same modes and cache geometry. Compare throughput and
acceptance together; equal seeds do not imply equal stochastic output after a
proposal-policy change. `bench_mtp_sampling.py` also reports proposal/verification
sampling time for depths 1 and 3 on synthetic 248320-token logits, excluding model
forwards. This microbenchmark is not end-to-end throughput.

```bash
FREETOKEN_MTP_GREEDY_DRAFT=1 python benchmarks/bench_mtp_chat.py \
  --model /path/to/model --modes 0,1,3 --tokens 512 --repeats 5 \
  --output /tmp/mtp-chat-greedy-draft.json
```

`FREETOKEN_MTP_QSA_REUSE=1` independently enables draft-only QSA block reuse at
depths 2-4. The chat benchmark records refreshed/reused head counts. Use
`--prompt-file /path/to/prompt.txt` to test a UTF-8 prompt longer than the QSA
selection budget, and report the resulting `input_tokens`. The resolved prompt is
stored in the output JSON. Compare acceptance and throughput; sparse-context draft
probabilities can differ even though target selection is unchanged.

**`bench_mtp_http.py`** measures streamed chat delivery against a running server.
It omits temperature, top-k and top-p so the server defaults apply. The JSON
includes the exact request, first/last text delivery times, token usage, finish
reasons and generated text. Throughput includes reasoning tokens reported in
completion usage; SSE chunks can contain multiple tokens, so this is a delivery
metric rather than a per-token trace. Configure MTP on the server before running.
`--validate` additionally checks a stop string, recovery after closing a stream,
and two simultaneous streaming requests, including usage and terminal events.

For automatic-depth comparisons, add `--mtp-auto` to `bench_mtp_chat.py` and
keep a positive maximum in `--modes 0,3`. The result includes time and token counts
by depth, zero-depth draft-maintenance overhead and graph-capture time. The zero
mode uses the same loaded model/cache geometry. `--compare-ple --modes 1,3`
alternates synchronous/asynchronous disk PLE for each fixed depth and seed.
`bench_mtp.py --mtp-auto --validate --validation-contexts 8190,32760` adds
greedy parity checks around long-context bucket boundaries; set `--max-seq-len`,
`--pages` and `--moe-slots` explicitly to keep the production cache geometry.

`--compare-lrfu` compares ordinary LRU with decayed expert frequency (half-life
256 layer calls). `--compare-row-streams` compares sequential BF16 rows with
the experimental parallel-row path. Use fixed positive `--modes` and one
comparison flag at a time. Both keep separate graphs per variant and reset/warm
the cache before each measurement. `--diagnostics --routing-trace --repeats 0`
records eager expert routes for policy simulation; its timings are diagnostic.
`--reset-cache-between-runs` applies identical seeded cache warmup to independent
process comparisons, including MTP off. `--record-expert-stats` enables device
miss counters and reports estimated bank payload bytes; those runs are instrumented.
`--compare-gdn-input` adds a separate same-state eager comparison of logits and
filtered probabilities for the optional shared GDN input projection. It does not
include those extra forwards in the reported throughput measurements. Use fresh
processes for different GDN/cache environments so ordinary decode graphs also
capture the matching setting.
`bench_mtp_copy.py` sweeps copy launch geometry using the Qwen3.8 NVFP4 bank sizes;
`bench_mtp_linear.py --bf16-streams-only` checks independent-row stream candidates.

```bash
python benchmarks/bench_mtp_http.py --url http://127.0.0.1:1919 \
  --model /path/to/model --warmup-tokens 128 --tokens 512 --repeats 2 \
  --output /tmp/mtp-chat-http.json
```

**`bench_mtp_linear.py`** checks exact NVFP4 single-row parity and compares
sequential row launches with a multi-row GEMV grid using CUDA Graph timing.
It also evaluates BF16 batched GEMM as an experimental candidate; its equality
results do not change the engine's BF16 path.

```bash
python benchmarks/bench_mtp_linear.py --output /tmp/mtp-linear.json
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.

**`bench_kv_quant.py`** compares BF16, FP8 and NVFP4 KV storage bytes, one-step
scatter latency and paged decode latency on synthetic inputs. No checkpoint is
required. Keep the GPU idle and use identical arguments for A/B comparisons;
this does not measure model quality or end-to-end serving throughput.

```bash
PYTHONPATH=python:. uv run python benchmarks/bench_kv_quant.py --lengths 1024,8192,32768
```
