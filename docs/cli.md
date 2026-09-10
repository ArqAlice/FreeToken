# CLI reference

```
ft <command> [args]
```

| Command | Purpose |
|---|---|
| `ft serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft shell` | Chat with a server in the terminal |
| `ft ctl` | Query and manage a running server over HTTP |
| `ft launch` | Configure and launch a coding agent against a server |
| `ft checkpoint` | Convert an HF checkpoint to the FTW fast-load format |
| `ft bench bw` | Benchmark CPU vs PCIe bandwidth to calibrate the MoE backend |

`ft --version` prints the installed version (torch-free; nightly wheels carry a
`+g<sha>` build stamp, tagged releases a bare version). Every command supports
`--help`.

## ft serve

```bash
ft serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag — dtype, attention backend, MoE backend,
MoE cache size, KV capacity, CUDA-graph sizes and the tool-call/reasoning
parsers all resolve automatically from the checkpoint and the GPU.

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir, HF repo id, or an FTW dir (auto-detected) |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--gpu` | GPU 0 | GPU to run on: a UUID from `nvidia-smi -L` or an `nvidia-smi` index; see [below](#choosing-a-gpu) |
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |

### Choosing a GPU

For example, a machine with an RTX 5090 and an RTX 3060 Ti:

```console
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-2f3a9b1c-8d7e-4a05-b6c1-0e5f9a3d7b42)
GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-9e8d7c6b-5a49-4f13-8207-c1b0a4e6d3f5)
```

```bash
ft serve --model ... --gpu 1             # by nvidia-smi index -- the 5090
ft serve --model ... --gpu GPU-9e8d7c6b  # the same card by UUID (a unique prefix is enough)
```

### KV cache & memory

| Flag | Default | Meaning |
|---|---|---|
| `--memory-ratio` | 0.9 | Fraction of free VRAM the engine may use (weights + MoE cache + KV) |
| `--num-pages` / `--num-tokens` | auto | KV capacity override in pages / tokens (mutually exclusive; auto sizes from VRAM left after weights and MoE cache) |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--kv-cache-dtype` | bf16 | `bf16`, `fp8`, or `nvfp4` (see [NVFP4 KV cache](#nvfp4-kv-cache)): FP8 stores the KV cache as e4m3 codes plus one fp32 scale per (token, kv head), roughly doubling the tokens that fit in the same VRAM; see [FP8 KV cache](#fp8-kv-cache) |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU |

### FP8 KV cache

`ft serve --kv-cache-dtype fp8` halves the bytes per cached token (8-bit codes instead
of 16), so a card that held N tokens holds close to 2N. Each `(token, kv head)` row
keeps its own fp32 scale, which costs ~3% back at `head_dim=128`. Requirements and
trade-offs:

- Needs the **triton** attention backend; `--attn auto` selects it (and refuses an
  explicit `fi`/`fa`/`trtllm`, which cannot be shown to apply these scales).
- Works on the plain paged, hybrid-SWA and QSA sparse (Qwen3.8-Flash-Next) KV pools.
  On QSA the block-selection index keys stay 16-bit; only the selected K/V rows are
  read back as codes. MLA/DSA latent KV, DeepSeek-V4's tiered pool and the block-sparse
  MiniMax-M3 pool stay 16-bit; asking for fp8 there fails at startup rather than
  silently ignoring the flag.
- The same bytes on every GPU FreeToken targets: the codes sit in a plain byte buffer
  and are decoded in software, so the cache holds identical data and produces identical
  numbers on any card (the fp8 type is deliberately kept out of the kernels, which is
  also what makes the feature work on the RTX 30 series).
- Accuracy is checkpoint-dependent. Expect it to matter most on long contexts and on
  models with outlier key channels; keep `bf16` when a run must be bit-reproducible.
- `ft ctl stats` / `/v1/cache/status` report the smaller `kv_bytes_per_token`, and
  `ft ctl cache --kv N` moves the same (now cheaper) pool.

### MoE offload

See [models.md](models.md#moe-backends) for what each backend does.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-backend` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload, or hybrid with a `ft bench bw` profile |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family backends) |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`: which MoE layers decode on CPU (`3,7,11`, a count, or a fraction) |
| `--moe-hybrid-max-fetch` | auto | With `hybrid`: max experts fetched over PCIe per layer per step; rest computed on CPU |
| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |
| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |

### MTP speculative decoding

MTP uses the checkpoint's prediction head to propose tokens and verifies them
with the main model. It supports greedy generation and sampling with temperature,
top-k and top-p. This implementation requires a Qwen4-family checkpoint with
exactly one MTP layer and one GPU (TP=1). The tested checkpoint is
`aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP`.

| Flag | Default | Meaning |
|---|---|---|
| `--mtp-speculative-tokens N` | `0` | Maximum draft tokens per verification: `0` disables MTP; `1` through `4` enable it. Start with `1`. |
| `--mtp-auto` | off | Choose depth from measured time per emitted token for each request. Requires a positive `--mtp-speculative-tokens`; the automatic ceiling is `min(N, 3)`. |

For a first comparison, enable one draft token:

```bash
ft serve --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --mtp-speculative-tokens 1
```

To let the server try depths 0, 1, 2 and 3:

```bash
ft serve --model aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP \
  --mtp-speculative-tokens 3 --mtp-auto
```

Auto mode calibrates per request and can stop speculation for the rest of that
request when it is unprofitable. A new request calibrates again. It is a heuristic,
not a guarantee of the fastest setting; calibration and first-use CUDA graph
capture can make short replies slower. Server logs report depth changes and
estimated milliseconds per token. To disable MTP, set
`--mtp-speculative-tokens 0` and remove `--mtp-auto`.

MTP is a server setting. OpenWebUI and other API clients need no MTP-specific
request option: keep using their normal temperature, top-p, top-k and output
limits. Unspecified sampling values follow `--sampling-defaults`. Equal random
seeds do not guarantee identical sampled text with MTP enabled and disabled.

Larger lookahead can be slower, especially when experts must be fetched from
CPU memory or many proposals are rejected. MTP also disables prefix-cache reuse
and overlap scheduling; speculative requests currently run serially within a
batch. These restrictions can affect repeated conversations and concurrent use.
Automatic depth-zero sampling can use ordinary batched decode.

#### Docker Compose

Append these entries to the serving service's existing `command` list, or replace
the value of an existing MTP argument. Keep its model and other serving arguments:

```yaml
- --mtp-speculative-tokens
- "3"
- --mtp-auto
```

For fixed depth one, use `"1"` and omit `--mtp-auto`. Environment variables below
belong in the service's `environment` mapping, for example:

```yaml
FREETOKEN_MTP_FAST_PREPARE: "1"
FREETOKEN_MTP_DRAFT_MIN_PROB: "0.5"
```

Restart the serving process after changing MTP arguments or environment variables.
If the image copies source at build time, rebuild it to pick up code changes:

```bash
docker compose up -d --build <service-name>
```

Replace `<service-name>` with your existing serving service name. Changing the
Compose file without recreating the container does not apply the new settings.

#### Optional performance settings

These are environment variables, not `ft serve` arguments. All options in this
table are disabled by default. Try changes individually and compare warmed
requests on your workload before combining them.

| Variable | Suggested trial | Effect and tradeoff |
|---|---|---|
| `FREETOKEN_MTP_FAST_PREPARE` | `1` | Reduces metadata preparation for short single-request QSA continuations. Does not change model arithmetic or sampling; unsupported paths use ordinary preparation. |
| `FREETOKEN_MTP_DRAFT_MIN_PROB` | `0.5` | Stops additional stochastic lookahead when the largest filtered draft probability is below this threshold. Valid range `0`-`1`; `0` disables it. Keeps proposals already generated. Most useful to try with maximum depth 2 or 3; does not apply to greedy proposals. Can change seeded text and adds a confidence-check cost. |
| `FREETOKEN_FAST_LINEAR` | `1` | Uses batched/shared-weight BF16 projections, including the vocabulary head and greedy verification. Changes rounding and can change generated text. Also affects MTP-off execution. |
| `FREETOKEN_GDN_FP8_INPUT` | `1` | Converts unquantized Qwen4 GDN input weights to FP8 at load time, keeping activations BF16. Requires SM89 or newer. Saves about 1.40 GiB for the tested checkpoint; adds numerical approximation and can change text. Also affects MTP-off execution; the checkpoint on disk stays unchanged. |
| `FREETOKEN_EXPERT_LRFU_HALF_LIFE` | `256` | Uses decayed access frequency for GPU expert retention. Integer range `0`-`4096`; `0` keeps the default policy. Alters cache behavior, not weight precision; gains depend on routing. |
| `FREETOKEN_EXPERT_LAYER_DISTANCE` | `2` | Adds layer distance to expert eviction decisions. Range `0`-`16`; requires a positive LRFU half-life. Also affects ordinary GPU offload decoding. |

Fast BF16 and GDN FP8 can be enabled separately. Fast BF16 includes the shared
GDN projection path and takes precedence over the older BF16 rowwise/batched/
parallel switches. Exact greedy MTP-on/off output equality is not promised with
these speed-first settings. Evaluate answer quality as well as throughput.

For CPU expert offload, automatic cache sizing can use memory reclaimed by GDN
FP8 for more experts. A fixed `--moe-cache-size` stays fixed. Likewise, reserving
more KV capacity leaves less VRAM for experts: choose `--max-seq-len-override`
and `--kv-reserve-tokens` for the context capacity you actually need. Reducing
capacity is a separate tradeoff, not an MTP kernel speedup. `--kv-cache-dtype`
controls KV storage independently of GDN FP8 weights.

#### Additional experiments and diagnostic switches

These switches are available for targeted comparisons. They are not required
to enable MTP, and enabling all of them is not a recommended speed preset.

| Variable | Default | Purpose / limitation |
|---|---|---|
| `FREETOKEN_MTP_FUSED_COMMIT` | `0` | GPU accepted-prefix state updates; only a small full-generation gain was measured. |
| `FREETOKEN_MTP_DRAFT_FP8_HEAD` | `0` | FP8 copy of the draft vocabulary head; CUDA BF16 source weights and SM89+ required. Adds about 607 MiB for the tested checkpoint and reduces available expert-cache memory. Target head is unchanged. |
| `FREETOKEN_MTP_DRAFT_PROBS_GRAPH` | `0` | CUDA graphs for draft probability calculation. No consistent full-generation gain measured on its own. |
| `FREETOKEN_MTP_GREEDY_DRAFT` | `0` | Deterministic draft proposals with stochastic target sampling. Reduced acceptance and throughput in the tested chat workload. |
| `FREETOKEN_MTP_QSA_REUSE` | `0` | Reuses draft-only QSA selections for depths 2-4. Can change draft probabilities; short-chat gains were not established. |
| `FREETOKEN_MTP_EXPERT_SPEC_WEIGHT` | `1` | LRFU credit for experts used only by unverified rows; range `(0, 1]`, values below `1` require positive LRFU half-life. Leave at `1` initially: lower credit did not establish reduced transfer volume. |
| `FREETOKEN_NVFP4_MOE_SHARED_ROUTES` | `0` | Shares resident NVFP4 expert reads between pairs of routes. No consistent chat throughput gain established. |
| `FREETOKEN_NVFP4_MOE_GROUPED_ROUTES` | `0` | Groups up to four matching routes; takes precedence over pairing where supported. Improved isolated MoE timing, but not full chat throughput consistently. |
| `FREETOKEN_GDN_SHARED_INPUT` | `0` | Shared BF16 GDN input projection. Can change rounding; already included in fast-linear mode. |
| `FREETOKEN_MTP_BATCHED_LINEAR` | `0` | Older batched BF16 experiment for non-greedy verification; can change numerical results. Superseded by fast-linear mode when that is enabled. |
| `FREETOKEN_MTP_PARALLEL_LINEAR` | `0` | Parallel single-row BF16 execution for selected shapes; measured gains were small. Superseded by fast-linear mode when enabled. |
| `FREETOKEN_MTP_EXPERT_STATS` | `0` | Adds confirmed/speculative expert access and miss counters with LRFU. Diagnostic overhead; leave off for normal serving. |
| `FREETOKEN_MTP_CUDA_GRAPH` / `FREETOKEN_MTP_DRAFT_GRAPH` | `1` | Target/draft continuation graphs. Set to `0` only to compare eager execution. |
| `FREETOKEN_MTP_STATE_HISTORY` | `1` | Saves prefix states to avoid target replay after rejection. `0` selects restore-and-replay. |
| `FREETOKEN_MTP_EXPERT_OVERLAP` / `FREETOKEN_MTP_ASYNC_PLE` | `1` | Overlap expert fetching / stage disk PLE asynchronously where supported. `0` disables the respective path for comparison. |
| `FREETOKEN_MTP_NVFP4_SHARED_ROWS` / `FREETOKEN_NVFP4_MOE_ARITHMETIC` | `1` | Existing NVFP4 projection weight reuse / expert unpacking optimizations. Normally leave enabled. |

For a useful speed comparison, keep the same model, prompt, sampling settings,
output budget, context capacity, expert-cache budget and precision settings.
Compare MTP off, fixed depth one, then auto or deeper lookahead over several
warmed requests. Check both first-token latency and decode tokens/s. Auto falling
back to zero and an isolated kernel improvement do not establish faster active
MTP. Short-chat measurements on the tested 32 GB system have not established a
consistent large advantage over MTP off.

### API behaviour

| Flag | Default | Meaning |
|---|---|---|
| `--sampling-defaults` | model | Fill unspecified sampling params from the checkpoint's `generation_config.json` (`none` = framework defaults) |
| `--tool-call-parser` | auto | Tool-call format; auto-inferred from the model family |
| `--reasoning-parser` | auto | Splits chain-of-thought into `reasoning_content`; auto-inferred; `off` disables |
| `--enable-cache-report` | off | Report prefix-cache hits in each response's usage block |

## ft shell

```bash
ft shell                                    # attach to a running server
ft shell --model ~/models/Qwen3.6-35B-A3B   # serve + chat in one process
```

- Attach mode talks to `--server URL` (default `http://127.0.0.1:1919`)
- `/help` inside the shell lists the commands (`/think`, `/cache`, `/reset`).

## ft ctl

```bash
ft ctl [--base-url http://127.0.0.1:1919] [--timeout 10] [--json] <subcommand>
```

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `health` | `GET /health` | Server status, model, load progress |
| `stats` | `GET /v1/stats` | Throughput, latency, VRAM, pool occupancy |
| `generate [prompt] [--max-tokens N] [--ignore-eos]` | `POST /generate` | Raw completion smoke test (no chat template) |
| `cache` | `GET /v1/cache/status` | Cache pool table |
| `cache --moe N \| --kv N \| --mamba N \| --swa N [--wait 300]` | `POST /v1/cache/rebuild` | Live pool resizing without a restart (`k`/`m` suffixes; `--kv`/`--swa` in tokens) |
| `requests [--since N] [--limit N]` | `GET /v1/requests` | Recent request ring |

## ft launch

```bash
ft launch {claude,codex,dsh,hermes,openclaw,opencode} [options] [-- <agent args>]
```

Discovers the served model via `/v1/models`, writes the agent's provider
config, installs the agent CLI if missing, then launches it. Cloud API keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) are cleared from the child
environment so the agent cannot silently fall back to a paid endpoint.

| Flag | Meaning |
|---|---|
| `--server URL` | Server to point the agent at (default `http://127.0.0.1:1919`) |
| `--dry-run` | Print the planned config changes and command, touch nothing |
| `-y`, `--yes` | Approve install/config prompts |
| `--config` | Configure without launching |
| `--install-only` | Just install the agent CLI (needs no server) |
| `--force-reinstall` | Re-run the agent installer |
| `-- <args>` | Forwarded verbatim to the agent |

## ft checkpoint

```bash
ft checkpoint --model <hf_dir> --out <ftw_dir> [--dtype bfloat16] [--moe-backend offload] [--shard-gib 8] [--gpu <uuid-or-index>]
```

Converts an HF safetensors checkpoint to FTW, FreeToken's self-contained
fast-load format; point `ft serve --model` at the output dir. `--moe-backend
offload` (default) packs experts into offload banks; `--moe-backend triton`
keeps them dense for resident serving. See the FTW caveats in
[models.md](models.md#notes).

## ft bench bw

```bash
ft bench bw                       # once per GPU
ft bench bw --dtype nvfp4,bf16    # only the formats you serve
ft bench bw --gpu 1               # a specific GPU (UUID or nvidia-smi index, as for ft serve)
```

Measures host-RAM vs PCIe bandwidth with the real cpu/offload MoE kernels and writes a
profile that `ft serve --moe-backend auto` and `--moe-hybrid-max-fetch -1` then read.

- One profile per GPU, at `~/.cache/freetoken/benchbw/<gpu-uuid>.json`.
- Keyed on expert format + GPU, so a profile from other hardware is ignored rather than
  misapplied. An older single `benchbw.json` still counts if its GPU name matches.
- What to measure: `--dtype`, `--model`, `--formats`, `--isa`.
- `--threshold` (default 2.0) sets the call: recommend hybrid when CPU bandwidth beats PCIe
  by that factor.


### NVFP4 KV cache

`ft serve --model <checkpoint> --kv-cache-dtype nvfp4 --attention-backend triton`
opts into packed E2M1 KV storage. The initial implementation supports plain paged
FULL attention (MHA/GQA), hybrid-SWA, and the full-attention portion of hybrid-linear
models, QSA, and MLA/DSA (including GLM-5.3-Flash). Head dimensions must be
divisible by 16. DSV4 and BSA pools are rejected at startup. `auto` selects
Triton, QSA sparse, or DSA attention for supported models. For MLA/DSA use
`--attention-backend auto` or `--attention-backend dsa`. Only the latent slab is
quantized; indexer keys, kpool tails/gates, and recurrent states retain their
existing precision. A 512-element latent row occupies 292 bytes instead of 1024
bytes in BF16, excluding those other tiers.

Each K or V row stores `head_dim / 2` packed bytes, `head_dim / 16` E4M3 block-scale
bytes, and one FP32 row scale. At head_dim 128 this is 76 bytes, versus 256 for
BF16 and 132 for the existing FP8 format. Pool management, recurrent states,
attention workspace and model weights consume additional memory.

The second-level scale is dynamic per token/head, so appending a token never
rescales an existing prefix. This is a FreeToken KV layout, not an external
NVFP4 checkpoint or attention-library ABI. K/V are restored inside attention;
Q and attention arithmetic retain their compute precision. The MoE weight option
`--nvfp4-backend` is independent. Paged MHA prefill uses fresh compute-dtype K/V
while cached prefixes are restored, as in the FP8 path. MLA/DSA stores fresh
latent rows first and reads the quantized cache in both prefill and decode.

NVFP4 is opt-in: assess quality on your checkpoint and workload before using it
for long-context inference. Capacity savings do not guarantee faster decode;
packing, reconstruction, and the selected attention backend affect throughput.
