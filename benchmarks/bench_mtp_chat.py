"""Offline chat-template MTP comparison with server-sized KV capacity."""

import argparse
from copy import copy
import json
import os
from pathlib import Path
import time

os.environ["FREETOKEN_DISABLE_OVERLAP_SCHEDULING"] = "1"

import torch

from freetoken.core import SamplingParams
from freetoken.llm.llm import LLM
from freetoken.message import DetokenizeMsg
from freetoken.scheduler.mtp import MTPRunner


def tensor_bytes(value):
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    return value.numel() * value.element_size() if isinstance(value, torch.Tensor) else 0


def compare_batched_linear(llm, ids, mode, *, gdn_shared=False):
    import faulthandler

    runner = llm._mtp_runner
    original_target = runner._target
    previous_mode = llm.config.mtp_speculative_tokens
    env_key = "FREETOKEN_GDN_SHARED_INPUT" if gdn_shared else "FREETOKEN_MTP_BATCHED_LINEAR"
    previous_env = os.environ.get(env_key)
    evaluate = runner._target_eager if gdn_shared else original_target
    cases = []
    sampling = SamplingParams(max_tokens=128, ignore_eos=True, temperature=1., top_k=20, top_p=.95)

    def filtered(probs):
        threshold = probs.topk(20, dim=-1).values[:, -1:]
        probs = probs.masked_fill(probs < threshold, 0)
        probs /= probs.sum(-1, keepdim=True)
        sorted_probs = probs.sort(dim=-1, descending=True).values
        index = (sorted_probs.cumsum(-1) < .95).sum(-1, keepdim=True)
        threshold = sorted_probs.gather(-1, index.clamp_max(probs.shape[-1] - 1))
        probs = probs.masked_fill(probs < threshold, 0)
        return probs / probs.sum(-1, keepdim=True)

    def compared(batch, **kwargs):
        if not kwargs.get("all_tokens", False) or len(cases) >= 16:
            return original_target(batch, **kwargs)
        reference_batch = copy(batch)
        reference_batch.mtp_batched_linear = False
        saved = runner._snapshot(batch.reqs[0])
        try:
            if gdn_shared:
                os.environ[env_key] = "0"
            reference = evaluate(reference_batch, **kwargs)[0].float().cpu()
        finally:
            runner._restore(batch.reqs[0], saved)
            if gdn_shared:
                os.environ[env_key] = "1"
        batch.mtp_batched_linear = not gdn_shared
        result = evaluate(batch, **kwargs)
        fast = result[0].float().cpu()
        p, q = reference.softmax(-1), fast.softmax(-1)
        softmax_tv = .5 * (p - q).abs().sum(-1)
        filtered_tv = .5 * (filtered(p) - filtered(q)).abs().sum(-1)
        agrees = reference.argmax(-1).eq(fast.argmax(-1))
        delta = (reference - fast).abs().amax(-1)
        values = torch.stack((softmax_tv, filtered_tv, agrees.float(), delta), dim=-1).tolist()
        cases.append(dict(
            verification=len(cases), cached_len=batch.reqs[0].cached_len,
            rows=[dict(softmax_tv=row[0], filtered_tv=row[1], top1_equal=bool(row[2]),
                       max_logit_delta=row[3]) for row in values],
        ))
        return result

    object.__setattr__(llm.config, "mtp_speculative_tokens", mode)
    os.environ[env_key] = "1"
    runner._target = compared
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        with torch.random.fork_rng(devices=[llm.engine.device]):
            torch.manual_seed(20260909)
            output = llm.generate([ids], sampling)[0]
    finally:
        faulthandler.cancel_dump_traceback_later()
        runner._target = original_target
        object.__setattr__(llm.config, "mtp_speculative_tokens", previous_mode)
        if previous_env is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous_env
    rows = [row for case in cases for row in case["rows"]]
    aggregate = dict(verifications=len(cases), rows=len(rows))
    if rows:
        aggregate.update(
            mean_softmax_tv=sum(row["softmax_tv"] for row in rows) / len(rows),
            max_softmax_tv=max(row["softmax_tv"] for row in rows),
            mean_filtered_tv=sum(row["filtered_tv"] for row in rows) / len(rows),
            max_filtered_tv=max(row["filtered_tv"] for row in rows),
            top1_agreement=sum(row["top1_equal"] for row in rows) / len(rows),
            max_logit_delta=max(row["max_logit_delta"] for row in rows),
        )
    return dict(
        settings=dict(mtp=mode, temperature=1., top_k=20, top_p=.95, seed=20260909,
                      generated_tokens=len(output["token_ids"]), verification_limit=16,
                      probability_reference="cpu fp32 softmax and threshold filters including ties",
                      reference_batched_linear=False, candidate_batched_linear=not gdn_shared,
                      shared_gdn_comparison=gdn_shared, eager=gdn_shared),
        aggregate=aggregate, cases=cases,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="猫の魅力について熱くたくさん語って。")
    parser.add_argument("--prompt-file", help="UTF-8 prompt file, overrides --prompt")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", default="0,3")
    parser.add_argument("--mtp-auto", action="store_true")
    parser.add_argument("--compare-ple", action="store_true")
    parser.add_argument("--compare-lrfu", action="store_true")
    parser.add_argument("--compare-row-streams", action="store_true")
    parser.add_argument("--reset-cache-between-runs", action="store_true",
                        help="Reset expert residency and run the same seeded warmup before each measurement")
    parser.add_argument("--record-expert-stats", action="store_true",
                        help="Include GPU expert miss counters; marks the run as instrumented")
    parser.add_argument("--max-seq-len", type=int, default=1000128)
    parser.add_argument("--pages", type=int, default=15629)
    parser.add_argument("--moe-slots", type=int, default=3073)
    parser.add_argument("--nvfp4-backend", choices=("triton", "flashinfer", "auto"), default="triton")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--routing-trace", action="store_true",
                        help="Record eager expert routes for cache-policy simulation; requires diagnostics")
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-trace", help="Chrome trace path for the separate --profile generation")
    parser.add_argument("--compare-batched-linear", action="store_true")
    parser.add_argument("--compare-gdn-input", action="store_true",
                        help="Separate same-state eager comparison of GDN logit/probability differences")
    args = parser.parse_args()
    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    modes = [int(x) for x in args.modes.split(",")]
    if args.profile_trace and not args.profile:
        parser.error("--profile-trace requires --profile")
    if args.compare_batched_linear and max(modes) <= 0:
        parser.error("--compare-batched-linear needs at least one positive MTP mode")
    if args.compare_gdn_input and (max(modes) <= 0 or args.compare_batched_linear):
        parser.error("--compare-gdn-input needs positive MTP and no batched-linear comparison")
    if args.diagnostics and (args.repeats != 0 or max(modes) <= 0):
        parser.error("--diagnostics requires --repeats 0 and a positive MTP mode; run timing separately")
    if args.routing_trace:
        if not args.diagnostics:
            parser.error("--routing-trace requires --diagnostics")
        os.environ['FREETOKEN_MTP_CUDA_GRAPH'] = '0'
        os.environ['FREETOKEN_MTP_DRAFT_GRAPH'] = '0'
    if args.compare_lrfu:
        if args.compare_ple or args.mtp_auto or min(modes) <= 0:
            parser.error("--compare-lrfu requires fixed positive modes without --compare-ple")
        os.environ['FREETOKEN_EXPERT_LRFU_HALF_LIFE'] = '256'
    if args.compare_row_streams and (args.compare_lrfu or args.compare_ple or args.mtp_auto or min(modes) <= 0):
        parser.error('--compare-row-streams requires fixed positive modes and no other comparison')
    llm = LLM(args.model, max_running_req=2, max_extend_tokens=8192,
        max_seq_len_override=args.max_seq_len, kv_quant="nvfp4", moe_cache_size=args.moe_slots,
        num_page_override=args.pages, mtp_speculative_tokens=max(modes), cache_type="naive",
        nvfp4_backend=args.nvfp4_backend, moe_collect_stats=args.diagnostics or args.record_expert_stats,
        mtp_auto=args.mtp_auto)
    if max(modes) > 0 and not hasattr(llm, "_mtp_runner"):
        llm._mtp_runner = MTPRunner(llm)
    ids = llm.tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
                                           tokenize=True, add_generation_prompt=True, return_dict=False)
    environment = {
        "FREETOKEN_MTP_BATCHED_LINEAR": "0",
        "FREETOKEN_MTP_CUDA_GRAPH": "1",
        "FREETOKEN_MTP_DRAFT_GRAPH": "1",
        "FREETOKEN_MTP_NVFP4_SHARED_ROWS": "1",
        "FREETOKEN_MTP_STATE_HISTORY": "1",
        "FREETOKEN_MTP_EXPERT_OVERLAP": "1",
        "FREETOKEN_MTP_GREEDY_DRAFT": "0",
        "FREETOKEN_MTP_QSA_REUSE": "0",
        "FREETOKEN_NVFP4_MOE_ARITHMETIC": os.environ.get("FREETOKEN_NVFP4_MOE_ARITHMETIC", "1"),
        "FREETOKEN_EXPERT_LRFU_HALF_LIFE": os.environ.get("FREETOKEN_EXPERT_LRFU_HALF_LIFE", "0"),
        "FREETOKEN_EXPERT_LAYER_DISTANCE": os.environ.get("FREETOKEN_EXPERT_LAYER_DISTANCE", "0"),
        "FREETOKEN_GDN_SHARED_INPUT": os.environ.get("FREETOKEN_GDN_SHARED_INPUT", "0"),
    }
    environment.update({key: value for key, value in os.environ.items() if key.startswith("FREETOKEN_MTP_")})
    report = {"settings": {**vars(args), "environment": environment}, "input_tokens": len(ids), "runs": []}

    def params(n):
        return SamplingParams(max_tokens=n, ignore_eos=True, temperature=args.temperature,
                              top_k=args.top_k, top_p=args.top_p)

    cases = [(mode, skip) for mode in modes for skip in (("0", "1") if args.compare_ple else (None,))]
    if args.compare_lrfu:
        cases = [(mode, control) for mode in modes for control in ('0', '256')]
    if args.compare_row_streams:
        cases = [(mode, control) for mode in modes for control in ('0', '1')]
    graph_sets = {}

    def select_case(mode, control):
        if args.compare_lrfu:
            cache = llm.engine.moe_offload_cache
            cache.lrfu_half_life = int(control)
            graphs = graph_sets.setdefault(control, ({}, {}))
            llm._mtp_runner._target_graphs, llm._mtp_runner._draft_graphs = graphs
        elif args.compare_row_streams:
            os.environ['FREETOKEN_MTP_PARALLEL_LINEAR'] = control
            graphs = graph_sets.setdefault(control, ({}, {}))
            llm._mtp_runner._target_graphs, llm._mtp_runner._draft_graphs = graphs
        elif control is not None:
            os.environ['FREETOKEN_MTP_ASYNC_PLE'] = control
        object.__setattr__(llm.config, 'mtp_speculative_tokens', mode)

    for mode, skip in cases:
        select_case(mode, skip)
        torch.manual_seed(137)
        llm.generate([ids], params(args.warmup_tokens))
    arrivals = [None, None]
    original_send = llm.send_result

    def received(reply):
        if any(isinstance(msg, DetokenizeMsg) for msg in reply):
            now = time.perf_counter()
            if arrivals[0] is None:
                arrivals[0] = now
            arrivals[1] = now
        original_send(reply)

    llm.send_result = received
    for repeat in range(args.repeats):
        for mode, skip in cases if repeat % 2 == 0 else reversed(cases):
            select_case(mode, skip)
            if args.compare_lrfu or args.compare_row_streams or args.reset_cache_between_runs:
                llm.engine.moe_offload_cache.reset()
                torch.manual_seed(137)
                llm.generate([ids], params(args.warmup_tokens))
            torch.manual_seed(1000 + repeat)
            runner = getattr(llm, "_mtp_runner", None)
            counts = (runner.proposed, runner.accepted) if runner is not None else (0, 0)
            qsa_counts = {name: getattr(runner, name, 0)
                          for name in ("qsa_reused_heads", "qsa_refreshed_heads")}
            auto_counts = {key: dict(value) for key, value in getattr(runner, "auto_stats", {}).items()}
            lengths = dict(runner.acceptance_lengths) if runner is not None else {}
            arrivals[:] = [None, None]
            if args.record_expert_stats:
                llm.engine.moe_offload_cache.reset_stats()
            torch.cuda.synchronize()
            before = time.perf_counter()
            output = llm.generate([ids], params(args.tokens))[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - before
            decode = arrivals[1] - arrivals[0]
            entry = dict(repeat=repeat, mtp=mode, seconds=elapsed,
                async_ple=skip if args.compare_ple else None,
                lrfu_half_life=int(skip) if args.compare_lrfu else None,
                parallel_linear=skip if args.compare_row_streams else None,
                ttft_seconds=arrivals[0] - before, decode_seconds=decode,
                decode_tokens_per_second=(len(output["token_ids"]) - 1) / decode,
                proposed=(runner.proposed if runner is not None else 0) - counts[0],
                accepted=(runner.accepted if runner is not None else 0) - counts[1], **output)
            entry["acceptance_lengths"] = {
                key: value - lengths.get(key, 0) for key, value in runner.acceptance_lengths.items()
                if value > lengths.get(key, 0)
            } if runner is not None else {}
            history = runner._verify_history if runner is not None else None
            entry["history_bytes"] = tensor_bytes(history)
            if args.mtp_auto:
                runner._drain_timings()
                entry["auto"] = {key: {name: value - auto_counts.get(key, {}).get(name, 0)
                                        for name, value in values.items()}
                                 for key, values in runner.auto_stats.items()}
            entry.update({name: getattr(runner, name, 0) - value for name, value in qsa_counts.items()})
            entry["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
            if args.record_expert_stats:
                from flashlib.kernels.slot_cache import Stat

                cache = llm.engine.moe_offload_cache
                values = cache.lru_stats.tolist()
                row_bytes = [sum(tensors[layer][0].numel() * tensors[layer].element_size()
                                 for tensors in cache.bank_sources.values())
                             for layer in range(cache.num_layers)]
                entry['expert_stats'] = dict(
                    active=sum(v[Stat.ACTIVE] for v in values),
                    misses=sum(v[Stat.MISS] for v in values),
                    layer_calls=sum(v[Stat.CALLS] for v in values),
                    estimated_fetch_bytes=sum(v[Stat.MISS] * b for v, b in zip(values, row_bytes)))
            report["runs"].append(entry)
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps({k: v for k, v in entry.items() if k not in ("text", "token_ids")}), flush=True)
    llm.send_result = original_send
    if args.diagnostics:
        from mtp_diagnostics import collect_diagnostics

        report["diagnostics"] = {}
        for mode in modes:
            if mode <= 0:
                continue
            object.__setattr__(llm.config, "mtp_speculative_tokens", mode)
            report["diagnostics"][str(mode)] = collect_diagnostics(
                llm, ids, params(args.tokens), routing_trace=args.routing_trace)
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if args.compare_batched_linear:
        report["batched_linear_comparison"] = compare_batched_linear(llm, ids, max(modes))
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({"batched_linear_comparison": report["batched_linear_comparison"]["aggregate"]}), flush=True)
    if args.compare_gdn_input:
        report["gdn_input_comparison"] = compare_batched_linear(llm, ids, max(modes), gdn_shared=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({"gdn_input_comparison": report["gdn_input_comparison"]["aggregate"]}), flush=True)
    if args.profile:
        object.__setattr__(llm.config, "mtp_speculative_tokens", max(modes))
        events = []
        originals = []
        for owner, name in [(llm._mtp_runner, x) for x in ("_one", "_target", "_draft", "_lookahead", "_batch")
                            if hasattr(llm, "_mtp_runner")
                            ] + [(llm.engine.sampler, "probabilities"), (llm.engine.sampler, "sample_probs")]:
            fn = getattr(owner, name)
            originals.append((owner, name, fn))

            def timed(*a, _fn=fn, _name=name, **kw):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                t = time.perf_counter()
                result = _fn(*a, **kw)
                host = time.perf_counter() - t
                end.record()
                events.append((_name, start, end, host))
                return result

            setattr(owner, name, timed)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            llm.generate([ids], params(128))
        torch.cuda.synchronize()
        report["timings"] = {}
        for name, start, end, host in events:
            row = report["timings"].setdefault(name, dict(calls=0, stream_seconds=0., host_seconds=0.))
            row["calls"] += 1
            row["stream_seconds"] += start.elapsed_time(end) / 1000
            row["host_seconds"] += host
        for owner, name, fn in originals:
            setattr(owner, name, fn)
        report["profile_cuda"] = profile.key_averages().table(sort_by="self_device_time_total", row_limit=35)
        report["profile_cpu"] = profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
        if args.profile_trace:
            profile.export_chrome_trace(args.profile_trace)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    llm.shutdown()


if __name__ == "__main__":
    main()
