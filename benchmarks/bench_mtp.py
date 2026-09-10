"""Same-process MTP A/B and same-state, teacher-forced layer diagnostics.

Run with a downloaded checkpoint on an otherwise idle GPU. Both paths use the same
weights, expert cache budget, naive KV cache, and non-overlap scheduling.
"""
from __future__ import annotations

import argparse
from copy import copy
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

os.environ["FREETOKEN_DISABLE_OVERLAP_SCHEDULING"] = "1"

import torch

from freetoken.core import Batch, SamplingParams
from freetoken.llm.llm import LLM
from freetoken.message import DetokenizeMsg
from freetoken.scheduler.mtp import MTPRunner


def diagnose(runner, req):
    scheduler = runner.scheduler
    start, end = req.cached_len, req.device_len
    count = scheduler.config.mtp_speculative_tokens
    ids = torch.cat((scheduler.token_pool[req.table_idx, start:end], runner.drafts[req.uid].repeat(count)))
    saved = runner._snapshot(req)
    allocation = copy(req)
    allocation.cached_len, allocation.device_len = end, end + count
    scheduler.cache_manager.allocate_paged([allocation])
    traces = {}
    originals = []
    current = []
    for i, layer in enumerate(runner.engine.model.model.layers.op_list):
        original = layer.forward
        originals.append((layer, original))

        def traced(*args, _fn=original, _i=i, **kwargs):
            result = _fn(*args, **kwargs)
            current.append((_i, result.detach().float().cpu()))
            return result

        layer.forward = traced

    def single(position, token, phase):
        view = copy(req)
        view.cached_len, view.device_len = position, position + 1
        view.input_ids = torch.cat((req.input_ids[:start], ids[:position - start + 1].cpu()))
        view.mamba_ping_pong = None
        batch = Batch([view], phase)
        batch.use_decode_moe = True
        scheduler._prepare_batch(batch, allocate=False)
        batch.input_ids = token
        return runner._target(batch)[0].float().cpu()

    try:
        batch = runner._batch(req, start, end + count, ids)
        joint = runner._target(batch, all_tokens=True)[0].float().cpu()
        traces["joint"] = dict(current)
        logits = {"joint": joint}
        for phase in ("prefill", "decode"):
            runner._restore(req, saved)
            per_layer = {}
            rows = []
            for j in range(count + 1):
                current.clear()
                rows.append(single(start + j, ids[j:j + 1], phase))
                for i, value in current:
                    per_layer.setdefault(i, []).append(value)
            traces[phase] = {i: torch.cat(values) for i, values in per_layer.items()}
            logits[phase] = torch.cat(rows)
        runner._restore(req, saved)
        current.clear()
        linear = torch.nn.functional.linear

        def row_linear(x, weight, bias=None):
            if x.ndim == 2 and x.shape[0] > 1:
                return torch.cat([linear(row, weight, bias) for row in x.split(1)])
            return linear(x, weight, bias)

        with patch("torch.nn.functional.linear", row_linear), patch.dict(
            os.environ, {"FREETOKEN_MTP_CUDA_GRAPH": "0"}
        ):
            batch = runner._batch(req, start, end + count, ids)
            logits["row_linear"] = runner._target(batch, all_tokens=True)[0].float().cpu()
        traces["row_linear"] = dict(current)
        result = {"position": start, "input_ids": ids.tolist(), "layers": []}
        for i in traces["joint"]:
            entry = {"layer": i}
            for phase in ("prefill", "decode", "row_linear"):
                delta = traces["joint"][i] - traces[phase][i]
                entry[phase] = {"max_abs": delta.abs().amax(1).tolist(),
                                "rms": delta.square().mean(1).sqrt().tolist()}
            result["layers"].append(entry)
        result["logits"] = {
            key: {"top_ids": value.topk(5).indices.tolist(),
                  "top_values": value.topk(5).values.tolist(),
                  "max_abs_vs_decode": (value - logits["decode"]).abs().amax(1).tolist(),
                  "max_abs_vs_joint": (value - joint).abs().amax(1).tolist()}
            for key, value in logits.items()
        }
        return result
    finally:
        for layer, original in originals:
            layer.forward = original
        runner._restore(req, saved)
        ps = scheduler.config.page_size
        first, last = ((end + ps - 1) // ps) * ps, ((end + count + ps - 1) // ps) * ps
        if last > first:
            cm = scheduler.cache_manager
            cm._free(cm.page_table[req.table_idx, first:last].clone())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--diagnose-position", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validation-prompt-file", help="UTF-8 prompt for the chunked validation case")
    parser.add_argument("--max-extend-tokens", type=int, default=128)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compare-draft-norms", action="store_true")
    parser.add_argument("--timings", action="store_true")
    parser.add_argument("--compare-graphs", action="store_true")
    parser.add_argument("--long-prompt", action="store_true")
    parser.add_argument("--compare-legacy-linear", action="store_true")
    parser.add_argument("--state-diagnostic", action="store_true")
    parser.add_argument("--draft-diagnostic", action="store_true")
    parser.add_argument("--sampling-smoke", action="store_true")
    parser.add_argument("--sampling-tokens", type=int, default=64)
    parser.add_argument("--speculative-tokens", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--mtp-auto", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--pages", type=int, default=132)
    parser.add_argument("--moe-slots", type=int, default=6000)
    parser.add_argument("--validation-contexts", default="")
    parser.add_argument("--trace-stalls", action="store_true")
    parser.add_argument("--force-auto-depths", action="store_true")
    args = parser.parse_args()
    if args.trace_stalls:
        import faulthandler
        faulthandler.dump_traceback_later(60, repeat=True)
    if args.force_auto_depths:
        if not args.mtp_auto:
            parser.error("--force-auto-depths requires --mtp-auto")
        from freetoken.scheduler.mtp_policy import MTPPolicy
        observed = MTPPolicy.observe
        def force_exploration(self, depth, seconds, emitted):
            observed(self, depth, (.02 if depth == 0 else .012 / depth) * emitted, emitted)
        MTPPolicy.observe = force_exploration
    mtp_mode = args.speculative_tokens
    llm = LLM(args.model, max_running_req=2, max_extend_tokens=args.max_extend_tokens,
              max_seq_len_override=args.max_seq_len, kv_quant="nvfp4", moe_cache_size=args.moe_slots,
              num_page_override=args.pages, mtp_speculative_tokens=mtp_mode, cache_type="naive",
              mtp_auto=args.mtp_auto)
    report = {"model": args.model, "tokens": args.tokens, "warmup_tokens": args.warmup_tokens,
              "settings": vars(args),
              "instrumented": args.timings or args.force_auto_depths,
              "max_extend_tokens": args.max_extend_tokens, "runs": []}
    prompts = ["The capital of France is", "Write a Python function that computes Fibonacci numbers."]
    if args.long_prompt:
        prompts = ["hello " * 140 + "Complete: one two three"]
    if args.diagnose:
        original = MTPRunner._one

        def measured(self, req, phase):
            if (phase == "decode" and req.uid in self.drafts and "diagnostic" not in report
                    and req.cached_len >= args.diagnose_position):
                report["diagnostic"] = diagnose(self, req)
            return original(self, req, phase)

        MTPRunner._one = measured
        try:
            llm.generate([prompts[0]], SamplingParams(
                max_tokens=args.tokens if args.diagnose_position else 4, ignore_eos=True))
        finally:
            MTPRunner._one = original
        Path(args.output).write_text(json.dumps(report, indent=2))
        llm.shutdown()
        return
    for mode in (0, mtp_mode):
        object.__setattr__(llm.config, "mtp_speculative_tokens", mode)
        for prompt in prompts:
            llm.generate([prompt], SamplingParams(max_tokens=args.warmup_tokens, ignore_eos=True))
    timing_events = []
    timing_originals = {}
    timing_phase = ["prefill"]
    if args.timings:
        runner = llm._mtp_runner
        for name in ("_one", "_target", "_draft", "_lookahead", "_snapshot", "_restore", "_batch"):
            original = getattr(runner, name)
            timing_originals[name] = original

            def timed(*a, _name=name, _fn=original, **kw):
                if _name == "_one":
                    timing_phase[0] = a[1]
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                before = time.perf_counter()
                result = _fn(*a, **kw)
                elapsed = time.perf_counter() - before
                end.record()
                timing_events.append((_name, timing_phase[0], begin, end, elapsed))
                return result

            setattr(runner, name, timed)
    arrivals = [None, None]
    send_result = llm.send_result

    def received(reply):
        if any(isinstance(msg, DetokenizeMsg) for msg in reply):
            now = time.perf_counter()
            if arrivals[0] is None:
                arrivals[0] = now
            arrivals[1] = now
        send_result(reply)

    llm.send_result = received
    for repeat in range(args.repeats):
        for mode in (0, mtp_mode):
            object.__setattr__(llm.config, "mtp_speculative_tokens", mode)
            for prompt in prompts:
                torch.cuda.synchronize()
                arrivals[:] = [None, None]
                before = time.perf_counter()
                counts = (llm._mtp_runner.proposed, llm._mtp_runner.accepted)
                result = llm.generate([prompt], SamplingParams(max_tokens=args.tokens, ignore_eos=True))[0]
                torch.cuda.synchronize()
                entry = {"repeat": repeat, "mtp": mode, "prompt": prompt,
                         "seconds": time.perf_counter() - before,
                         "proposed": llm._mtp_runner.proposed - counts[0],
                         "accepted": llm._mtp_runner.accepted - counts[1], **result}
                report["runs"].append(entry)
                if arrivals[0] is not None:
                    entry["ttft_seconds"] = arrivals[0] - before
                    entry["decode_seconds"] = arrivals[1] - arrivals[0]
                Path(args.output).write_text(json.dumps(report, indent=2))
                print(json.dumps(entry), flush=True)
    llm.send_result = send_result
    if args.timings:
        torch.cuda.synchronize()
        report["timings"] = {}
        for name, phase, begin, end, elapsed in timing_events:
            entry = report["timings"].setdefault(f"{phase}:{name}",
                {"calls": 0, "host_seconds": 0., "stream_seconds": 0.})
            entry["calls"] += 1
            entry["host_seconds"] += elapsed
            entry["stream_seconds"] += begin.elapsed_time(end) / 1000
        for name, original in timing_originals.items():
            delattr(llm._mtp_runner, name)
    report["parity"] = [
        {"repeat": repeat, "prompt": prompt,
         "equal": next(r["token_ids"] for r in report["runs"]
                       if r["repeat"] == repeat and r["prompt"] == prompt and r["mtp"] == 0)
                  == next(r["token_ids"] for r in report["runs"]
                          if r["repeat"] == repeat and r["prompt"] == prompt and r["mtp"] == mtp_mode)}
        for repeat in range(args.repeats) for prompt in prompts
    ]
    if args.validate:
        sp = SamplingParams(max_tokens=args.tokens, ignore_eos=True)
        checks = {}
        object.__setattr__(llm.config, "mtp_speculative_tokens", 0)
        expected = llm.generate([prompts[0]], sp)[0]
        object.__setattr__(llm.config, "mtp_speculative_tokens", mtp_mode)
        original = MTPRunner._one

        def rejected(self, req, phase):
            if phase == "decode" and req.uid in self.drafts:
                self.drafts[req.uid] = torch.zeros_like(self.drafts[req.uid])
            return original(self, req, phase)

        MTPRunner._one = rejected
        accepted_before = llm._mtp_runner.accepted
        proposed_before = llm._mtp_runner.proposed
        try:
            actual = llm.generate([prompts[0]], sp)[0]
        finally:
            MTPRunner._one = original
        checks["forced_rejection"] = {
            "equal": actual["token_ids"] == expected["token_ids"],
            "accepted": llm._mtp_runner.accepted - accepted_before,
            "proposed": llm._mtp_runner.proposed - proposed_before,
            "expected": expected, "actual": actual,
        }
        long_prompt = "hello " * 140 + "Complete: one two three"
        if args.validation_prompt_file:
            long_prompt = Path(args.validation_prompt_file).read_text(encoding="utf-8")
        object.__setattr__(llm.config, "mtp_speculative_tokens", 0)
        expected = llm.generate([long_prompt], sp)[0]
        object.__setattr__(llm.config, "mtp_speculative_tokens", mtp_mode)
        actual = llm.generate([long_prompt], sp)[0]
        checks["chunked"] = {"equal": actual["token_ids"] == expected["token_ids"],
                             "expected": expected, "actual": actual}
        singles = [llm.generate([prompt], sp)[0] for prompt in prompts]
        concurrent = llm.generate(prompts, sp)
        checks["concurrent"] = {"equal": [x["token_ids"] for x in singles]
                                == [x["token_ids"] for x in concurrent],
                                "separate": singles, "simultaneous": concurrent}
        report["checks"] = checks
        report["contexts"] = []
        for length in filter(None, args.validation_contexts.split(",")):
            length = int(length)
            fragment = llm.tokenizer.encode("Cats are curious, playful companions. They like warm places. ")
            tail = llm.tokenizer.encode("\nSummarize the passage in a few sentences:")
            ids = (fragment * (length // len(fragment) + 1))[:length - len(tail)] + tail
            outputs, timings = [], []
            for mode in (0, mtp_mode):
                object.__setattr__(llm.config, "mtp_speculative_tokens", mode)
                before = time.perf_counter()
                outputs.append(llm.generate([ids], sp)[0])
                timings.append(time.perf_counter() - before)
            report["contexts"].append(dict(input_tokens=len(ids), equal=outputs[0]["token_ids"] == outputs[1]["token_ids"],
                                           seconds=timings, outputs=outputs))
            Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if args.sampling_smoke:
        report["sampling"] = []
        object.__setattr__(llm.config, "mtp_speculative_tokens", mtp_mode)
        for temperature, top_k, top_p in ((.7, -1, 1.), (.8, 20, .9), (1., 1, .8)):
            before = time.perf_counter()
            counts = llm._mtp_runner.proposed, llm._mtp_runner.accepted
            results = llm.generate(prompts, SamplingParams(temperature=temperature,
                top_k=top_k, top_p=top_p, max_tokens=args.sampling_tokens, ignore_eos=True))
            assert all(len(result["token_ids"]) == args.sampling_tokens for result in results)
            report["sampling"].append({"temperature": temperature, "top_k": top_k, "top_p": top_p,
                "seconds": time.perf_counter() - before,
                "proposed": llm._mtp_runner.proposed - counts[0],
                "accepted": llm._mtp_runner.accepted - counts[1], "outputs": results})
    if args.profile:
        object.__setattr__(llm.config, "mtp_speculative_tokens", mtp_mode)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            llm.generate([prompts[0]], SamplingParams(max_tokens=args.tokens, ignore_eos=True))
        report["profile_cpu"] = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
        report["profile_cuda"] = prof.key_averages().table(sort_by="self_device_time_total", row_limit=25)
    if args.draft_diagnostic:
        runner = llm._mtp_runner
        report["draft_diagnostic"] = []
        original = runner._head

        def compared(ids, residual, batch, *, compute_logits=True):
            if not batch.use_decode_moe or len(report["draft_diagnostic"]) >= 20:
                return original(ids, residual, batch, compute_logits=compute_logits)
            saved = runner._snapshot(batch.reqs[0])
            with runner.engine.ctx.forward_batch(batch):
                eager = runner.engine.model.forward_mtp(ids, residual, batch,
                    return_residual=True, compute_logits=compute_logits)
            eager = tuple(x.clone() if x is not None else None for x in eager)
            runner._restore(batch.reqs[0], saved)
            actual = runner._head_graph(ids, residual, batch, compute_logits=compute_logits)
            report["draft_diagnostic"].append({"rows": ids.numel(),
                "position": batch.reqs[0].cached_len,
                "equal": [torch.equal(x, y) if x is not None else y is None
                          for x, y in zip(eager, actual)],
                "max_abs": [(x.float() - y.float()).abs().max().item() if x is not None else 0
                            for x, y in zip(eager, actual)]})
            return actual

        runner._head = compared
        try:
            llm.generate([prompts[0]], SamplingParams(max_tokens=64, ignore_eos=True))
        finally:
            del runner._head
    if args.compare_draft_norms:
        from freetoken.models.qwen4_exp.hc import GroupedPlusOneRMSNorm

        head = llm.engine.model.mtp
        correct = head.pre_fc_norm_hidden
        legacy = GroupedPlusOneRMSNorm(correct.size, correct.eps, head.hc_count)
        legacy.weight = correct.weight
        report["draft_norms"] = []
        object.__setattr__(llm.config, "mtp_speculative_tokens", mtp_mode)
        try:
            for label, norm in (("legacy_grouped", legacy), ("global", correct)):
                head.pre_fc_norm_hidden = norm
                getattr(llm._mtp_runner, "_draft_graphs", {}).clear()
                for prompt in prompts:
                    before = time.perf_counter()
                    counts = (llm._mtp_runner.proposed, llm._mtp_runner.accepted)
                    result = llm.generate([prompt], SamplingParams(max_tokens=args.tokens, ignore_eos=True))[0]
                    report["draft_norms"].append({"norm": label, "prompt": prompt,
                        "seconds": time.perf_counter() - before,
                        "proposed": llm._mtp_runner.proposed - counts[0],
                        "accepted": llm._mtp_runner.accepted - counts[1], **result})
        finally:
            head.pre_fc_norm_hidden = correct
            getattr(llm._mtp_runner, "_draft_graphs", {}).clear()
    if args.compare_graphs:
        report["graph_comparison"] = []
        prior = os.environ.get("FREETOKEN_MTP_CUDA_GRAPH")
        object.__setattr__(llm.config, "mtp_speculative_tokens", mtp_mode)
        try:
            for repeat in range(args.repeats):
                for enabled in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    os.environ["FREETOKEN_MTP_CUDA_GRAPH"] = str(enabled)
                    llm.generate([prompts[0]], SamplingParams(max_tokens=8, ignore_eos=True))
                    for prompt in prompts:
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        result = llm.generate([prompt], SamplingParams(max_tokens=args.tokens, ignore_eos=True))[0]
                        torch.cuda.synchronize()
                        expected = next(r["token_ids"] for r in report["runs"]
                                        if r["mtp"] == 0 and r["prompt"] == prompt)
                        report["graph_comparison"].append({"repeat": repeat,
                            "graph": enabled, "prompt": prompt,
                            "seconds": time.perf_counter() - before,
                            "equal": result["token_ids"] == expected, **result})
        finally:
            if prior is None:
                os.environ.pop("FREETOKEN_MTP_CUDA_GRAPH", None)
            else:
                os.environ["FREETOKEN_MTP_CUDA_GRAPH"] = prior
    if args.compare_legacy_linear:
        def legacy_linear(x, weight, bias=None):
            return torch.cat([torch.nn.functional.linear(row, weight, bias) for row in x.split(1)])

        report["legacy_linear"] = []
        with patch("freetoken.layers.linear.rowwise_linear", legacy_linear), patch.dict(
            os.environ, {"FREETOKEN_MTP_CUDA_GRAPH": "0", "FREETOKEN_MTP_DRAFT_GRAPH": "0"}
        ):
            object.__setattr__(llm.config, "mtp_speculative_tokens", mtp_mode)
            for prompt in prompts:
                result = llm.generate([prompt], SamplingParams(max_tokens=args.tokens, ignore_eos=True))[0]
                expected = next(r["token_ids"] for r in report["runs"]
                                if r["mtp"] == 0 and r["prompt"] == prompt)
                report["legacy_linear"].append({"prompt": prompt,
                    "equal": result["token_ids"] == expected, **result})
    if args.state_diagnostic:
        report["state_diagnostic"] = []
        original = llm._forward
        baseline = {}
        pool = llm.engine.linear_state_pool
        target_layers = [i for layer, i in pool._local_index.items() if layer < llm.engine.model._config.num_layers]

        def fingerprint(req):
            slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
            values = {}
            for name, tensor in (("conv", pool.conv_states), ("ssm", pool.recurrent_states),
                                 *pool.slot_states.items()):
                value = tensor[:, slot]
                if name in ("conv", "ssm"):
                    value = value[target_layers]
                value = value.float().flatten(1)
                values[name] = torch.stack((value.sum(1), value.abs().sum(1)), 1).cpu()
            return values

        def traced(forward_input):
            result = original(forward_input)
            req = forward_input.batch.reqs[0]
            pos = req.cached_len
            values = fingerprint(req)
            if llm.config.mtp_speculative_tokens == 0:
                baseline[pos] = values
            elif pos in baseline:
                report["state_diagnostic"].append({"position": pos,
                    "delta": {name: (value - baseline[pos][name]).abs().tolist()
                              for name, value in values.items()}})
            return result

        llm._forward = traced
        try:
            for mode in (0, mtp_mode):
                object.__setattr__(llm.config, "mtp_speculative_tokens", mode)
                llm.generate([prompts[0]], SamplingParams(max_tokens=args.tokens, ignore_eos=True))
        finally:
            llm._forward = original
    report["acceptance_lengths"] = llm._mtp_runner.acceptance_lengths
    Path(args.output).write_text(json.dumps(report, indent=2))
    llm.shutdown()
    checks = [*report["parity"], *report.get("checks", {}).values(),
              *report.get("graph_comparison", []), *report.get("legacy_linear", []),
              *report.get("contexts", [])]
    if not all(check["equal"] for check in checks):
        raise SystemExit("MTP output parity failed; see the saved report")
    if any(not all(check["equal"]) for check in report.get("draft_diagnostic", [])):
        raise SystemExit("MTP draft graph parity failed; see the saved report")


if __name__ == "__main__":
    main()
