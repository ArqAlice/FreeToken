"""Instrumented MTP diagnosis, deliberately separate from throughput measurements."""

import time

import torch
from flashlib.kernels.slot_cache import Stat


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(map(tensor_bytes, value.values()))
    if isinstance(value, (list, tuple)):
        return sum(map(tensor_bytes, value))
    return 0


def collect_diagnostics(llm, ids, sampling, *, routing_trace=False):
    runner, cache = llm._mtp_runner, llm.engine.moe_offload_cache
    if cache is None or not cache.collect_stats or cache.decode_target != "gpu":
        raise ValueError("MTP diagnostics require GPU expert offload with moe_collect_stats enabled")
    rows, originals = [], []
    context = {"phase": "outside", "cycle": -1}
    routes, initial = [], {}

    if routing_trace:
        fn = cache.ensure_experts
        originals.append((cache, 'ensure_experts', fn))

        def trace_routes(layer_id, expert_ids):
            if not initial:
                initial.update(ids=cache.id_of_slot.clone(), usage=cache.usage.clone())
            routes.append((layer_id, expert_ids.clone(), dict(context)))
            return fn(layer_id, expert_ids)

        cache.ensure_experts = trace_routes

    def graph_counts():
        return len(runner._target_graphs), len(runner._draft_graphs)

    def wrap(owner, name):
        fn = getattr(owner, name)
        originals.append((owner, name, fn))

        def measured(*args, **kwargs):
            if name == "_one":
                context["phase"] = args[1]
                context["cycle"] += 1
            row = dict(operation=name, **context)
            if name == "_one":
                row["cached_len"] = args[0].cached_len
            counts = runner.proposed, runner.accepted
            graphs = graph_counts()
            track_experts = name in ("_target", "_draft", "_lookahead")
            before_stats = cache.lru_stats.clone() if track_experts else None
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            before = time.perf_counter()
            result = fn(*args, **kwargs)
            row["host_seconds"] = time.perf_counter() - before
            end.record()
            row["events"] = begin, end
            row["captured_graph"] = graph_counts() != graphs
            if track_experts:
                row["expert_delta"] = cache.lru_stats - before_stats
            if name == "_one":
                row.update(proposed=runner.proposed - counts[0], accepted=runner.accepted - counts[1],
                           emitted=result.numel())
            if name == "_target":
                row["query_rows"] = args[0].input_ids.numel()
            if name == "_snapshot":
                row["state_copy_bytes"] = tensor_bytes([value for _, value in result[1]]) + tensor_bytes(result[2])
            if name == "_restore":
                saved = args[1]
                row["state_copy_bytes"] = tensor_bytes(saved[2])
                if not kwargs.get("ring_only", False):
                    row["state_copy_bytes"] += tensor_bytes([value for _, value in saved[1]])
            rows.append(row)
            return result

        setattr(owner, name, measured)

    cache.reset_stats()
    try:
        for name in ("_one", "_target", "_draft", "_lookahead", "_batch", "_snapshot", "_restore", "_commit_prefix"):
            wrap(runner, name)
        for name in ("probabilities", "sample_probs", "sample_speculative"):
            wrap(llm.engine.sampler, name)
        torch.manual_seed(20260910)
        output = llm.generate([ids], sampling)[0]
        torch.cuda.synchronize()
    finally:
        for owner, name, fn in reversed(originals):
            setattr(owner, name, fn)
    bytes_per_expert = [sum(tensors[layer][0].numel() * tensors[layer].element_size()
                            for tensors in cache.bank_sources.values())
                        for layer in range(cache.num_layers)]
    captured_cycles = {row["cycle"] for row in rows if row["captured_graph"]}
    aggregates = {}
    for row in rows:
        begin, end = row.pop("events")
        row["stream_seconds"] = begin.elapsed_time(end) / 1000
        delta = row.pop("expert_delta", None)
        if delta is not None:
            values = delta.cpu().tolist()
            row["expert_active_union"] = sum(v[Stat.ACTIVE] for v in values)
            row["expert_misses"] = sum(v[Stat.MISS] for v in values)
            row["expert_layer_calls"] = sum(v[Stat.CALLS] for v in values)
            row["expert_fetch_bytes"] = sum(v[Stat.MISS] * b for v, b in zip(values, bytes_per_expert))
            row["expert_per_layer"] = [dict(layer=layer, active=v[Stat.ACTIVE], misses=v[Stat.MISS],
                                             calls=v[Stat.CALLS]) for layer, v in enumerate(values) if v[Stat.CALLS]]
        row["steady_decode"] = row["phase"] == "decode" and row["cycle"] not in captured_cycles
        key = f'{"steady_decode" if row["steady_decode"] else row["phase"] + "_nonsteady"}/{row["operation"]}'
        total = aggregates.setdefault(key, {"calls": 0})
        total["calls"] += 1
        for field in ("host_seconds", "stream_seconds", "proposed", "accepted", "emitted", "query_rows",
                      "state_copy_bytes", "expert_active_union", "expert_misses", "expert_layer_calls", "expert_fetch_bytes"):
            if field in row:
                total[field] = total.get(field, 0) + row[field]
    return dict(
        instrumented=True, quant_format=cache.quant_format, history_bytes=tensor_bytes(runner._verify_history),
        history_steps=getattr(runner, "_history_steps", 5),
        draft_state_isolated=getattr(runner, "_draft_state_isolated", False),
        bytes_per_expert=bytes_per_expert, generated_tokens=len(output["token_ids"]),
        notes=["Nested operation times overlap and must not be added together.",
               "CUDA event intervals include stream idle time; these are not exclusive kernel times.",
               "Graph capture cycles are excluded from steady decode aggregates.",
               "Fetch bytes are cache misses times packed bank bytes, not measured PCIe traffic.",
               "Active union counts are unique experts per layer invocation, not cross-row overlap.",
               "Copy bytes cover snapshots/restores only, excluding commit and graph staging."],
        aggregate=aggregates, cycles=rows,
        routing_trace=(dict(initial={k: v.cpu().tolist() for k, v in initial.items()},
                            experts_per_layer=cache.num_experts, capacity=cache.cache_size,
                            routes=[dict(layer=layer, ids=route.cpu().tolist(), **ctx)
                                    for layer, route, ctx in routes]) if routing_trace else None),
    )
