"""MTP verification with speculative sampling and recurrent-state rollback."""

from copy import copy
from collections import deque
import os
import time

import torch

from freetoken.core import Batch
from freetoken.engine.engine import ForwardOutput
from freetoken.engine.speculative import acceptance_prefix, matching_prefix
from freetoken.utils import div_ceil, init_logger

from .prefill import ChunkedReq
from .mtp_policy import MTPPolicy

logger = init_logger(__name__)


class MTPRunner:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.engine = scheduler.engine
        self.drafts = {}
        self.draft_probs = {}
        self.draft_states = {}
        self.pending = {}
        self.proposed = 0
        self.accepted = 0
        self.acceptance_lengths = {}
        self._policies = {}
        self._timings = deque()
        self.auto_stats = {}
        self._target_graphs = {}
        self._draft_graphs = {}
        self._sampling_args = {}
        self._greedy_draft = os.environ.get("FREETOKEN_MTP_GREEDY_DRAFT", "0") == "1"
        self._verify_history = None
        max_drafts = getattr(getattr(scheduler, "config", None), "mtp_speculative_tokens", 4)
        self._history_steps = min(5, max(2, max_drafts + 1))
        config = getattr(getattr(self.engine, "config", None), "model_config", None)
        args = getattr(config, "qwen4_args", None)
        layers = int(getattr(args, "mtp_num_hidden_layers", 0))
        self._draft_state_isolated = bool(layers and all(
            not config.is_linear_layer(lid) and lid not in args.ple_layer_ids
            for lid in range(config.num_layers, config.num_layers + layers)))
        self._qsa_reuse = (os.environ.get("FREETOKEN_MTP_QSA_REUSE", "0") == "1"
                           and self._draft_state_isolated
                           and getattr(args, "index_ratio", 0) > 1)
        self._qsa_blocks = {}
        self._qsa_owner = None
        self.qsa_reused_heads = 0
        self.qsa_refreshed_heads = 0
        if self._qsa_reuse:
            self._qsa_ratio = args.index_ratio
            self._qsa_width = args.index_budget // args.index_ratio
            self._qsa_layers = range(config.num_layers, config.num_layers + layers)
            self._qsa_ring = self.engine.kv_cache.ring_capacity

    def _auto_enabled(self):
        return getattr(getattr(self.scheduler, "config", None), "mtp_auto", False)

    def _policy(self, req):
        policy = self._policies.get(req.uid)
        if policy is None:
            policy = MTPPolicy(max_depth=min(3, self.scheduler.config.mtp_speculative_tokens))
            self._policies[req.uid] = policy
        policy.context(req.cached_len)
        return policy

    def _depth(self, req):
        if self._auto_enabled():
            return self._policy(req).depth
        return getattr(getattr(self.scheduler, "config", None), "mtp_speculative_tokens", 1)

    def _drain_timings(self):
        while self._timings and self._timings[0][-1].query():
            uid, bucket, depth, emitted, host, captured, begin, baseline, end = self._timings.popleft()
            elapsed = max(host, begin.elapsed_time(end) / 1000)
            row = self.auto_stats.setdefault(depth, dict(cycles=0, tokens=0, seconds=0.,
                                                        capture_seconds=0., probe_seconds=0.))
            row["cycles"] += 1
            row["tokens"] += emitted
            row["seconds"] += elapsed
            if captured:
                row["capture_seconds"] += elapsed
                continue
            cost = elapsed
            if baseline is not None:
                cost = max(host, begin.elapsed_time(baseline) / 1000)
                row["probe_seconds"] += max(0., elapsed - cost)
            subjects = uid if isinstance(uid, list) else [(uid, bucket)]
            for request_id, request_bucket in subjects:
                self._observe_timing(request_id, request_bucket, depth, cost, emitted)

    def _observe_timing(self, uid, bucket, depth, cost, emitted):
        policy = self._policies.get(uid)
        if policy is not None and policy.bucket == bucket and not policy.stopped:
            before = policy.depth
            policy.observe(depth, cost, emitted)
            if policy.depth != before:
                self._qsa_owner = None
                if policy.stopped:
                    for values in (self.drafts, self.draft_probs, self.draft_states, self.pending):
                        values.pop(uid, None)
                logger.info_rank0(f"MTP auto request={uid}: depth {before}->{policy.depth}, ms/token="
                                  + str({k: round(v * 1000, 2) for k, v in policy.costs.items()}))

    def _timed_one(self, req, phase):
        if not self._auto_enabled() or phase != "decode":
            return self._one(req, phase)
        policy = self._policy(req)
        if policy.depth == 0:
            batch = Batch([req], phase="decode")
            inputs = self.scheduler._prepare_batch(batch, allocate=False)
            return self._normal(inputs).next_tokens_gpu
        depth, bucket = policy.depth, policy.bucket
        graphs = len(self._target_graphs), len(self._draft_graphs)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(self.engine.stream)
        started = time.perf_counter()
        tokens = self._one(req, phase)
        host = time.perf_counter() - started
        end.record(self.engine.stream)
        captured = graphs != (len(self._target_graphs), len(self._draft_graphs))
        self._timings.append((req.uid, bucket, depth, tokens.numel(), host, captured,
                              begin, None, end))
        return tokens

    def close(self):
        self._timings.clear()
        self._policies.clear()
        self._target_graphs.clear()
        self._draft_graphs.clear()
        self._verify_history = None
        self._qsa_blocks.clear()
        self._qsa_owner = None

    def forget(self, uid):
        self._drain_timings()
        self._policies.pop(uid, None)
        if self._qsa_owner is not None and self._qsa_owner[0] == uid:
            self._qsa_owner = None
        self.drafts.pop(uid, None)
        self.draft_probs.pop(uid, None)
        self.draft_states.pop(uid, None)
        self.pending.pop(uid, None)
        self._sampling_args.pop(uid, None)

    def _sample_args(self, batch):
        req = batch.reqs[0]
        p = req.sampling_params
        key = (p.temperature, p.top_k, p.top_p)
        cached = self._sampling_args.get(req.uid)
        if cached is None or cached[0] != key:
            cached = (key, self.engine.sampler.prepare(batch))
            self._sampling_args[req.uid] = cached
        return cached[1]

    def _batch(self, req, start, end, input_ids, *, host_input=True):
        view = copy(req)
        view.cached_len, view.device_len = start, end
        view.mamba_ping_pong = None
        batch = Batch([view], phase="prefill")
        batch.mtp_batched_linear = (
            not req.sampling_params.is_greedy
            and req.sampling_params.temperature > 0 and req.sampling_params.top_k != 1
            and os.environ.get("FREETOKEN_MTP_BATCHED_LINEAR", "0") == "1"
        )
        disk_ple = getattr(getattr(self.engine, "config", None), "ple_backend", "disk") == "disk"
        if host_input and disk_ple:
            # Disk PLE needs only two history tokens, not a copy of the entire prompt.
            batch.ple_prefix_ids = req.input_ids[max(0, start - 2):start]
            batch.ple_input_ids = (input_ids if os.environ.get("FREETOKEN_MTP_ASYNC_PLE", "1") == "1"
                                   else input_ids.to("cpu"))
        # Tiny verification batches need routed experts only, not a full layer transfer.
        batch.use_decode_moe = start > 0 and input_ids.numel() <= 5
        self.scheduler._prepare_batch(batch, allocate=False, sample=False)
        batch.input_ids = input_ids
        return batch

    def _target(self, batch, *, all_tokens=False, compute_logits=True):
        if (batch.is_prefill and batch.use_decode_moe and 1 <= batch.input_ids.numel() <= 5
                and self.engine.graph_runner.max_graph_bs > 0
                and os.environ.get("FREETOKEN_MTP_CUDA_GRAPH", "1") == "1"):
            return self._target_graph(batch, all_tokens=all_tokens, compute_logits=compute_logits)
        return self._target_eager(batch, all_tokens=all_tokens, compute_logits=compute_logits)

    def _target_eager(self, batch, *, all_tokens=False, compute_logits=True):
        with self.engine.ctx.forward_batch(batch), self.engine.model.forward_host_ctx(batch, False):
            result = self.engine.model.forward_with_target_residual(all_tokens=all_tokens, compute_logits=compute_logits)
        if self.engine.cpu_moe_executor is not None:
            self.engine.cpu_moe_executor.raise_if_unhealthy()
        return result

    def _target_graph(self, batch, *, all_tokens=False, compute_logits=True):
        table = batch.attn_metadata.block_table
        key = (batch.input_ids.numel(), all_tokens, compute_logits,
               table.shape[1] if table is not None else 0,
               getattr(batch, "mtp_recurrent_history", None) is not None,
               getattr(batch, "mtp_batched_linear", False))
        first_replay = key not in self._target_graphs
        if first_replay:
            # Warm kernels on the live slot, then undo recurrent writes before replay.
            # KV writes are at these same positions and are overwritten by the replay.
            saved = self._snapshot(batch.reqs[0])
            self._target_eager(batch, all_tokens=all_tokens, compute_logits=compute_logits)
            self._restore(batch.reqs[0], saved)
            static, bindings = self._static_batch(batch)
            graph = torch.cuda.CUDAGraph()
            with self.engine.ctx.forward_batch(static):
                with torch.cuda.graph(graph, stream=self.engine.stream):
                    output = self.engine.model.forward_with_target_residual(all_tokens=all_tokens, compute_logits=compute_logits)
            self._target_graphs[key] = (graph, static, bindings, output)
        graph, static, bindings, output = self._target_graphs[key]
        self._stage_batch(batch, static, bindings)
        host_batch = batch
        if first_replay and getattr(batch, "ple_input_ids", None) is not None:
            # First launch can block before deferred fill can signal the graph's PLE wait.
            host_batch = copy(batch)
            host_batch.ple_input_ids = batch.ple_input_ids.to("cpu")
        with self.engine.model.forward_host_ctx(host_batch, True):
            graph.replay()
        if self.engine.cpu_moe_executor is not None:
            self.engine.cpu_moe_executor.raise_if_unhealthy()
        return output

    @staticmethod
    def _static_batch(batch):
        static = copy(batch)
        static.attn_metadata = copy(batch.attn_metadata)
        static.fla_metadata = copy(batch.fla_metadata)
        if hasattr(static.attn_metadata, "cmp_rows"):
            # A head starts after QSA slot zero: capture its plan, not warmup buffers.
            static.attn_metadata.cmp_rows = None
        bindings = []
        for owner, names in (
            (static, ("input_ids", "out_loc", "positions", "linear_table_idx")),
            (static.attn_metadata, ("last_indices", "token_to_req", "cu_seqlens",
                                   "seq_lens", "ring_slots", "block_table")),
            (static.fla_metadata, ("cu_seqlens", "cache_indices", "has_initial_state")),
        ):
            for name in names:
                value = getattr(owner, name)
                if value is not None:
                    setattr(owner, name, value.clone())
                    bindings.append((owner, name))
        return static, bindings

    @staticmethod
    def _stage_batch(batch, static, bindings):
        for owner, name in bindings:
            source = (batch if owner is static else batch.attn_metadata
                      if owner is static.attn_metadata else batch.fla_metadata)
            getattr(owner, name).copy_(getattr(source, name))

    def _draft(self, req, start, residual, input_ids, next_token, batch=None):
        if self._auto_enabled() and self._policy(req).stopped:
            self.drafts.pop(req.uid, None)
            self.draft_probs.pop(req.uid, None)
            self.draft_states.pop(req.uid, None)
            self.pending.pop(req.uid, None)
            return
        if next_token is not None and getattr(req, "remain_len", 1) == 0:
            return
        previous = self.pending.pop(req.uid, None)
        if previous is None:
            shifted = input_ids[1:]
            states = residual[:-1]
        else:
            start -= 1
            shifted = input_ids
            states = torch.cat((previous, residual[:-1]), dim=0)
        if next_token is not None:
            shifted = torch.cat((shifted, next_token.reshape(1)))
            states = torch.cat((states, residual[-1:]), dim=0)
        else:
            self.pending[req.uid] = residual[-1:].clone()
        if shifted.numel() == 0:
            return
        if (batch is None or batch.reqs[0].cached_len != start
                or batch.reqs[0].extend_len != shifted.numel()):
            batch = self._batch(req, start, start + shifted.numel(), shifted, host_input=False)
        elif getattr(batch, "mtp_recurrent_history", None) is not None:
            batch = copy(batch)
            for name in ("mtp_recurrent_history", "mtp_state_indices", "mtp_conv_inputs", "mtp_ple_inputs"):
                setattr(batch, name, None)
        proposal = next_token is not None
        if proposal and self._auto_enabled():
            policy = self._policy(req)
            if policy.depth == 0:
                proposal = (policy.warmup >= policy.warmup_samples
                            and policy.samples.get(0, 0) >= policy.min_samples - 1)
        logits, hidden = self._head(shifted, states, batch, compute_logits=proposal)
        if not proposal:
            self.drafts.pop(req.uid, None)
            self.draft_probs.pop(req.uid, None)
            self.draft_states.pop(req.uid, None)
            return
        if next_token is not None and getattr(getattr(self.scheduler, "config", None), "mtp_speculative_tokens", 1) > 1:
            self.draft_states[req.uid] = hidden[-1:].clone()
        if next_token is not None:
            if req.sampling_params.is_greedy or self._greedy_draft:
                self.drafts[req.uid] = logits[-1].argmax().to(torch.int32).reshape(1)
            else:
                args = self._sample_args(batch)
                probs = self.engine.sampler.probabilities(logits[-1:], args)
                self.draft_probs[req.uid] = probs
                self.drafts[req.uid] = self.engine.sampler.sample_probs(probs).to(torch.int32)

    def _lookahead(self, req, draft, probs, count):
        tokens = [draft]
        distributions = [probs] if probs is not None else []
        hidden = self.draft_states.pop(req.uid, None)
        for i in range(1, count):
            position = req.cached_len + i - 1
            batch = self._batch(req, position, position + 1, tokens[-1], host_input=False)
            logits, hidden = self._head(tokens[-1], hidden, batch, reuse_qsa=True)
            if req.sampling_params.is_greedy or self._greedy_draft:
                token = logits[-1].argmax().to(torch.int32).reshape(1)
            else:
                args = self._sample_args(batch)
                distribution = self.engine.sampler.probabilities(logits[-1:], args)
                distributions.append(distribution)
                token = self.engine.sampler.sample_probs(distribution).to(torch.int32)
            tokens.append(token)
        return torch.cat(tokens), torch.cat(distributions) if distributions else None

    def _qsa_head_batch(self, ids, batch, compute_logits, reuse_qsa):
        count = getattr(getattr(self.scheduler, "config", None), "mtp_speculative_tokens", 1)
        if not self._qsa_reuse or count <= 1 or not compute_logits:
            self._qsa_owner = None
            return batch, None
        req = batch.reqs[0]
        position = req.cached_len + ids.numel() - 1
        key = (req.uid, count, batch.attn_metadata.block_table.shape[1],
               (position + 1) // self._qsa_ratio, position // self._qsa_ring, position)
        reuse = (reuse_qsa and ids.numel() == 1 and self._qsa_owner is not None
                 and key[:-1] == self._qsa_owner[:-1] and position == self._qsa_owner[-1] + 1)
        if not self._qsa_blocks:
            self._qsa_blocks.update({lid: torch.empty((1, self._qsa_width), dtype=torch.int32,
                                                     device=self.engine.device)
                                     for lid in self._qsa_layers})
        batch = copy(batch)
        batch.mtp_qsa_blocks = self._qsa_blocks
        batch.mtp_qsa_reuse = reuse
        # Graphs share these fixed buffers. A different request must refresh them,
        # even if it resumes at the same position in a recycled request slot.
        self._qsa_owner = None
        return batch, key

    def _head(self, ids, residual, batch, *, compute_logits=True, reuse_qsa=False):
        batch, owner = self._qsa_head_batch(ids, batch, compute_logits, reuse_qsa)
        runner = getattr(self.engine, "graph_runner", None)
        if (batch.use_decode_moe and ids.numel() <= 5 and runner is not None
                and runner.max_graph_bs > 0
                and os.environ.get("FREETOKEN_MTP_DRAFT_GRAPH", "1") == "1"):
            result = self._head_graph(ids, residual, batch, compute_logits=compute_logits)
        else:
            with self.engine.ctx.forward_batch(batch):
                result = self.engine.model.forward_mtp(ids, residual, batch,
                    return_residual=True, compute_logits=compute_logits)
        self._qsa_owner = owner
        if owner is not None:
            if batch.mtp_qsa_reuse:
                self.qsa_reused_heads += 1
            else:
                self.qsa_refreshed_heads += 1
        return result

    def _head_graph(self, ids, residual, batch, *, compute_logits=True):
        table = batch.attn_metadata.block_table
        key = (ids.numel(), compute_logits, table.shape[1] if table is not None else 0,
               getattr(batch, "mtp_batched_linear", False),
               bool(getattr(batch, "mtp_qsa_blocks", None)), getattr(batch, "mtp_qsa_reuse", False))
        if key not in self._draft_graphs:
            saved = self._snapshot(batch.reqs[0])
            with self.engine.ctx.forward_batch(batch):
                self.engine.model.forward_mtp(ids, residual, batch,
                    return_residual=True, compute_logits=compute_logits)
            self._restore(batch.reqs[0], saved)
            static, bindings = self._static_batch(batch)
            static_ids, static_residual = ids.clone(), residual.clone()
            graph = torch.cuda.CUDAGraph()
            with self.engine.ctx.forward_batch(static):
                with torch.cuda.graph(graph, stream=self.engine.stream):
                    output = self.engine.model.forward_mtp(static_ids, static_residual, static,
                        return_residual=True, compute_logits=compute_logits)
            self._draft_graphs[key] = (graph, static, bindings, static_ids, static_residual, output)
        graph, static, bindings, static_ids, static_residual, output = self._draft_graphs[key]
        self._stage_batch(batch, static, bindings)
        static_ids.copy_(ids)
        static_residual.copy_(residual)
        graph.replay()
        if self.engine.cpu_moe_executor is not None:
            self.engine.cpu_moe_executor.raise_if_unhealthy()
        return output

    def _snapshot(self, req, *, recurrent=True):
        pool = self.engine.linear_state_pool
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        tensors = [pool.conv_states, *([pool.recurrent_states] if recurrent else []), *pool.slot_states.values()]
        states = [(t, t[:, slot].clone()) for t in tensors]
        kv = self.engine.kv_cache
        ring = kv._pending_ring[req.table_idx].clone()
        return slot, states, ring

    def _restore(self, req, saved, *, ring_only=False):
        slot, recurrent, ring = saved
        if not ring_only:
            for tensor, value in recurrent:
                tensor[:, slot].copy_(value)
        self.engine.kv_cache._pending_ring[req.table_idx].copy_(ring)
        # Speculative KV and compressed rows lie beyond the committed length. They
        # remain invisible, and the next write replaces them before they become visible.

    def _history_supported(self, num_tokens, use_decode_moe):
        if os.environ.get("FREETOKEN_MTP_STATE_HISTORY", "1") != "1":
            return False
        config = getattr(self.engine, "config", None)
        args = getattr(getattr(config, "model_config", None), "qwen4_args", None)
        if args is None:
            return False
        if not use_decode_moe or not 1 <= num_tokens <= self._history_steps:
            return False
        pool = self.engine.linear_state_pool
        if set(pool.slot_states) - {"ple_conv", "ple_ngram_ctx"}:
            return False
        return True

    def _track_verification(self, batch):
        ids = getattr(batch, "input_ids", None)
        if not self._history_supported(ids.numel() if ids is not None else 0,
                                       getattr(batch, "use_decode_moe", False)):
            return False
        args = self.engine.config.model_config.qwen4_args
        pool = self.engine.linear_state_pool
        if self._verify_history is None:
            rec, conv = pool.recurrent_states, pool.conv_states
            steps = self._history_steps
            # Allocate for the configured maximum so graph references survive depth changes.
            self._verify_history = dict(
                mtp_recurrent_history=rec.new_empty(rec.shape[0], 1, steps, *rec.shape[2:]),
                mtp_state_indices=torch.zeros(1, dtype=torch.int32, device=rec.device),
                mtp_conv_inputs=conv.new_empty(conv.shape[0], steps, conv.shape[2]),
                mtp_ple_inputs={lid: pool.slot_state("ple_conv", lid).new_empty(steps, args.ple_state_width)
                                for lid in args.ple_layer_ids},
            )
        for name, value in self._verify_history.items():
            setattr(batch, name, value)
        return True

    def _commit_prefix(self, req, saved, batch, count):
        slot, recurrent, old_ring = saved
        pool = self.engine.linear_state_pool
        pool.recurrent_states[:, slot].copy_(batch.mtp_recurrent_history[:, 0, count - 1])
        old_conv = recurrent[0][1]
        inputs = batch.mtp_conv_inputs[:, :count].transpose(1, 2)
        pool.conv_states[:, slot].copy_(torch.cat((old_conv, inputs), dim=-1)[..., -old_conv.shape[-1]:])
        previous = {id(tensor): value for tensor, value in recurrent}
        for name, tensor in pool.slot_states.items():
            old = previous[id(tensor)]
            if name == "ple_conv":
                for lid, inputs in batch.mtp_ple_inputs.items():
                    li = pool._state_layer_index[name][lid]
                    tail = torch.cat((old[li], inputs[:count].T), dim=-1)[..., -old.shape[-1]:]
                    tensor[li, slot].copy_(tail)
            elif name == "ple_ngram_ctx":
                ids = batch.input_ids[:count].to(old.dtype).expand(old.shape[0], -1)
                tensor[:, slot].copy_(torch.cat((old, ids), dim=-1)[..., -old.shape[-1]:])
            else:
                raise RuntimeError(f"MTP prefix restoration does not support slot state {name!r}")
        ring = self.engine.kv_cache._pending_ring[req.table_idx]
        # Verification is shorter than the ring, so accepted writes still survive.
        positions = (torch.arange(count, device=ring.device) + req.cached_len) % ring.shape[1]
        old_ring.index_copy_(1, positions, ring.index_select(1, positions))
        ring.copy_(old_ring)

    def _one(self, req, phase):
        start, end = req.cached_len, req.device_len
        ids = self.scheduler.token_pool[req.table_idx, start:end].clone()
        greedy = req.sampling_params.is_greedy
        draft = self.drafts.pop(req.uid, None)
        draft_probs = self.draft_probs.pop(req.uid, None)
        depth = self._depth(req)
        speculate = phase == "decode" and draft is not None and req.remain_len >= 2 and depth > 0
        if speculate:
            count = min(depth, req.remain_len - 1)
            ps = self.scheduler.config.page_size
            capacity = (div_ceil(end, ps) + len(self.scheduler.cache_manager.free_slots)) * ps
            count = min(count, capacity - end)
            speculate = count > 0
        if not speculate:
            batch = self._batch(req, start, end, ids)
            if phase == "prefill":
                batch.use_decode_moe = False
            if isinstance(req, ChunkedReq):
                _, residual = self._target(batch, compute_logits=False)
                req.complete_one()
                self._draft(req, start, residual, ids, None, batch)
                return ids.new_empty(0)
            logits, residual = self._target(batch)
            args = self._sample_args(batch)
            tokens = self.engine.sampler.sample(logits, args).to(torch.int32)
            req.complete_one()
            self.scheduler.token_pool[req.table_idx, end] = tokens[0]
            self._draft(req, start, residual, ids, tokens[0], batch)
            return tokens

        # The scheduler already allocated through end; reserve only the draft inputs.
        allocation = copy(req)
        allocation.cached_len, allocation.device_len = end, end + count
        self.scheduler.cache_manager.allocate_paged([allocation])
        compact = (self._draft_state_isolated
                   and self._history_supported(ids.numel() + count, start > 0))
        # Draft QSA does not touch target GDN/PLE; history supplies the committed recurrence.
        saved = self._snapshot(req, recurrent=not compact)
        if count > 1:
            draft, draft_probs = self._lookahead(req, draft, draft_probs, count)
            self._restore(req, saved, ring_only=self._draft_state_isolated)
        verify_ids = torch.cat((ids, draft))
        batch = self._batch(req, start, end + count, verify_ids)
        tracked = self._track_verification(batch)
        logits, residual = self._target(batch, all_tokens=True)
        if greedy or self._greedy_draft:
            if greedy:
                verified = logits.argmax(dim=-1).to(torch.int32)
            else:
                args = self._sample_args(batch)
                probs = self.engine.sampler.probabilities(logits, args)
                verified = self.engine.sampler.sample_probs(probs).to(torch.int32)
            # With deterministic proposals, a target draw both verifies and supplies
            # the correction; samples after the first mismatch are discarded.
            accepted = int(matching_prefix(verified, draft).item())
            last = verified[accepted:accepted + 1]
        else:
            args = self._sample_args(batch)
            probs = self.engine.sampler.probabilities(logits, args)
            accepted_gpu = acceptance_prefix(probs[:-1], draft_probs, draft)
            last = self.engine.sampler.sample_speculative(probs, draft_probs, accepted_gpu).to(torch.int32)
            accepted = int(accepted_gpu.item())
        tokens = torch.cat((draft[:accepted], last))
        self.proposed += count
        self.accepted += accepted
        self.acceptance_lengths[accepted] = self.acceptance_lengths.get(accepted, 0) + 1
        if accepted < count:
            if tracked:
                self._commit_prefix(req, saved, batch, accepted + 1)
                residual = residual[:accepted + 1]
            else:
                self._restore(req, saved)
            verify_ids = verify_ids[:accepted + 1]
            batch = self._batch(req, start, end + accepted, verify_ids)
            if not tracked:
                _, residual = self._target(batch, compute_logits=False)
            ps = self.scheduler.config.page_size
            first_unused = div_ceil(end + accepted, ps) * ps
            allocated_end = div_ceil(end + count, ps) * ps
            if allocated_end > first_unused:
                cm = self.scheduler.cache_manager
                cm._free(cm.page_table[req.table_idx, first_unused:allocated_end].clone())
        req.cached_len, req.device_len = end + accepted, end + accepted + 1
        self.scheduler.token_pool[req.table_idx, end:end + tokens.numel()] = tokens
        self._draft(req, start, residual, verify_ids, tokens[-1], batch)
        logger.debug_rank0(
            f"MTP verify: accepted={int(accepted)}, total={self.accepted}/{self.proposed}"
        )
        return tokens

    def forward(self, forward_input):
        batch = forward_input.batch
        self._drain_timings()
        if self._auto_enabled() and batch.phase == "decode":
            # Keep greedy's single-row reduction order across depth and batch changes.
            batch_zero = batch.size == 1 or not any(r.sampling_params.is_greedy for r in batch.reqs)
            if batch_zero and all(self._depth(r) == 0 for r in batch.reqs):
                return self._normal(forward_input)
        rows = [self._timed_one(req, batch.phase) for req in batch.reqs]
        width = self.scheduler.config.mtp_speculative_tokens + 1
        tokens = torch.full((batch.size, width), -1, dtype=torch.int32, device=self.engine.device)
        for i, row in enumerate(rows):
            tokens[i, :row.numel()] = row
        host = tokens.to("cpu", non_blocking=True)
        event = torch.cuda.Event()
        event.record(self.engine.stream)
        self.scheduler.decode_manager.filter_reqs(batch.reqs)
        return ForwardOutput(tokens, host, event)

    def _normal(self, forward_input):
        batch = forward_input.batch
        policies = [self._policy(r) for r in batch.reqs]
        batch.mtp_return_residual = any(not p.stopped for p in policies)
        graphs = len(self._draft_graphs)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(self.engine.stream)
        before = time.perf_counter()
        result = self.scheduler._forward_standard(forward_input)
        host = time.perf_counter() - before
        baseline = torch.cuda.Event(enable_timing=True)
        baseline.record(self.engine.stream)
        for i, (req, policy) in enumerate(zip(batch.reqs, policies)):
            if not policy.stopped:
                self._draft(req, req.cached_len - 1, self.engine.mtp_target_residual[i:i + 1],
                            batch.input_ids[i:i + 1], result.next_tokens_gpu[i])
        self.engine.mtp_target_residual = None
        end.record(self.engine.stream)
        captured = graphs != len(self._draft_graphs)
        # Amortize the ordinary batched decode cost across its emitted tokens.
        subjects = [(r.uid, p.bucket) for r, p in zip(batch.reqs, policies) if not p.stopped]
        self._timings.append((subjects, None, 0 if subjects else "stopped",
                              batch.size, host, captured, begin, baseline if subjects else None, end))
        return result._replace(copy_done_event=end)
