from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from freetoken.scheduler.mtp import MTPRunner
from freetoken.core import Req, SamplingParams
from freetoken.engine.speculative import verification_distribution
from freetoken.scheduler.mtp_policy import MTPPolicy


def _observe_cost(policy, depth, cost, emitted=1):
    for _ in range(policy.min_samples):
        policy.observe(depth, cost * emitted, emitted)


def test_auto_stops_slow_drafts_and_does_not_resume_in_request():
    policy = MTPPolicy(warmup_samples=0)
    policy.context(60)
    _observe_cost(policy, 0, .02)
    assert policy.depth == 1
    _observe_cost(policy, 1, .03, 2)
    assert policy.stopped and policy.depth == 0
    policy.context(32768)
    _observe_cost(policy, 1, .001, 2)
    assert policy.stopped and policy.depth == 0
    assert MTPPolicy().stopped is False


@pytest.mark.parametrize("costs,best", [([.02, .015, .01, .008], 3),
                                        ([.02, .01, .018, .03], 1),
                                        ([.02, .014, .009, .017], 2)])
def test_auto_explores_then_selects_time_per_emitted_token(costs, best):
    policy = MTPPolicy(warmup_samples=0)
    policy.context(60)
    for depth, cost in enumerate(costs):
        assert policy.depth == depth
        _observe_cost(policy, depth, cost, depth + 1)
    assert policy.depth == best and not policy.stopped
    policy.context(8192)
    assert policy.depth == 0 and not policy.samples


def test_auto_requires_samples_and_ignores_empty_cycles():
    policy = MTPPolicy(max_depth=1, warmup_samples=0)
    policy.observe(0, .01, 0)
    assert not policy.samples
    for _ in range(3):
        policy.observe(0, .02, 1)
    assert policy.depth == 0
    policy.observe(0, .02, 1)
    _observe_cost(policy, 1, .01, 2)
    assert policy.depth == 1
    for _ in range(32):
        policy.observe(1, .06, 1)
    assert policy.stopped


def test_stopped_auto_never_generates_or_retains_drafts():
    runner = MTPRunner(SimpleNamespace(engine=None, config=SimpleNamespace(
        mtp_auto=True, mtp_speculative_tokens=3)))
    req = SimpleNamespace(uid=1, cached_len=60)
    runner._policy(req).stopped = True
    for values in (runner.drafts, runner.draft_probs, runner.draft_states, runner.pending):
        values[1] = object()
    runner._draft(req, 60, None, None, torch.tensor(1))
    assert not any((runner.drafts, runner.draft_probs, runner.draft_states, runner.pending))
    runner.forget(1)
    assert not runner._policies


def test_auto_excludes_initial_cold_cache_and_refreshes_baseline():
    policy = MTPPolicy(max_depth=1)
    policy.context(60)
    for _ in range(16):
        policy.observe(0, 1., 1)
    assert not policy.costs
    _observe_cost(policy, 0, .04)
    _observe_cost(policy, 1, .03)
    assert policy.depth == 1
    for _ in range(32):
        policy.observe(1, .03, 1)
    assert policy.depth == 0 and not policy.stopped
    for _ in range(8):
        policy.observe(0, .01, 1)
        if policy.stopped:
            break
    assert policy.stopped


def test_auto_weights_cycles_by_emitted_tokens():
    policy = MTPPolicy(min_samples=100, warmup_samples=0)
    policy.observe(1, .04, 4)
    policy.observe(1, .04, 1)
    assert policy.costs[1] == pytest.approx(.04 / 3.25)


def test_auto_hysteresis_does_not_keep_an_unprofitable_depth():
    policy = MTPPolicy(depth=1, probing=False, costs={0: .02, 1: .02, 2: .019})
    policy._choose()
    assert policy.depth == 2 and not policy.stopped


@pytest.mark.parametrize("captured,ready", [(True, True), (False, False), (False, True)])
def test_auto_uses_completed_uncaptured_timings_once_per_batch(captured, ready):
    runner = MTPRunner(SimpleNamespace(engine=None, config=SimpleNamespace(
        mtp_auto=True, mtp_speculative_tokens=3)))
    for uid in (1, 2):
        runner._policies[uid] = MTPPolicy(warmup_samples=0, bucket=0)
    begin = SimpleNamespace(elapsed_time=lambda end: end.ms)
    baseline = SimpleNamespace(ms=20.)
    end = SimpleNamespace(ms=24., query=lambda: ready)
    runner._timings.append(([(1, 0), (2, 0)], None, 0, 2, .001,
                            captured, begin, baseline, end))
    runner._drain_timings()
    for policy in runner._policies.values():
        assert bool(policy.costs) == (ready and not captured)
        if policy.costs:
            assert policy.costs[0] == pytest.approx(.01)
    if ready:
        assert runner.auto_stats[0]["tokens"] == 2
        assert runner.auto_stats[0]["seconds"] == pytest.approx(.024)
    assert len(runner._timings) == int(not ready)


@pytest.mark.parametrize("stopped", [False, True])
def test_auto_zero_uses_standard_batch_and_aligns_draft(monkeypatch, stopped):
    from freetoken.core import Batch
    from freetoken.engine.engine import ForwardOutput

    class Event:
        def __init__(self, **kw): pass
        def record(self, stream): pass
        def query(self): return False

    monkeypatch.setattr(torch.cuda, "Event", Event)
    reqs = [Req(torch.tensor([1, 2, 3]), i, 2, 3, i, SamplingParams(temperature=1.), None) for i in (0, 1)]
    batch = Batch(reqs, phase="decode")
    batch.input_ids = torch.tensor([3, 4])
    engine = SimpleNamespace(stream=None, mtp_target_residual=torch.tensor([[10.], [20.]]))
    calls, drafts = [], []
    def normal(inputs):
        calls.append(inputs)
        for req in reqs:
            req.complete_one()
        tokens = torch.tensor([5, 6], dtype=torch.int32)
        return ForwardOutput(tokens, tokens, Event())
    runner = MTPRunner(SimpleNamespace(engine=engine, config=SimpleNamespace(
        mtp_auto=True, mtp_speculative_tokens=3), _forward_standard=normal))
    for req in reqs:
        runner._policy(req).stopped = stopped
    runner._draft = lambda *args: drafts.append(args)
    inputs = SimpleNamespace(batch=batch)
    result = runner.forward(inputs)
    assert calls == [inputs]
    assert result.next_tokens_cpu.tolist() == [5, 6]
    assert len(drafts) == (0 if stopped else 2)
    if not stopped:
        for i, args in enumerate(drafts):
            assert args[0] is reqs[i] and args[1] == reqs[i].cached_len - 1
            assert args[2].item() == (i + 1) * 10
            assert args[3].item() == i + 3 and args[4].item() == i + 5
    assert len(runner._timings) == 1


def test_auto_greedy_zero_keeps_single_row_arithmetic(monkeypatch):
    from freetoken.core import Batch
    class Event:
        def record(self, stream): pass
    monkeypatch.setattr(torch.cuda, "Event", Event)
    reqs = [Req(torch.tensor([1, 2, 3]), i, 2, 3, i, SamplingParams(), None) for i in (0, 1)]
    runner = MTPRunner(SimpleNamespace(engine=SimpleNamespace(device="cpu", stream=None),
        config=SimpleNamespace(mtp_auto=True, mtp_speculative_tokens=3),
        decode_manager=SimpleNamespace(filter_reqs=lambda reqs: None)))
    calls = []
    runner._timed_one = lambda req, phase: (calls.append(req.uid) or torch.tensor([req.uid + 5]))
    result = runner.forward(SimpleNamespace(batch=Batch(reqs, phase="decode")))
    assert calls == [0, 1]
    assert result.next_tokens_cpu.tolist() == [[5, -1, -1, -1], [6, -1, -1, -1]]


@pytest.mark.parametrize("ready", [False, True])
def test_auto_calibration_maintains_head_but_only_samples_before_probe(ready):
    runner = MTPRunner(SimpleNamespace(engine=None, config=SimpleNamespace(
        mtp_auto=True, mtp_speculative_tokens=3)))
    req = SimpleNamespace(uid=1, cached_len=60, remain_len=10, sampling_params=SamplingParams())
    policy = runner._policy(req)
    policy.warmup = policy.warmup_samples if ready else 0
    policy.samples[0] = policy.min_samples - 1
    batch = SimpleNamespace(reqs=[SimpleNamespace(cached_len=60, extend_len=1)])
    calls = []
    def head(ids, residual, batch, *, compute_logits):
        calls.append((ids.tolist(), residual.tolist(), compute_logits))
        return (torch.tensor([[0., 2.]]) if compute_logits else None), torch.ones(1, 2)
    runner._head = head
    runner._draft(req, 60, torch.tensor([[3., 4.]]), torch.tensor([7]), torch.tensor(8), batch)
    assert calls == [([8], [[3., 4.]], ready)]
    assert bool(runner.drafts) == ready
    assert bool(runner.draft_states) == ready


def test_chunked_prefill_skips_logits_and_sampling():
    from freetoken.scheduler.prefill import ChunkedReq

    req = ChunkedReq(torch.tensor([1, 2, 3]), 0, 0, 3, 1, SamplingParams(), None)
    runner = MTPRunner(SimpleNamespace(engine=None, token_pool=torch.tensor([[1, 2, 3]])))
    batch = SimpleNamespace()
    runner._batch = lambda *a: batch
    calls = []
    runner._target = lambda batch, **kw: (calls.append(kw) or None, torch.ones(3, 2))
    runner._draft = lambda *a: calls.append(a[4])
    assert runner._one(req, "prefill").numel() == 0
    assert calls == [{"compute_logits": False}, None]


@pytest.mark.parametrize("enabled,max_bs,expected", [("0", 1, "eager"), ("1", 0, "eager"), ("1", 1, "graph")])
def test_target_graph_respects_disable_controls(monkeypatch, enabled, max_bs, expected):
    monkeypatch.setenv("FREETOKEN_MTP_CUDA_GRAPH", enabled)
    runner = MTPRunner(SimpleNamespace(engine=SimpleNamespace(
        graph_runner=SimpleNamespace(max_graph_bs=max_bs))))
    runner._target_graph = lambda *a, **kw: "graph"
    runner._target_eager = lambda *a, **kw: "eager"
    batch = SimpleNamespace(is_prefill=True, use_decode_moe=True, input_ids=torch.tensor([1, 2]))
    assert runner._target(batch, all_tokens=True) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_verification_graph_restages_slots_and_restores_warmup():
    from contextlib import contextmanager

    current = []

    @contextmanager
    def forward_batch(batch):
        current.append(batch)
        try:
            yield
        finally:
            current.pop()

    states = torch.zeros(1, 2, 1, device="cuda")
    ring = torch.zeros(2, 1, device="cuda")

    def forward(all_tokens=False, compute_logits=True):
        batch = current[-1]
        slot = batch.fla_metadata.cache_indices.long()
        previous = states[0].index_select(0, slot)
        value = previous + batch.input_ids.float().sum() + batch.positions.float().sum()
        states[0].index_copy_(0, slot, value)
        return value.clone(), value.clone()

    host_inputs = []
    def host_context(batch, graph):
        if graph:
            host_inputs.append(batch.ple_input_ids.is_cuda)
        return nullcontext()

    engine = SimpleNamespace(
        ctx=SimpleNamespace(forward_batch=forward_batch),
        model=SimpleNamespace(forward_with_target_residual=forward,
                              forward_host_ctx=host_context),
        cpu_moe_executor=None, stream=torch.cuda.Stream(),
        linear_state_pool=SimpleNamespace(conv_states=states, recurrent_states=states.clone(), slot_states={}),
        kv_cache=SimpleNamespace(_pending_ring=ring),
    )
    runner = MTPRunner(SimpleNamespace(engine=engine))
    for slot, count, expected in ((0, 2, 8), (1, 2, 10), (0, 1, 11), (0, 2, 19)):
        def tensor(values):
            return torch.tensor(values, dtype=torch.int32, device="cuda")

        req = SimpleNamespace(table_idx=slot, linear_slot_idx=None)
        batch = SimpleNamespace(
            reqs=[req], input_ids=tensor([1, 2][:count]), positions=tensor([2 + slot, 3 + slot][:count]),
            out_loc=tensor([2, 3][:count]), linear_table_idx=tensor([slot]),
            attn_metadata=SimpleNamespace(last_indices=tensor([count - 1]), token_to_req=tensor([0] * count),
                cu_seqlens=tensor([0, count]), seq_lens=tensor([2 + count]), ring_slots=tensor([slot]),
                block_table=tensor([[0]])),
            fla_metadata=SimpleNamespace(cu_seqlens=tensor([0, count]), cache_indices=tensor([slot]),
                has_initial_state=torch.tensor([True], device="cuda")),
        )
        batch.ple_input_ids = batch.input_ids
        logits, residual = runner._target_graph(batch)
        assert logits.item() == expected
        assert residual.item() == expected
        assert states[0, slot].item() == expected
    assert len(runner._target_graphs) == 2
    assert host_inputs == [False, True, False, True]
    runner.close()
    assert not runner._target_graphs


def test_verify_batch_exposes_draft_to_disk_ple():
    scheduler = SimpleNamespace(engine=None, _prepare_batch=lambda batch, allocate, sample: None)
    runner = MTPRunner(scheduler)
    req = Req(torch.tensor([1, 2, 3]), 0, 2, 4, 1, SamplingParams(), None)
    batch = runner._batch(req, 2, 4, torch.tensor([3, 7]))
    assert batch.ple_prefix_ids.tolist() == [1, 2]
    assert batch.ple_input_ids.tolist() == [3, 7]
    assert req.input_ids.tolist() == [1, 2, 3]


@pytest.mark.parametrize("backend,copy_ids", [("disk", True), ("pinned", False), ("none", False)])
def test_verify_host_ids_only_for_disk_ple(backend, copy_ids):
    scheduler = SimpleNamespace(engine=SimpleNamespace(config=SimpleNamespace(ple_backend=backend)),
                                _prepare_batch=lambda batch, allocate, sample: None)
    runner = MTPRunner(scheduler)
    req = Req(torch.tensor([1, 2, 3]), 0, 2, 4, 1, SamplingParams(), None)
    batch = runner._batch(req, 2, 4, torch.tensor([3, 7]))
    assert hasattr(batch, "ple_input_ids") == copy_ids


@pytest.mark.parametrize("enabled", [None, "0", "1"])
@pytest.mark.parametrize("temperature,top_k,top_p,stochastic", [
    (0., -1, 1., False), (.8, 1, 1., False), (.8, -1, 1., True), (.8, 20, .95, True),
    (0., -1, .95, False), (.8, 1, .95, False),
])
def test_batched_linear_opt_in_applies_only_to_stochastic_requests(
    monkeypatch, enabled, temperature, top_k, top_p, stochastic,
):
    if enabled is None:
        monkeypatch.delenv("FREETOKEN_MTP_BATCHED_LINEAR", raising=False)
    else:
        monkeypatch.setenv("FREETOKEN_MTP_BATCHED_LINEAR", enabled)
    prepared = []
    scheduler = SimpleNamespace(engine=None, _prepare_batch=lambda batch, allocate, sample:
                                prepared.append(batch.mtp_batched_linear))
    runner = MTPRunner(scheduler)
    req = Req(torch.tensor([1, 2, 3]), 0, 2, 4, 1,
              SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p), None)
    batch = runner._batch(req, 2, 4, torch.tensor([3, 7]))
    expected = enabled == "1" and stochastic
    assert batch.mtp_batched_linear is expected
    assert prepared == [expected]


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("greedy,greedy_draft", [(False, False), (True, False), (False, True)])
def test_verify_commit_and_page_boundary_rollback(accepted, greedy, greedy_draft, monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_GREEDY_DRAFT", str(int(greedy_draft)))
    states = torch.zeros(1, 1, 2)
    ring = torch.zeros(1, 4)
    freed = []
    allocated = []
    cm = SimpleNamespace(
        free_slots=torch.arange(4),
        allocate_paged=lambda reqs: allocated.append((reqs[0].cached_len, reqs[0].device_len)),
        page_table=torch.arange(16).reshape(1, 16),
        _free=lambda slots: freed.append(slots.tolist()),
    )
    scheduler = SimpleNamespace(
        engine=SimpleNamespace(
            linear_state_pool=SimpleNamespace(
                conv_states=states, recurrent_states=states.clone(), slot_states={}),
            kv_cache=SimpleNamespace(_pending_ring=ring),
            sampler=SimpleNamespace(prepare=lambda batch: None,
                probabilities=lambda logits, args: torch.zeros_like(logits).scatter_(
                    1, logits.argmax(-1, keepdim=True), 1.),
                sample_probs=lambda probs: probs.argmax(-1),
                sample_speculative=lambda p, q, n: verification_distribution(p, q, n).argmax(-1)),
        ),
        token_pool=torch.zeros(1, 16, dtype=torch.int32),
        cache_manager=cm, config=SimpleNamespace(page_size=4),
    )
    runner = MTPRunner(scheduler)
    runner.drafts[1] = torch.tensor([3], dtype=torch.int32)
    runner.draft_probs[1] = torch.nn.functional.one_hot(torch.tensor([3]), 8).float()
    runner._batch = lambda req, start, end, ids: ids
    calls = []

    def target(ids, all_tokens=False, compute_logits=True):
        calls.append(ids.tolist())
        states.add_(ids.numel())
        ring.add_(ids.numel())
        logits = torch.zeros(ids.numel() if all_tokens else 1, 8)
        logits[0, 3 if accepted else 4] = 1
        logits[-1, 5] = 2 if all_tokens else 0
        return logits, torch.zeros(ids.numel(), 2)

    runner._target = target
    runner._sample_args = lambda batch: None
    runner._draft = lambda *args: None
    req = SimpleNamespace(
        uid=1, cached_len=3, device_len=4, table_idx=0, linear_slot_idx=None,
        remain_len=4, sampling_params=SimpleNamespace(is_greedy=greedy),
    )
    result = runner._one(req, "decode")
    assert allocated == [(4, 5)]
    if accepted:
        assert result.tolist() == [3, 5]
        assert (req.cached_len, req.device_len) == (5, 6)
        assert not freed
        assert len(calls) == 1
        assert torch.all(states == 2)
        assert torch.all(ring == 2)
    else:
        assert result.tolist() == [4]
        assert (req.cached_len, req.device_len) == (4, 5)
        assert freed == [[4, 5, 6, 7]]
        assert len(calls) == 2
        assert torch.all(states == 1)
        assert torch.all(ring == 1)


def test_chunked_draft_preserves_shifted_residual_alignment():
    calls = []

    def draft(ids, states, batch):
        calls.append(((batch.start, batch.end), ids.tolist(), states.flatten().tolist()))
        return torch.tensor([[0., 1.]])

    runner = MTPRunner(SimpleNamespace(engine=SimpleNamespace(
        ctx=SimpleNamespace(forward_batch=lambda batch: nullcontext()),
        model=SimpleNamespace(forward_mtp=draft),
    )))
    runner._head = lambda ids, states, batch, **kw: (draft(ids, states, batch), states)
    runner._batch = lambda req, start, end, ids, **kwargs: SimpleNamespace(start=start, end=end)
    req = SimpleNamespace(uid=1, sampling_params=SamplingParams())
    runner._draft(req, 0, torch.tensor([[10.], [20.]]), torch.tensor([1, 2]), None)
    runner._draft(req, 2, torch.tensor([[30.], [40.]]), torch.tensor([3, 4]), torch.tensor(5))
    assert calls == [((0, 1), [2], [10.]), ((1, 4), [3, 4, 5], [20., 30., 40.])]
    assert not runner.pending
    runner.forget(1)
    assert not runner.drafts


def test_draft_reuses_matching_target_metadata():
    seen = []
    runner = MTPRunner(SimpleNamespace(engine=SimpleNamespace(
        ctx=SimpleNamespace(forward_batch=lambda batch: nullcontext()),
        model=SimpleNamespace(forward_mtp=lambda ids, states, batch:
                              seen.append(batch) or torch.tensor([[0., 1.]])),
    )))
    def unexpected(*args, **kwargs):
        raise AssertionError("matching metadata must be reused")
    runner._head = lambda ids, states, batch, **kw: (runner.engine.model.forward_mtp(ids, states, batch), states)
    runner._batch = unexpected
    req = SimpleNamespace(uid=1, sampling_params=SamplingParams())
    batch = SimpleNamespace(reqs=[SimpleNamespace(cached_len=5, extend_len=2)])
    histories = {name: object() for name in ("mtp_recurrent_history", "mtp_state_indices",
                                            "mtp_conv_inputs", "mtp_ple_inputs")}
    for name, value in histories.items():
        setattr(batch, name, value)
    runner._draft(req, 5, torch.tensor([[10.], [20.]]), torch.tensor([1, 2]),
                  torch.tensor(3), batch)
    assert len(seen) == 1
    assert seen[0].reqs is batch.reqs
    for name, value in histories.items():
        assert getattr(seen[0], name) is None
        assert getattr(batch, name) is value


@pytest.mark.parametrize("greedy,greedy_draft", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("count,accepted", [(n, a) for n in range(1, 5) for a in range(n + 1)])
@pytest.mark.parametrize("end", [4, 6, 7])
def test_multi_token_prefix_rollback(greedy, greedy_draft, count, accepted, end, monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_GREEDY_DRAFT", str(int(greedy_draft)))
    state = torch.zeros(1, 1, 1)
    ring = torch.zeros(1, 1)
    allocated, freed, calls = [], [], []

    def sample_probs(probs):
        if greedy_draft:
            probs = probs.clone()
            probs[:, 6] = 0
        return probs.argmax(-1)

    sampler = SimpleNamespace(prepare=lambda batch: None,
        probabilities=lambda logits, args: logits,
        sample_probs=sample_probs,
        sample_speculative=lambda p, q, n: verification_distribution(p, q, n).argmax(-1))
    cm = SimpleNamespace(page_table=torch.arange(32).reshape(1, 32), free_slots=torch.arange(4),
        allocate_paged=lambda reqs: allocated.append((reqs[0].cached_len, reqs[0].device_len)),
        _free=lambda slots: freed.extend(slots.tolist()))
    runner = MTPRunner(SimpleNamespace(
        config=SimpleNamespace(page_size=4, mtp_speculative_tokens=4),
        token_pool=torch.zeros(1, 32, dtype=torch.int32), cache_manager=cm,
        engine=SimpleNamespace(sampler=sampler,
            linear_state_pool=SimpleNamespace(conv_states=state, recurrent_states=state.clone(), slot_states={}),
            kv_cache=SimpleNamespace(_pending_ring=ring))))
    draft = torch.arange(1, count + 1, dtype=torch.int32)
    q = torch.nn.functional.one_hot(draft.long(), 8).float()
    runner.drafts[1] = draft[:1]
    if not greedy_draft:
        runner.draft_probs[1] = q[:1]

    def lookahead(req, first, probs, n):
        assert n == count
        state.add_(100)
        ring.add_(100)
        return draft, None if greedy_draft else q

    runner._lookahead = lookahead
    runner._batch = lambda req, start, end, ids: ids
    runner._draft = lambda *args: None

    def target(ids, all_tokens=False, compute_logits=True):
        assert state.item() == 0
        assert ring.item() == 0
        calls.append(ids.tolist())
        state.add_(ids.numel())
        ring.add_(ids.numel())
        predictions = torch.cat((draft.long(), torch.tensor([7])))
        if accepted < count:
            predictions[accepted] = 7
        probabilities = torch.nn.functional.one_hot(predictions, 8).float()
        if greedy_draft:
            # Force the sampler's draw to differ from target argmax.
            probabilities *= .4
            probabilities[:, 6] = .6
        return probabilities, torch.zeros(ids.numel(), 2)

    runner._target = target
    runner._sample_args = lambda batch: None
    req = SimpleNamespace(uid=1, cached_len=end - 1, device_len=end, table_idx=0,
        linear_slot_idx=None, remain_len=count + 1, sampling_params=SimpleNamespace(is_greedy=greedy))
    tokens = runner._one(req, "decode")
    assert tokens.tolist() == draft[:accepted].tolist() + [7]
    assert allocated == [(end, end + count)]
    assert (req.cached_len, req.device_len) == (end + accepted, end + accepted + 1)
    assert state.item() == ring.item() == accepted + 1
    assert calls == [[0] + draft.tolist()] + ([[0] + draft[:accepted].tolist()] if accepted < count else [])
    assert freed == list(range(((end + accepted + 3) // 4) * 4, ((end + count + 3) // 4) * 4))
    assert (runner.proposed, runner.accepted) == (count, accepted)


@pytest.mark.parametrize("count", range(1, 5))
@pytest.mark.parametrize("greedy", [False, True])
def test_greedy_draft_skips_sampling_and_full_vocabulary_probs(count, greedy, monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_GREEDY_DRAFT", "1")
    runner = MTPRunner(SimpleNamespace(
        engine=SimpleNamespace(), config=SimpleNamespace(mtp_speculative_tokens=count)))
    runner._batch = lambda *a, **kw: None
    runner._head = lambda ids, states, batch, **kw: (torch.tensor([[0., 2., 1.]]), states)
    params = SamplingParams(temperature=0. if greedy else .8, top_k=2, top_p=1. if greedy else .9)
    req = SimpleNamespace(uid=1, cached_len=9, sampling_params=params)
    runner._draft(req, 8, torch.ones(1, 2), torch.tensor([0]), torch.tensor(2))
    assert runner.drafts[1].tolist() == [1]
    assert not runner.draft_probs
    tokens, probs = runner._lookahead(req, runner.drafts.pop(1), None, count)
    assert tokens.tolist() == [1] * count
    assert probs is None
    assert (params.temperature, params.top_k, params.top_p) == (0. if greedy else .8, 2, 1. if greedy else .9)


def _qsa_reuse_runner():
    runner = MTPRunner(SimpleNamespace(engine=SimpleNamespace(device="cpu"),
                                       config=SimpleNamespace(mtp_speculative_tokens=3)))
    runner._qsa_reuse = True
    runner._qsa_ratio, runner._qsa_width, runner._qsa_ring = 4, 8, 8
    runner._qsa_layers = [48, 49]
    return runner


def _qsa_stage(runner, position, *, uid=1, width=128, rows=1, reuse=True, logits=True):
    batch = SimpleNamespace(reqs=[SimpleNamespace(uid=uid, cached_len=position - rows + 1)],
                            attn_metadata=SimpleNamespace(block_table=torch.empty(1, width)))
    prepared, owner = runner._qsa_head_batch(torch.empty(rows), batch, logits, reuse)
    assert not hasattr(batch, "mtp_qsa_blocks")
    runner._qsa_owner = owner
    return prepared


@pytest.mark.parametrize("seed,next_position,expected", [(60, 61, True), (62, 63, False),
                                                        (63, 64, False), (64, 65, True),
                                                        (60, 62, False), (60, 60, False)])
def test_qsa_reuse_requires_adjacent_positions_without_group_or_ring_boundary(seed, next_position, expected):
    runner = _qsa_reuse_runner()
    first = _qsa_stage(runner, seed, rows=3, reuse=False)
    assert not first.mtp_qsa_reuse
    addresses = [t.data_ptr() for t in runner._qsa_blocks.values()]
    second = _qsa_stage(runner, next_position)
    assert second.mtp_qsa_reuse is expected
    assert [t.data_ptr() for t in runner._qsa_blocks.values()] == addresses


@pytest.mark.parametrize("change", ["request", "depth", "width", "repair", "chunk", "forget", "close"])
def test_qsa_reuse_invalidates_on_owner_or_shape_changes(change):
    runner = _qsa_reuse_runner()
    _qsa_stage(runner, 60, reuse=False)
    kwargs = {}
    if change == "request":
        kwargs["uid"] = 2
    elif change == "depth":
        runner.scheduler.config.mtp_speculative_tokens = 4
    elif change == "width":
        kwargs["width"] = 256
    elif change == "repair":
        kwargs["reuse"] = False
    elif change == "chunk":
        _qsa_stage(runner, 60, logits=False)
    elif change == "forget":
        runner.forget(1)
    else:
        runner.close()
        assert not runner._qsa_blocks
    assert not _qsa_stage(runner, 61, **kwargs).mtp_qsa_reuse


def test_qsa_reuse_depth_one_never_allocates_or_keeps_an_owner():
    runner = _qsa_reuse_runner()
    runner.scheduler.config.mtp_speculative_tokens = 1
    batch = _qsa_stage(runner, 60)
    assert not hasattr(batch, "mtp_qsa_blocks")
    assert not runner._qsa_blocks and runner._qsa_owner is None


def test_qsa_head_counts_completed_work_and_invalidates_after_failure():
    runner = _qsa_reuse_runner()
    runner.engine.ctx = SimpleNamespace(forward_batch=lambda batch: nullcontext())
    runner.engine.model = SimpleNamespace(forward_mtp=lambda *a, **kw: (None, None))
    req = SimpleNamespace(uid=1, cached_len=60)
    batch = SimpleNamespace(reqs=[req], use_decode_moe=False,
                            attn_metadata=SimpleNamespace(block_table=torch.empty(1, 128)))
    runner._head(torch.empty(1), None, batch)
    req.cached_len += 1
    runner._head(torch.empty(1), None, batch, reuse_qsa=True)
    assert (runner.qsa_refreshed_heads, runner.qsa_reused_heads) == (1, 1)

    def failed(*a, **kw):
        raise RuntimeError("head failed")

    runner.engine.model.forward_mtp = failed
    req.cached_len += 1
    with pytest.raises(RuntimeError, match="head failed"):
        runner._head(torch.empty(1), None, batch, reuse_qsa=True)
    assert runner._qsa_owner is None
    assert (runner.qsa_refreshed_heads, runner.qsa_reused_heads) == (1, 1)


def test_lookahead_passes_head_residual_and_positions():
    seen = []

    def head(ids, hidden, batch, *, return_residual):
        assert return_residual
        seen.append((ids.item(), hidden.item(), batch))
        return torch.tensor([[0., 0., 1.]]), hidden + 10

    runner = MTPRunner(SimpleNamespace(engine=SimpleNamespace(
        ctx=SimpleNamespace(forward_batch=lambda batch: nullcontext()),
        model=SimpleNamespace(forward_mtp=head))))
    runner._head = lambda ids, states, batch, **kw: head(ids, states, batch, return_residual=True)
    runner._batch = lambda req, start, end, ids, **kw: (start, end)
    runner.draft_states[1] = torch.tensor([[5.]])
    req = SimpleNamespace(uid=1, cached_len=9, sampling_params=SamplingParams())
    tokens, probs = runner._lookahead(req, torch.tensor([1]), None, 4)
    assert seen == [(1, 5., (9, 10)), (2, 15., (10, 11)), (2, 25., (11, 12))]
    assert tokens.tolist() == [1, 2, 2, 2]
    assert probs is None
    assert not runner.draft_states


def test_full_cache_falls_back_without_allocating_speculation():
    def unexpected(*args, **kwargs):
        raise AssertionError("no free page is available")

    req = Req(torch.tensor([1, 2, 3, 4]), 0, 3, 4, 1, SamplingParams(), None)
    runner = MTPRunner(SimpleNamespace(
        config=SimpleNamespace(page_size=4, mtp_speculative_tokens=4),
        token_pool=torch.zeros(1, 8, dtype=torch.int32),
        cache_manager=SimpleNamespace(free_slots=[], allocate_paged=unexpected),
        engine=SimpleNamespace(sampler=SimpleNamespace(prepare=lambda batch: None,
            sample=lambda logits, args: logits.argmax(-1)))))
    runner.drafts[1] = torch.tensor([1])
    runner._batch = lambda *args: None
    runner._sample_args = lambda batch: None
    runner._target = lambda batch: (torch.tensor([[0., 1.]]), torch.zeros(1, 2))
    runner._draft = lambda *args: None
    assert runner._one(req, "decode").tolist() == [1]
    assert (req.cached_len, req.device_len) == (4, 5)
    assert runner.proposed == 0


def test_finished_request_does_not_run_draft():
    runner = MTPRunner(SimpleNamespace(engine=None))
    runner._head = lambda *a, **kw: pytest.fail("finished request ran the head")
    runner._draft(SimpleNamespace(remain_len=0), 0, None, None, torch.tensor(1))


def test_sampling_args_reused_and_invalidated():
    calls = []
    runner = MTPRunner(SimpleNamespace(engine=SimpleNamespace(sampler=SimpleNamespace(
        prepare=lambda batch: calls.append(1) or object()))))
    req = SimpleNamespace(uid=7, sampling_params=SamplingParams(temperature=.8))
    batch = SimpleNamespace(reqs=[req])
    first = runner._sample_args(batch)
    assert runner._sample_args(batch) is first
    req.sampling_params.top_p = .9
    assert runner._sample_args(batch) is not first
    runner.forget(req.uid)
    runner._sample_args(batch)
    assert len(calls) == 3


def _history_runner(max_speculative_tokens=4, isolated=True):
    def values(shape, offset=0, dtype=torch.float32):
        size = 1
        for dim in shape:
            size *= dim
        return torch.arange(offset, offset + size, dtype=dtype).reshape(shape)

    pool = SimpleNamespace(
        conv_states=values((2, 4, 3, 3), 100),
        recurrent_states=values((2, 4, 2, 2, 3), 200),
        slot_states={"ple_conv": values((2, 4, 2, 4), 300),
                     "ple_ngram_ctx": values((1, 4, 2), 400, torch.int32)},
        _state_layer_index={"ple_conv": {5: 1, 1: 0}, "ple_ngram_ctx": {}},
    )
    pool.slot_state = lambda name, lid: pool.slot_states[name][pool._state_layer_index[name][lid]]
    engine = SimpleNamespace(
        config=SimpleNamespace(model_config=SimpleNamespace(num_layers=6,
            is_linear_layer=lambda lid: not isolated,
            qwen4_args=SimpleNamespace(ple_layer_ids=(5, 1), ple_state_width=2,
                                      mtp_num_hidden_layers=1))),
        linear_state_pool=pool,
        kv_cache=SimpleNamespace(_pending_ring=values((3, 3, 8, 2), 500)),
    )
    return MTPRunner(SimpleNamespace(engine=engine,
        config=SimpleNamespace(mtp_speculative_tokens=max_speculative_tokens)))


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_history_capacity_follows_initial_maximum_without_replacing_live_buffers(depth):
    runner = _history_runner(depth)
    first = SimpleNamespace(use_decode_moe=True, input_ids=torch.arange(depth + 1))
    assert runner._track_verification(first)
    assert first.mtp_recurrent_history.shape[2] == depth + 1
    runner.scheduler.config.mtp_speculative_tokens = 1
    second = SimpleNamespace(use_decode_moe=True, input_ids=torch.arange(2))
    assert runner._track_verification(second)
    assert second.mtp_recurrent_history is first.mtp_recurrent_history
    runner.scheduler.config.mtp_speculative_tokens = 4
    oversized = SimpleNamespace(use_decode_moe=True, input_ids=torch.arange(depth + 2))
    assert not runner._track_verification(oversized)
    assert runner._verify_history["mtp_recurrent_history"] is first.mtp_recurrent_history


def test_compact_snapshot_omits_recurrence_and_draft_restore_only_touches_ring():
    runner = _history_runner()
    req = SimpleNamespace(table_idx=1, linear_slot_idx=2)
    pool = runner.engine.linear_state_pool
    saved = runner._snapshot(req, recurrent=False)
    assert not any(tensor is pool.recurrent_states for tensor, _ in saved[1])
    assert len(saved[1]) == 1 + len(pool.slot_states)
    assert any(tensor is pool.recurrent_states for tensor, _ in runner._snapshot(req)[1])
    pool.conv_states.add_(10)
    pool.recurrent_states.add_(10)
    expected_conv, expected_recurrent = pool.conv_states.clone(), pool.recurrent_states.clone()
    runner.engine.kv_cache._pending_ring[1].zero_()
    runner._restore(req, saved, ring_only=True)
    torch.testing.assert_close(pool.conv_states, expected_conv)
    torch.testing.assert_close(pool.recurrent_states, expected_recurrent)
    torch.testing.assert_close(runner.engine.kv_cache._pending_ring[1], saved[2])


@pytest.mark.parametrize("isolated", [True, False])
def test_draft_state_isolation_requires_non_recurrent_head(isolated):
    assert _history_runner(isolated=isolated)._draft_state_isolated is isolated


def test_draft_state_isolation_excludes_ple_heads():
    runner = _history_runner()
    runner.engine.config.model_config.qwen4_args.ple_layer_ids = (6,)
    assert not MTPRunner(runner.scheduler)._draft_state_isolated


def test_verification_history_is_shared_and_released(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_STATE_HISTORY", "1")
    runner = _history_runner()
    first = SimpleNamespace(use_decode_moe=True, input_ids=torch.arange(5))
    second = SimpleNamespace(use_decode_moe=True, input_ids=torch.arange(2))
    assert runner._track_verification(first)
    assert runner._track_verification(second)
    assert first.mtp_recurrent_history.shape == (2, 1, 5, 2, 2, 3)
    assert first.mtp_conv_inputs.shape == (2, 5, 3)
    assert first.mtp_state_indices.tolist() == [0]
    assert first.mtp_state_indices.dtype == torch.int32
    assert set(first.mtp_ple_inputs) == {1, 5}
    assert all(inputs.shape == (5, 2) for inputs in first.mtp_ple_inputs.values())
    for name in runner._verify_history:
        assert getattr(first, name) is getattr(second, name)
    runner.close()
    assert runner._verify_history is None
    assert runner._track_verification(second)
    assert second.mtp_recurrent_history is not first.mtp_recurrent_history


@pytest.mark.parametrize("supported,enabled", [(False, "1"), (True, "0")])
def test_verification_history_falls_back_when_disabled_or_unsupported(monkeypatch, supported, enabled):
    monkeypatch.setenv("FREETOKEN_MTP_STATE_HISTORY", enabled)
    runner = _history_runner()
    if not supported:
        runner.engine.config.model_config.qwen4_args = None
    batch = SimpleNamespace()
    assert not runner._track_verification(batch)
    assert runner._verify_history is None
    assert not hasattr(batch, "mtp_recurrent_history")


@pytest.mark.parametrize("decode,count,extra_state", [(False, 5, False), (True, 6, False),
                                                      (True, 0, False), (True, 5, True)])
def test_verification_history_requires_supported_batch_and_states(monkeypatch, decode, count, extra_state):
    monkeypatch.setenv("FREETOKEN_MTP_STATE_HISTORY", "1")
    runner = _history_runner()
    if extra_state:
        runner.engine.linear_state_pool.slot_states["unknown"] = torch.zeros(1, 4, 2)
    batch = SimpleNamespace(use_decode_moe=decode, input_ids=torch.arange(count))
    assert not runner._track_verification(batch)
    assert runner._verify_history is None


@pytest.mark.parametrize("accepted", range(5))
@pytest.mark.parametrize("start", [0, 6, 7])
@pytest.mark.parametrize("compact", [True, False])
def test_commit_prefix_restores_every_state_at_accepted_boundary(monkeypatch, accepted, start, compact):
    monkeypatch.setenv("FREETOKEN_MTP_STATE_HISTORY", "1")
    runner = _history_runner()
    req = SimpleNamespace(table_idx=1, linear_slot_idx=2, cached_len=start)
    batch = SimpleNamespace(use_decode_moe=True, input_ids=torch.tensor([21, 22, 23, 24, 25], dtype=torch.int32))
    assert runner._track_verification(batch)
    for step in range(5):
        batch.mtp_recurrent_history[:, 0, step].fill_(1000 + step)
        batch.mtp_conv_inputs[:, step] = torch.tensor([[10, 20, 30], [40, 50, 60]]) + step
        for lid, inputs in batch.mtp_ple_inputs.items():
            inputs[step] = torch.tensor([70, 80]) + lid * 10 + step
    pool = runner.engine.linear_state_pool
    originals = {name: tensor.clone() for name, tensor in (
        ("conv", pool.conv_states), ("recurrent", pool.recurrent_states), *pool.slot_states.items())}
    old_ring = runner.engine.kv_cache._pending_ring.clone()
    saved = runner._snapshot(req, recurrent=not compact)
    pool.conv_states[:, 2].fill_(-1)
    pool.recurrent_states[:, 2].fill_(-2)
    for tensor in pool.slot_states.values():
        tensor[:, 2].fill_(-3)
    runner.engine.kv_cache._pending_ring[1].add_(10000)
    verified_ring = runner.engine.kv_cache._pending_ring[1].clone()

    count = accepted + 1
    runner._commit_prefix(req, saved, batch, count)
    expected_conv = originals["conv"][:, 2].clone()
    expected_ple = originals["ple_conv"][:, 2].clone()
    expected_ngram = originals["ple_ngram_ctx"][:, 2].clone()
    for step in range(count):
        expected_conv = torch.cat((expected_conv[..., 1:], batch.mtp_conv_inputs[:, step, :, None]), -1)
        for lid, inputs in batch.mtp_ple_inputs.items():
            layer = pool._state_layer_index["ple_conv"][lid]
            expected_ple[layer] = torch.cat((expected_ple[layer, :, 1:], inputs[step, :, None]), -1)
        expected_ngram = torch.cat((expected_ngram[:, 1:], batch.input_ids[step].expand(1, 1)), -1)
    torch.testing.assert_close(pool.recurrent_states[:, 2], torch.full_like(pool.recurrent_states[:, 2], 1000 + accepted))
    torch.testing.assert_close(pool.conv_states[:, 2], expected_conv)
    torch.testing.assert_close(pool.slot_states["ple_conv"][:, 2], expected_ple)
    torch.testing.assert_close(pool.slot_states["ple_ngram_ctx"][:, 2], expected_ngram)
    for step in range(count):
        position = (start + step) % old_ring.shape[2]
        old_ring[1, :, position] = verified_ring[:, position]
    torch.testing.assert_close(runner.engine.kv_cache._pending_ring, old_ring)
    for name, tensor in (("conv", pool.conv_states), ("recurrent", pool.recurrent_states), *pool.slot_states.items()):
        torch.testing.assert_close(tensor[:, [0, 1, 3]], originals[name][:, [0, 1, 3]])


@pytest.mark.parametrize("accepted", range(4))
def test_tracked_rejection_commits_prefix_without_second_target(monkeypatch, accepted):
    monkeypatch.setenv("FREETOKEN_MTP_STATE_HISTORY", "1")
    runner = _history_runner()
    end = 8
    pool = runner.engine.linear_state_pool
    freed, targets, drafts, commits = [], [], [], []
    runner.scheduler.config = SimpleNamespace(page_size=4, mtp_speculative_tokens=4)
    runner.scheduler.token_pool = torch.zeros(3, 32, dtype=torch.int32)
    runner.scheduler.token_pool[1, end - 1] = 6
    runner.scheduler.cache_manager = SimpleNamespace(
        free_slots=torch.arange(4), allocate_paged=lambda reqs: None,
        page_table=torch.arange(96).reshape(3, 32), _free=lambda slots: freed.extend(slots.tolist()))
    req = SimpleNamespace(uid=19, cached_len=end - 1, device_len=end, table_idx=1,
                          linear_slot_idx=2, remain_len=5, sampling_params=SamplingParams())
    draft = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    runner.drafts[req.uid] = draft[:1]
    runner._lookahead = lambda *a: (draft, None)
    runner._batch = lambda req, start, stop, ids: SimpleNamespace(use_decode_moe=True, input_ids=ids)
    residual = torch.arange(15).reshape(5, 3).float()

    def target(batch, *, all_tokens=False, compute_logits=True):
        targets.append(batch.input_ids.tolist())
        assert all_tokens and compute_logits
        for step in range(5):
            batch.mtp_recurrent_history[:, 0, step].fill_(1000 + step)
        batch.mtp_conv_inputs.zero_()
        for inputs in batch.mtp_ple_inputs.values():
            inputs.zero_()
        pool.recurrent_states[:, 2].fill_(-1)
        predictions = torch.tensor([1, 2, 3, 4, 7])
        predictions[accepted] = 7
        return torch.nn.functional.one_hot(predictions, 8).float(), residual

    commit = runner._commit_prefix

    def commit_prefix(req, saved, batch, count):
        commits.append(count)
        commit(req, saved, batch, count)

    def next_draft(req, start, states, ids, token, batch):
        drafts.append((start, states.clone(), ids.tolist(), token.item(), batch.input_ids.tolist()))

    runner._target = target
    runner._commit_prefix = commit_prefix
    runner._draft = next_draft
    result = runner._one(req, "decode")
    assert targets == [[6, 1, 2, 3, 4]]
    assert commits == [accepted + 1]
    assert result.tolist() == draft[:accepted].tolist() + [7]
    assert len(drafts) == 1
    start, states, ids, last, batch_ids = drafts[0]
    assert start == end - 1 and last == 7
    assert ids == batch_ids == [6] + draft[:accepted].tolist()
    torch.testing.assert_close(states, residual[:accepted + 1])
    assert (req.cached_len, req.device_len) == (end + accepted, end + accepted + 1)
    assert freed == (list(range(40, 44)) if accepted == 0 else [])
    torch.testing.assert_close(pool.recurrent_states[:, 2], torch.full_like(pool.recurrent_states[:, 2], 1000 + accepted))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_draft_graph_restages_residual_and_request_slot():
    from contextlib import contextmanager

    current = []

    @contextmanager
    def context(batch):
        current.append(batch)
        try:
            yield
        finally:
            current.pop()

    states = torch.zeros(1, 2, 1, device="cuda")
    def head(ids, residual, batch, **kw):
        slot = batch.fla_metadata.cache_indices.long()
        if batch.attn_metadata.cmp_rows is None:
            batch.attn_metadata.cmp_rows = batch.positions + 0
        value = (states[0].index_select(0, slot) + ids.sum() + residual.sum()
                 + batch.attn_metadata.cmp_rows.sum())
        states[0].index_copy_(0, slot, value)
        return value.clone(), value.clone()

    engine = SimpleNamespace(ctx=SimpleNamespace(forward_batch=context),
        model=SimpleNamespace(forward_mtp=head), cpu_moe_executor=None, stream=torch.cuda.Stream(),
        linear_state_pool=SimpleNamespace(conv_states=states, recurrent_states=states.clone(), slot_states={}),
        kv_cache=SimpleNamespace(_pending_ring=torch.zeros(2, 1, device="cuda")))
    runner = MTPRunner(SimpleNamespace(engine=engine))
    for slot, token, residual, expected in ((0, 1, 1., 3.), (1, 2, 3., 7.), (0, 4, 5., 16.)):
        ids = torch.tensor([token], device="cuda", dtype=torch.int32)
        batch = SimpleNamespace(reqs=[SimpleNamespace(table_idx=slot, linear_slot_idx=None)],
            input_ids=ids, out_loc=None, positions=ids.clone(), linear_table_idx=None,
            attn_metadata=SimpleNamespace(**{name: None for name in (
                "last_indices", "token_to_req", "cu_seqlens", "seq_lens", "ring_slots", "block_table", "cmp_rows")}),
            fla_metadata=SimpleNamespace(cu_seqlens=None, has_initial_state=None,
                cache_indices=torch.tensor([slot], device="cuda", dtype=torch.int32)))
        result, hidden = runner._head_graph(ids, torch.tensor([[residual]], device="cuda"), batch)
        assert result.item() == hidden.item() == expected
        assert states[0, slot].item() == expected
    runner.close()
    assert not runner._draft_graphs
