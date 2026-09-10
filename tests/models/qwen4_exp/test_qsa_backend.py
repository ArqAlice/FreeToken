"""The QSA backend behind the real Qwen4ExpAttention layer.

(a) dense-oracle equivalence -- while a request sees at most ``index_budget + index_ratio - 1``
    tokens every complete block is selected, so QSA IS dense attention: the selection must be
    exactly the causal prefix and the layer output must match ``TorchDenseQSAReference`` (fp32)
    and a flashinfer dense run over the same pool;
(b) chunked prefill at unaligned cut points equals one-shot prefill (the dual-source compress);
(c) a captured decode replay equals the eager decode step;
(d) an fp8 KV pool (``--kv-cache-dtype fp8``) keeps block selection bit-identical to the
    16-bit run -- only the selected K/V rows are read back as e4m3 codes -- and the layer
    output stays within quantization error of it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import Fixture, requires_cuda, parsed_config, selection_spy

QSA_LAYER = 3


@pytest.mark.parametrize(
    ("length", "expected_pages"),
    [(0, 128), (64, 128), (8192, 128), (8193, 256), (16385, 512), (1000128, 15627)],
)
def test_active_page_table_buckets(length, expected_pages):
    from freetoken.attention.qsa_sparse import QSASparseAttnBackend

    backend = object.__new__(QSASparseAttnBackend)
    backend.page_size = 64
    backend._block_topk_kernel = object()
    pages = torch.arange(3 * 15627, dtype=torch.int32).reshape(3, 15627) * 64
    backend._block_base_view = lambda: pages
    table_idx = torch.tensor([2, 0], dtype=torch.int64)
    actual = backend._block_table(table_idx, length)
    assert actual.shape == (2, expected_pages)
    assert torch.equal(actual, pages[table_idx, :expected_pages] // 64)
    assert backend._block_table(table_idx).shape == (2, 15627)
    backend._block_topk_kernel = None
    assert backend._block_table(table_idx, length).shape == (2, 15627)


@pytest.mark.parametrize("name", ["logits", "topk_scratch"])
def test_active_score_workspace_reuses_capacity(name):
    from freetoken.attention.qsa_sparse import QSASparseAttnBackend

    backend = object.__new__(QSASparseAttnBackend)
    buffer = torch.empty((4, 4096), dtype=torch.float32)
    backend._graph = {name: buffer}
    actual = backend._scratch(name, 2, 512, dtype=buffer.dtype)
    assert actual.shape == (2, 512)
    assert actual.stride() == (4096, 1)
    assert actual.data_ptr() == buffer.data_ptr()


@requires_cuda
@pytest.mark.parametrize("budget", [16, 2048])
@pytest.mark.parametrize("kv_quant", ["none", "fp8", "nvfp4"])
def test_draft_reuses_saved_blocks_but_updates_tail_and_index_cache(monkeypatch, budget, kv_quant):
    from freetoken.kernel.triton.qsa import expand_qsa_block_indices

    config = parsed_config(budget=budget)
    fixture = Fixture(config, num_pages=16, kv_quant=kv_quant)
    attn = fixture.layer(QSA_LAYER)
    x = _inputs(fixture, [66])[0]
    saved = torch.empty((1, budget // 4), device="cuda", dtype=torch.int32)
    seed = fixture.batch([fixture.req(1, 0, 61)], "prefill")
    seed.mtp_qsa_blocks = {QSA_LAYER: saved}
    attn.forward(x[:61], seed)
    blocks = saved.clone()
    assert (blocks[blocks >= 0] < 15).all()
    assert (blocks >= 0).sum().item() == min(15, budget // 4)

    def unexpected(*a, **kw):
        raise AssertionError("reuse must skip query norm, scores and top-k")

    original_select = fixture.backend._select
    monkeypatch.setattr(fixture.backend, "_select", unexpected)
    for position in (61, 62):
        batch = fixture.batch([fixture.req(1, position, position + 1)], "prefill")
        batch.mtp_qsa_blocks, batch.mtp_qsa_reuse = {QSA_LAYER: saved}, True
        got = attn.forward(x[position:position + 1], batch)
        expected_indices = torch.empty((1, budget + 3), device="cuda", dtype=torch.int32)
        expand_qsa_block_indices(blocks, batch.positions, batch.attn_metadata.seq_lens,
                                 batch.attn_metadata.token_to_req, 4, budget, expected_indices)
        assert position in expected_indices[0].tolist()
        assert not (expected_indices > position).any()
        torch.testing.assert_close(saved, blocks, rtol=0, atol=0)
        reference = fixture.batch([fixture.req(1, position, position + 1)], "prefill")
        monkeypatch.setattr(fixture.backend, "_select", lambda *a, **kw: expected_indices)
        expected = attn.forward(x[position:position + 1], reference)
        assert torch.equal(got, expected)
        if budget == 2048:
            monkeypatch.setattr(fixture.backend, "_select", original_select)
            assert torch.equal(got, attn.forward(x[position:position + 1], reference))
        monkeypatch.setattr(fixture.backend, "_select", unexpected)

    # A full selection at closure must see keys written during reused steps.
    monkeypatch.setattr(fixture.backend, "_select", original_select)
    closure = fixture.batch([fixture.req(1, 63, 64)], "prefill")
    closure.mtp_qsa_blocks = {QSA_LAYER: saved}
    got = attn.forward(x[63:64], closure)
    attn.forward(x[:63], fixture.batch([fixture.req(2, 0, 63)], "prefill"))
    expected = attn.forward(x[63:64], fixture.batch([fixture.req(2, 63, 64)], "prefill"))
    assert torch.equal(got, expected)


@requires_cuda
def test_draft_qsa_reuse_graph_reads_updated_blocks_positions_and_request_slot():
    from freetoken.kernel.triton.qsa import expand_qsa_block_indices

    fixture = Fixture(parsed_config(budget=16), num_pages=16, kv_quant="nvfp4")
    attn = fixture.layer(QSA_LAYER)
    x = _inputs(fixture, [70, 70])
    for slot in (1, 2):
        attn.forward(x[slot - 1][:69], fixture.batch([fixture.req(slot, 0, 69)], "prefill"))
    blocks = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device="cuda")
    static = fixture.batch([fixture.req(1, 61, 62)], "prefill")
    static.mtp_qsa_blocks, static.mtp_qsa_reuse = {QSA_LAYER: blocks}, True
    static_x = x[0][61:62].clone()
    attn.forward(static_x, static)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = attn.forward(static_x, static)
    for slot, position, selected in ((1, 61, [0, 1, 2, 3]), (2, 62, [10, 9, 8, 7]),
                                     (1, 65, [5, 3, 1, 0])):
        live = fixture.batch([fixture.req(slot, position, position + 1)], "prefill")
        static.positions.copy_(live.positions)
        static.out_loc.copy_(live.out_loc)
        for name in ("seq_lens", "ring_slots", "block_table"):
            getattr(static.attn_metadata, name).copy_(getattr(live.attn_metadata, name))
        static_x.copy_(x[slot - 1][position:position + 1])
        blocks.copy_(torch.tensor([selected], device="cuda", dtype=torch.int32))
        graph.replay()
        got = output.clone()
        indices = torch.empty((1, 19), device="cuda", dtype=torch.int32)
        expand_qsa_block_indices(blocks, live.positions, live.attn_metadata.seq_lens,
                                 live.attn_metadata.token_to_req, 4, 16, indices)
        original_select = fixture.backend._select
        fixture.backend._select = lambda *a, **kw: indices
        try:
            expected = attn.forward(static_x, live)
        finally:
            fixture.backend._select = original_select
        assert torch.equal(got, expected)


def _inputs(fixture: Fixture, lengths, extra: int = 0, seed: int = 11):
    generator = torch.Generator(device=fixture.device).manual_seed(seed)
    return [
        torch.randn(
            n + extra, fixture.config.hidden_size, device=fixture.device,
            dtype=fixture.dtype, generator=generator,
        )
        * 0.5
        for n in lengths
    ]


def _assert_selection_is_causal_prefix(indices: torch.Tensor, positions: torch.Tensor) -> None:
    for row, position in enumerate(positions.tolist()):
        selected = indices[row][indices[row] >= 0]
        assert torch.equal(
            selected.sort().values,
            torch.arange(position + 1, dtype=selected.dtype, device=selected.device),
        ), f"row {row} (position {position}) did not select its whole causal prefix"


@requires_cuda
def test_prefill_is_dense_below_the_budget(monkeypatch):
    """bs=3 ragged prefill, longest request exactly at budget + ratio - 1."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths = [2051, 1000, 137]
    inputs = _inputs(fixture, lengths)
    x = torch.cat([row[:n] for row, n in zip(inputs, lengths)])
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    seen = selection_spy(monkeypatch, fixture.backend)
    batch = fixture.batch(reqs, "prefill")
    got = attn.forward(x, batch)
    _assert_selection_is_causal_prefix(seen["indices"], batch.positions)

    fixture.ctx.attn_backend = _dense_oracle(fixture)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


def _dense_oracle(fixture: Fixture):
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference

    return TorchDenseQSAReference(
        fixture.config,
        num_slots=fixture.num_req_slots,
        max_len=4096,
        device=fixture.device,
        dtype=fixture.dtype,
    )


@requires_cuda
def test_decode_is_dense_below_the_budget(monkeypatch):
    """Prefill then five decode steps, sparse path vs the fp32 dense oracle."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411, 64], 5
    inputs = _inputs(fixture, lengths, extra=steps)
    oracle = _dense_oracle(fixture)

    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    seen = selection_spy(monkeypatch, fixture.backend)

    steps_x = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    steps_x += [
        torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)
    ]
    for step, x in enumerate(steps_x):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        fixture.ctx.attn_backend = fixture.backend
        got = attn.forward(x, batch)
        _assert_selection_is_causal_prefix(seen["indices"], batch.positions)
        fixture.ctx.attn_backend = oracle
        reference = attn.forward(x, batch)
        torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
def test_flashinfer_dense_matches_the_sparse_path():
    """The engine's dense FULL backend over the same pool, as an independent oracle."""
    pytest.importorskip("flashinfer")
    from freetoken.attention.fi import FlashInferBackend

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 500
    x = _inputs(fixture, [length])[0]
    req = fixture.req(0, 0, length)
    got = attn.forward(x, fixture.batch([req], "prefill"))

    dense = FlashInferBackend(config)
    fixture.ctx.attn_backend = SimpleNamespace(
        qsa_forward=lambda q, k, v, index, layer_id, batch: dense.forward(
            q, k, v, layer_id, batch
        )
    )
    batch = fixture.batch([req], "prefill")
    dense.prepare_metadata(batch)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
@pytest.mark.parametrize("cut", [1001, 4096, 4097], ids=["unaligned", "page-boundary", "boundary+1"])
def test_chunked_prefill_matches_one_shot(cut: int):
    """Cut points that are not multiples of index_ratio exercise the dual-source compress."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    attn = fixture.layer(QSA_LAYER)
    length = 5000
    x = _inputs(fixture, [length])[0]

    one_shot = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))
    head = fixture.req(1, 0, cut)
    attn.forward(x[:cut], fixture.batch([head], "prefill"))
    tail = fixture.req(1, cut, length)
    got = attn.forward(x[cut:], fixture.batch([tail], "prefill"))
    assert torch.equal(got, one_shot[cut:])


@requires_cuda
def test_decode_graph_replay_matches_eager():
    config = parsed_config()
    fixture = Fixture(config, num_pages=256)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411], 4
    bs = len(lengths)
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    attn.forward(
        torch.cat([row[:n] for row, n in zip(inputs, lengths)]),
        fixture.batch(reqs, "prefill"),
    )

    fixture.backend.init_capture_graph(max_seq_len=fixture.page_table.shape[1], bs_list=[bs])
    dummy = SimpleNamespace(
        table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=2, extend_len=1
    )
    static = {
        "x": torch.zeros(bs, config.hidden_size, device=fixture.device, dtype=fixture.dtype),
        "positions": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
        "out_loc": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
    }
    capture_batch = SimpleNamespace(
        padded_reqs=[dummy] * bs, reqs=[dummy] * bs, phase="decode", size=bs, padded_size=bs,
        is_prefill=False, is_decode=True, positions=static["positions"],
        use_decode_moe=False,
        out_loc=static["out_loc"], attn_metadata=None, active_table_idx=None,
    )
    fixture.backend.prepare_for_capture(capture_batch)
    attn.forward(static["x"], capture_batch)  # warmup, same metadata object as the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()

    for step in range(steps):
        for req in reqs:
            fixture.step(req)
        x = torch.stack([row[n + step] for row, n in zip(inputs, lengths)])
        batch = fixture.batch(reqs, "decode")
        static["x"].copy_(x)
        static["positions"].copy_(batch.positions)
        static["out_loc"].copy_(batch.out_loc)
        fixture.backend.prepare_for_replay(batch)
        # replay must stage into the captured buffers, never reallocate them
        md = batch.attn_metadata
        assert md.block_table.data_ptr() == fixture.backend._graph["block_table"].data_ptr()
        graph.replay()
        replayed = captured_out.clone()
        eager = attn.forward(x, fixture.batch(reqs, "decode"))
        assert torch.equal(replayed, eager), f"graph replay diverged at decode step {step}"


@requires_cuda
def test_row_chunked_scoring_matches_one_chunk(monkeypatch):
    """The scoring workspace bound splits long prefills into row chunks."""
    import freetoken.attention.qsa_sparse as qsa_sparse

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 600
    x = _inputs(fixture, [length])[0]
    whole = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))

    columns = fixture.page_table.shape[1] // config.qwen4_args.index_ratio
    monkeypatch.setattr(qsa_sparse, "_LOGITS_WORKSPACE_BYTES", 64 * columns * 4)
    chunked = attn.forward(x, fixture.batch([fixture.req(1, 0, length)], "prefill"))
    assert torch.equal(chunked, whole)


@requires_cuda
@pytest.mark.parametrize("cached_len", [63, 4097])
@pytest.mark.parametrize("kv_quant", ["none", "nvfp4"])
def test_speculative_active_pages_match_full_capacity(cached_len, kv_quant):
    config = parsed_config()
    fixture = Fixture(config, num_pages=128, kv_quant=kv_quant)
    fixture.page_table = torch.zeros(
        (fixture.num_req_slots, 1000128), dtype=torch.int32, device=fixture.device
    )
    fixture.ctx.page_table = fixture.page_table
    fixture.backend.init_capture_graph(max_seq_len=1000128, bs_list=[4])
    attn = fixture.layer(QSA_LAYER)
    x = _inputs(fixture, [cached_len], extra=4)[0]
    prefix = fixture.req(0, 0, cached_len)
    attn.forward(x[:cached_len], fixture.batch([prefix], "prefill"))
    req = fixture.req(0, cached_len, cached_len + 4)
    batch = fixture.batch([req], "prefill")
    batch.use_decode_moe = True
    assert batch.attn_metadata.block_table.shape[1] == 128
    ring = fixture.pool.pending_ring(0)
    saved_ring = ring.clone()
    bounded = attn.forward(x[cached_len:], batch)
    ring.copy_(saved_ring)
    batch = fixture.batch([req], "prefill")
    batch.use_decode_moe = True
    md = batch.attn_metadata
    md.block_table = fixture.backend._block_table(md.ring_slots.to(torch.int64))
    assert md.block_table.shape[1] == 15627
    full = attn.forward(x[cached_len:], batch)
    assert torch.equal(bounded, full)


@requires_cuda
def test_two_qsa_layers_keep_separate_slab_slots(monkeypatch):
    """Both QSA layers of one forward must hit their own slab slot and ring slice."""
    config = parsed_config(num_layers=8)
    assert config.attention_groups[1].layer_ids == (3, 7)
    fixture = Fixture(config, num_pages=64)
    layers = [fixture.layer(layer_id, seed=layer_id) for layer_id in (3, 7)]
    oracle = _dense_oracle(fixture)
    lengths, steps = [200, 71], 3
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    xs = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    xs += [torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)]
    for step, x in enumerate(xs):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        for attn in layers:
            fixture.ctx.attn_backend = fixture.backend
            got = attn.forward(x, batch)
            fixture.ctx.attn_backend = oracle
            reference = attn.forward(x, batch)
            torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)

    slab = fixture.pool.cmp_k_cache
    assert not torch.equal(slab(0), slab(1))


def _prefill_under_kv(monkeypatch, config, kv_quant: str, lengths):
    """One prefill of the QSA layer under a given KV store.

    Each call builds its own Fixture on purpose: a Fixture owns the global ctx (pool,
    page table, backend), so two KV stores cannot share one scenario. The weight seed
    (``Fixture.layer``) and the input seed (``_inputs``) are fixed, so the two runs differ
    ONLY in how the K/V rows are stored.
    """
    fixture = Fixture(config, num_pages=128, kv_quant=kv_quant)
    attn = fixture.layer(QSA_LAYER)
    seen = selection_spy(monkeypatch, fixture.backend)
    inputs = _inputs(fixture, lengths)
    x = torch.cat([row[:n] for row, n in zip(inputs, lengths)])
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    batch = fixture.batch(reqs, "prefill")
    out = attn.forward(x, batch)
    # the selection lives in a scratch buffer the next forward overwrites
    return fixture, out.clone(), seen["indices"].clone(), batch.positions.clone()


@requires_cuda
def test_fp8_kv_pool_keeps_selection_and_output(monkeypatch):
    """--kv-cache-dtype fp8 through the real layer: e4m3 codes + per-row scales in, same
    answer out to within quantization error -- and, because block selection scores 16-bit
    compressed index keys that fp8 never touches, the SAME selection bit for bit."""
    config = parsed_config()
    lengths = [2051, 1000, 137]  # every complete block is selected here

    plain, plain_out, plain_idx, _ = _prefill_under_kv(monkeypatch, config, "none", lengths)
    quant, quant_out, quant_idx, positions = _prefill_under_kv(
        monkeypatch, config, "fp8", lengths
    )

    # The tripwire for the field failure: the backend sizes its indexer scratch with
    # pool.dtype, which must stay the COMPUTE dtype even when store_dtype is e4m3. An
    # fp8 q_index compiles into qsa_mqa_paged's dot and dies at graph capture.
    assert quant.backend.dtype is torch.bfloat16
    assert plain.backend.dtype is torch.bfloat16
    assert quant.pool.store_dtype != torch.bfloat16
    assert quant.pool.kv_quant == "fp8" and plain.pool.kv_quant == "none"
    assert quant.pool.k_cache(QSA_LAYER).element_size() == 1
    assert quant.pool.v_cache(QSA_LAYER).element_size() == 1
    assert plain.pool.k_scale(QSA_LAYER) is None and plain.pool.v_scale(QSA_LAYER) is None
    pages, page_size, kv_heads = quant.pool.k_cache(QSA_LAYER).shape[:3]
    assert quant.pool.k_scale(QSA_LAYER).shape == (pages * page_size, kv_heads)
    assert quant.pool.k_scale(QSA_LAYER).dtype is torch.float32

    for pool in (plain.pool, quant.pool):
        assert pool.cmp_k_cache(0).dtype is torch.bfloat16
    assert torch.equal(quant_idx, plain_idx), (
        "quantizing the KV rows changed which blocks the indexer selected -- the index "
        "tier is supposed to be 16-bit in both runs"
    )
    _assert_selection_is_causal_prefix(quant_idx, positions)

    # Looser than the 2e-2 the 16-bit run needs against the same reference: e4m3 carries
    # four significant bits, so ~1e-2 relative per stored element is the floor here.
    torch.testing.assert_close(quant_out.float(), plain_out.float(), rtol=4e-2, atol=4e-2)

