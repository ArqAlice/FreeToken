"""Commit a verified recurrent prefix without reading its length on the host."""

import triton
import triton.language as tl


@triton.jit
def _recurrent(D, H, A, slot, WIDTH: tl.constexpr, VERIFIED: tl.constexpr,
               DS0: tl.constexpr, DS1: tl.constexpr, HS0: tl.constexpr,
               HS2: tl.constexpr, BLOCK: tl.constexpr):
    accepted = tl.load(A)
    if accepted + 1 < VERIFIED:
        layer = tl.program_id(1)
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(H + layer * HS0 + accepted * HS2 + offset, offset < WIDTH, 0)
        tl.store(D + layer * DS0 + slot * DS1 + offset, value, offset < WIDTH)


@triton.jit
def _window(D, OLD, X, A, slot, WIDTH: tl.constexpr, WINDOW: tl.constexpr,
            VERIFIED: tl.constexpr, DS0: tl.constexpr, DS1: tl.constexpr,
            OS0: tl.constexpr, XS0: tl.constexpr, XS1: tl.constexpr,
            XS2: tl.constexpr, BLOCK: tl.constexpr):
    count = tl.load(A) + 1
    if count < VERIFIED:
        layer = tl.program_id(1)
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offset < WIDTH * WINDOW
        channel, position = offset // WINDOW, offset % WINDOW + count
        previous = tl.load(OLD + layer * OS0 + channel * WINDOW + position,
                           mask & (position < WINDOW), 0)
        current = tl.load(X + layer * XS0 + (position - WINDOW) * XS1 + channel * XS2,
                          mask & (position >= WINDOW), 0)
        tl.store(D + layer * DS0 + slot * DS1 + offset,
                 tl.where(position < WINDOW, previous, current), mask)


@triton.jit
def _ring(R, OLD, A, start, SIZE: tl.constexpr, PERIOD: tl.constexpr,
          ROW: tl.constexpr, VERIFIED: tl.constexpr, BLOCK: tl.constexpr):
    count = tl.load(A) + 1
    if count < VERIFIED:
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        position = offset // ROW % PERIOD
        # Accepted writes stay in place; only restore the uncommitted ring entries.
        restore = (position - start % PERIOD + PERIOD) % PERIOD >= count
        mask = (offset < SIZE) & restore
        tl.store(R + offset, tl.load(OLD + offset, mask, 0), mask)


def commit_prefix_cuda(pool, saved, batch, ring, accepted, start):
    slot, states, old_ring = saved
    verified = batch.input_ids.numel()
    rec = pool.recurrent_states
    history = batch.mtp_recurrent_history
    assert rec.is_contiguous() and history.is_contiguous()
    width = rec[0, 0].numel()
    _recurrent[(triton.cdiv(width, 1024), rec.shape[0])](
        rec, history, accepted, slot, width, verified, *rec.stride()[:2],
        history.stride(0), history.stride(2), 1024)

    def window(destination, old, inputs, strides):
        assert destination.is_contiguous() and old.is_contiguous()
        size = old[0].numel()
        _window[(triton.cdiv(size, 256), old.shape[0])](
            destination, old, inputs, accepted, slot, size // old.shape[-1], old.shape[-1],
            verified, *destination.stride()[:2], old.stride(0), *strides, 256)

    window(pool.conv_states, states[0][1], batch.mtp_conv_inputs, batch.mtp_conv_inputs.stride())
    previous = {id(tensor): value for tensor, value in states}
    for name, tensor in pool.slot_states.items():
        old = previous[id(tensor)]
        if name == "ple_conv":
            for lid, inputs in batch.mtp_ple_inputs.items():
                layer = pool._state_layer_index[name][lid]
                window(tensor[layer:layer + 1], old[layer:layer + 1], inputs,
                       (0, *inputs.stride()))
        elif name == "ple_ngram_ctx":
            window(tensor, old, batch.input_ids, (0, batch.input_ids.stride(0), 0))
        else:
            raise RuntimeError(f"MTP prefix restoration does not support slot state {name!r}")
    assert ring.is_contiguous() and old_ring.is_contiguous()
    _ring[(triton.cdiv(ring.numel(), 256),)](
        ring, old_ring, accepted, start, ring.numel(), ring.shape[1], ring.stride(1), verified, 256)
