"""Reusable metadata storage for short QSA continuations."""

import torch
import triton

from freetoken.attention.linear import FLAMetadata
from freetoken.attention.qsa_sparse import QSASparseMetadata, _MIN_ACTIVE_PAGES
from freetoken.kernel.triton.mtp_prepare import prepare


def prepare_small_batch(batch, table, backend, buffers):
    req = batch.reqs[0]
    n = req.extend_len
    page = backend.page_size
    width = table[:, ::page].shape[1]
    if backend._block_topk_kernel is not None:
        needed = max(1, (req.device_len + page - 1) // page)
        width = min(width, max(_MIN_ACTIVE_PAGES, 1 << (needed - 1).bit_length()))
    key = n, width
    if key not in buffers:
        buffers[key] = (
            torch.empty(2*n + 6 + width, dtype=torch.int32, device=table.device),
            torch.empty(n, dtype=table.dtype, device=table.device),
            torch.empty(2, dtype=torch.int64, device=table.device),
            torch.empty(1, dtype=torch.bool, device=table.device),
        )
    meta, out, cu, initial = buffers[key]
    slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
    # Fresh pinned storage stays alive through its asynchronous copy; do not overwrite it.
    host = torch.tensor([req.cached_len, req.device_len, req.table_idx, slot, 0, n],
                        dtype=torch.int32, device="cpu", pin_memory=True)
    params = host.to(table.device, non_blocking=True)
    prepare[(1,)](params, table, meta, out, cu, initial, n, width, *table.stride(), page,
                  triton.next_power_of_2(max(width, n, 2)))
    batch.padded_reqs = batch.reqs
    batch.positions, batch.out_loc = meta[:n], out
    batch.fla_metadata = FLAMetadata(cu, meta[2*n + 5:2*n + 6], initial)
    batch.attn_metadata = QSASparseMetadata(
        is_decode=False, last_indices=meta[2*n + 4:2*n + 5],
        qo_indptr_cpu=host[4:6], kv_len_cpu=host[1:2],
        token_to_req=meta[n:2*n], cu_seqlens=meta[2*n:2*n + 2],
        seq_lens=meta[2*n + 2:2*n + 3], ring_slots=meta[2*n + 3:2*n + 4],
        block_table=meta[2*n + 6:].view(1, width))
