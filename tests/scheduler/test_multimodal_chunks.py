from types import SimpleNamespace

import torch

from freetoken.core import SamplingParams
from freetoken.message import UserMsg
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.table import TableManager


def test_media_survives_chunking_without_prefix_sharing():
    table = TableManager(4, torch.zeros(4, 64, dtype=torch.int32))
    cache = CacheManager(64, 1, table.page_table, "radix")
    manager = PrefillManager(cache, table, SimpleNamespace(inflight_tokens=0))
    ids = torch.tensor([4, 9, 9, 9, 9, 5, 6, 7], dtype=torch.int32)
    media = [dict(start=1, types=torch.tensor([0, 1, 2, 3]), patches=torch.zeros(1, 3, 2, 2), n_vit_h=1, n_vit_w=1)]
    manager.add_one_req(UserMsg(1, ids, SamplingParams(max_tokens=1), media=media))
    original_free = len(cache.free_slots)
    forwarded = 0
    finished = None
    while manager.runnable:
        batch = manager.schedule_next_batch(3)
        assert batch is not None
        [req] = batch.reqs
        assert req.media is media
        assert req.cached_len == forwarded
        cache.allocate_paged([req])
        forwarded += req.extend_len
        req.cached_len = req.device_len
        with cache.lazy_free_region():
            cache.cache_req(req, finished=False)
        assert cache.match_req(SimpleNamespace(input_ids=ids, input_len=len(ids), mm_embeds=None, media=media)).cuda_handle.cached_len == 0
        assert cache.match_req(SimpleNamespace(input_ids=ids, input_len=len(ids), mm_embeds=None)).cuda_handle.cached_len == 0
        if not isinstance(req, ChunkedReq):
            finished = req
    assert forwarded == len(ids)
    assert finished.media[0]["types"].tolist() == [0, 1, 2, 3]
    with cache.lazy_free_region():
        cache.cache_req(finished, finished=True)
    assert len(cache.free_slots) == original_free
