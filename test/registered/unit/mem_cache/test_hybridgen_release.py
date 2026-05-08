import unittest

import torch

from sglang.srt.layers.attention.hybrid_kvcache_backend import (
    HybridKVCacheAttnBackend,
)
from sglang.srt.mem_cache.hybridgen_release import (
    RELEASED_DEVICE_KV_SLOT,
    count_released_device_indices,
    has_released_device_indices,
    mark_device_indices_released,
    valid_device_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _Allocator:
    def __init__(self):
        self.freed = []

    def free(self, indices: torch.Tensor):
        self.freed.extend(indices.detach().cpu().tolist())


class TestHybridGenReleaseHelpers(unittest.TestCase):
    def test_released_slot_sentinel_filters_only_zero(self):
        indices = torch.tensor([0, 11, 0, 12], dtype=torch.int32)

        self.assertEqual(RELEASED_DEVICE_KV_SLOT, 0)
        self.assertTrue(has_released_device_indices(indices))
        self.assertEqual(count_released_device_indices(indices), 2)
        self.assertEqual(valid_device_indices(indices).tolist(), [11, 12])

    def test_mark_device_indices_released_marks_contiguous_span(self):
        req_to_token = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.int32)

        mark_device_indices_released(
            req_to_token, req_pool_idx=0, start_pos=1, num_tokens=3
        )

        self.assertEqual(req_to_token.tolist(), [[11, 0, 0, 0, 15]])


class TestHybridKVCacheShadowRelease(unittest.TestCase):
    def _backend(self):
        backend = object.__new__(HybridKVCacheAttnBackend)
        backend._device_allocator = _Allocator()
        return backend

    def test_release_device_slots_filters_sentinel_and_is_idempotent(self):
        backend = self._backend()
        req_to_token = torch.tensor([[11, 12, 0, 14, 21]], dtype=torch.int32)

        released = backend._release_device_slots(
            req_to_token,
            req_pool_idx=0,
            start_pos=0,
            device_indices=req_to_token[0, :4],
        )

        self.assertEqual(released, 3)
        self.assertEqual(backend._device_allocator.freed, [11, 12, 14])
        self.assertEqual(req_to_token.tolist(), [[0, 0, 0, 0, 21]])

        released = backend._release_device_slots(
            req_to_token,
            req_pool_idx=0,
            start_pos=0,
            device_indices=req_to_token[0, :4],
        )

        self.assertEqual(released, 0)
        self.assertEqual(backend._device_allocator.freed, [11, 12, 14])
        self.assertEqual(req_to_token.tolist(), [[0, 0, 0, 0, 21]])

    def test_release_existing_host_shadows_requires_decode_and_unprotected_prefix(self):
        backend = self._backend()
        req_to_token = torch.tensor([[31, 32, 33, 34]], dtype=torch.int32)

        released = backend._release_existing_host_shadows(
            req_to_token,
            req_pool_idx=0,
            n_host=3,
            cache_protected_len=0,
            is_prefill=True,
        )
        self.assertEqual(released, 0)
        self.assertEqual(backend._device_allocator.freed, [])
        self.assertEqual(req_to_token.tolist(), [[31, 32, 33, 34]])

        released = backend._release_existing_host_shadows(
            req_to_token,
            req_pool_idx=0,
            n_host=3,
            cache_protected_len=1,
            is_prefill=False,
        )
        self.assertEqual(released, 0)
        self.assertEqual(backend._device_allocator.freed, [])
        self.assertEqual(req_to_token.tolist(), [[31, 32, 33, 34]])

        released = backend._release_existing_host_shadows(
            req_to_token,
            req_pool_idx=0,
            n_host=3,
            cache_protected_len=0,
            is_prefill=False,
        )
        self.assertEqual(released, 3)
        self.assertEqual(backend._device_allocator.freed, [31, 32, 33])
        self.assertEqual(req_to_token.tolist(), [[0, 0, 0, 34]])

        released = backend._release_existing_host_shadows(
            req_to_token,
            req_pool_idx=0,
            n_host=3,
            cache_protected_len=0,
            is_prefill=False,
        )
        self.assertEqual(released, 0)
        self.assertEqual(backend._device_allocator.freed, [31, 32, 33])


if __name__ == "__main__":
    unittest.main()
