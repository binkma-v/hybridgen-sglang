import unittest

from sglang.srt.layers.attention.hybrid_kvcache_backend import (
    HybridKVCacheAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestHybridGenResidencyCap(unittest.TestCase):
    def test_min_recent_protects_short_prompt_generation(self):
        self.assertEqual(
            HybridKVCacheAttnBackend._gpu_residency_cap(
                prompt_len=128,
                gpu_cache_factor=0.1,
                min_gpu_recent_tokens=512,
            ),
            512,
        )

    def test_factor_still_wins_for_long_prompt(self):
        self.assertEqual(
            HybridKVCacheAttnBackend._gpu_residency_cap(
                prompt_len=8192,
                gpu_cache_factor=0.1,
                min_gpu_recent_tokens=512,
            ),
            819,
        )

    def test_zero_min_recent_preserves_old_factor_semantics(self):
        self.assertEqual(
            HybridKVCacheAttnBackend._gpu_residency_cap(
                prompt_len=128,
                gpu_cache_factor=0.1,
                min_gpu_recent_tokens=0,
            ),
            12,
        )


if __name__ == "__main__":
    unittest.main()
