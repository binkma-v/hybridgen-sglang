import unittest

from sglang.srt.layers.attention.hybridgen_feedback import (
    FeedbackController,
    FeedbackPolicy,
    adapt,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestHybridGenFeedbackCap(unittest.TestCase):
    def test_bottleneck_shrinks_ratio_and_cap_together(self):
        ratio, cap = adapt(
            topk_ratio=0.05,
            cpu_k_cap=2048,
            cpu_cache_len=8192,
            host_lat=2.0,
            device_lat=1.0,
            policy=FeedbackPolicy(),
        )

        self.assertAlmostEqual(ratio, 0.04)
        self.assertEqual(cap, 1024)

    def test_bottleneck_initializes_cap_when_unlimited(self):
        ratio, cap = adapt(
            topk_ratio=0.01,
            cpu_k_cap=0,
            cpu_cache_len=8192,
            host_lat=2.0,
            device_lat=1.0,
            policy=FeedbackPolicy(),
        )

        self.assertEqual(ratio, 0.01)
        self.assertEqual(cap, 4096)

    def test_hidden_host_relaxes_cap_gradually(self):
        ratio, cap = adapt(
            topk_ratio=0.02,
            cpu_k_cap=128,
            cpu_cache_len=8192,
            host_lat=1.0,
            device_lat=2.0,
            policy=FeedbackPolicy(),
        )

        self.assertAlmostEqual(ratio, 0.024)
        self.assertEqual(cap, 153)

    def test_controller_uses_effective_cpu_len_for_latency(self):
        class FakeEstimator:
            def __init__(self):
                self.host_seq_lens = []
                self.device_cpu_lens = []

            def host_latency(self, seq_len, topk):
                self.host_seq_lens.append(seq_len)
                return 2.0

            def device_latency(self, gpu_seq_len, cpu_seq_len, topk):
                self.device_cpu_lens.append(cpu_seq_len)
                return 1.0

        estimator = FakeEstimator()
        controller = FeedbackController(
            estimator=estimator,
            policy=FeedbackPolicy(),
            interval=1,
        )

        ratio, cap = controller.maybe_step(
            topk_ratio=0.05,
            cpu_k_cap=2048,
            cpu_cache_len=8192,
            gpu_cache_len=819,
            effective_cpu_k_len=2048,
        )

        self.assertEqual(estimator.host_seq_lens, [2048])
        self.assertEqual(estimator.device_cpu_lens, [2048])
        self.assertAlmostEqual(ratio, 0.04)
        self.assertEqual(cap, 1024)


if __name__ == "__main__":
    unittest.main()
