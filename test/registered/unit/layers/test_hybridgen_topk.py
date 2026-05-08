import unittest

import torch

from sglang.srt.layers.attention.hybridgen_topk import (
    CPUTopKWorkspace,
    compute_topk_on_cpu,
    gpu_dense_attention_partial_triton,
    merge_gpu_partial_with_topk_triton,
    merged_softmax_attention,
    merged_softmax_attention_per_head_v,
    merged_softmax_attention_per_head_v_triton,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestHybridGenTopKWorkspace(unittest.TestCase):
    def test_workspace_matches_plain_topk_and_reuses_buffers(self):
        torch.manual_seed(0)
        nq, nkv, head_dim, seq_len = 8, 2, 16, 64
        q = torch.randn(nq, head_dim, dtype=torch.float32)
        k_host = torch.randn(seq_len, nkv, head_dim, dtype=torch.float32)

        ref_scores, ref_indices = compute_topk_on_cpu(
            q, k_host, 7, nq, nkv, scaling=0.25
        )

        workspace = CPUTopKWorkspace()
        got_scores, got_indices = compute_topk_on_cpu(
            q, k_host, 7, nq, nkv, scaling=0.25, workspace=workspace
        )
        scores_ptr = workspace.topk_scores.data_ptr()
        indices_ptr = workspace.topk_indices.data_ptr()

        got_scores_2, got_indices_2 = compute_topk_on_cpu(
            q, k_host, 7, nq, nkv, scaling=0.25, workspace=workspace
        )

        self.assertTrue(torch.equal(got_indices, ref_indices))
        self.assertTrue(torch.allclose(got_scores, ref_scores))
        self.assertTrue(torch.equal(got_indices_2, ref_indices))
        self.assertTrue(torch.allclose(got_scores_2, ref_scores))
        self.assertEqual(workspace.topk_scores.data_ptr(), scores_ptr)
        self.assertEqual(workspace.topk_indices.data_ptr(), indices_ptr)


class TestHybridGenPerHeadMergedAttention(unittest.TestCase):
    def test_per_head_v_matches_unique_scatter_representation(self):
        torch.manual_seed(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nq, nkv, head_dim = 8, 2, 16
        host_len, gpu_len, k_eff = 32, 9, 5
        scaling = 0.25

        q = torch.randn(nq, head_dim, dtype=torch.float32, device=device)
        k_gpu = torch.randn(gpu_len, nkv, head_dim, dtype=torch.float32, device=device)
        v_gpu = torch.randn(gpu_len, nkv, head_dim, dtype=torch.float32, device=device)
        v_host = torch.randn(host_len, nkv, head_dim, dtype=torch.float32)
        host_scores = torch.randn(nq, host_len, dtype=torch.float32)
        topk_scores, topk_indices = host_scores.topk(k_eff, dim=1, sorted=False)

        unique_local, inverse_flat = torch.unique(
            topk_indices.flatten(), return_inverse=True
        )
        scores_remapped = torch.full(
            (nq, unique_local.shape[0]), float("-inf"), dtype=topk_scores.dtype
        )
        scores_remapped.scatter_(1, inverse_flat.view(nq, -1), topk_scores)
        v_unique = v_host[unique_local]

        old_out = merged_softmax_attention(
            q,
            k_gpu,
            v_gpu,
            scores_remapped.to(device),
            v_unique.to(device),
            nq,
            nkv,
            scaling,
        )

        kv_heads = (torch.arange(nq, dtype=torch.long) // (nq // nkv)).repeat_interleave(
            k_eff
        )
        v_per_head = v_host[
            topk_indices.reshape(-1), kv_heads
        ].view(nq, k_eff, head_dim)
        new_out = merged_softmax_attention_per_head_v(
            q,
            k_gpu,
            v_gpu,
            topk_scores.to(device),
            v_per_head.to(device),
            nq,
            nkv,
            scaling,
        )

        self.assertTrue(torch.allclose(new_out, old_out, atol=1e-6, rtol=1e-6))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for Triton")
    def test_triton_fused_matches_pytorch_per_head_v(self):
        torch.manual_seed(2)
        device = torch.device("cuda")
        nq, nkv, head_dim = 8, 2, 64
        gpu_len, k_eff = 37, 11
        scaling = head_dim**-0.5

        q = torch.randn(nq, head_dim, dtype=torch.bfloat16, device=device)
        k_gpu = torch.randn(gpu_len, nkv, head_dim, dtype=torch.bfloat16, device=device)
        v_gpu = torch.randn(gpu_len, nkv, head_dim, dtype=torch.bfloat16, device=device)
        topk_scores = torch.randn(nq, k_eff, dtype=torch.bfloat16, device=device)
        v_topk = torch.randn(nq, k_eff, head_dim, dtype=torch.bfloat16, device=device)

        ref = merged_softmax_attention_per_head_v(
            q, k_gpu, v_gpu, topk_scores, v_topk, nq, nkv, scaling
        )
        got = merged_softmax_attention_per_head_v_triton(
            q, k_gpu, v_gpu, topk_scores, v_topk, nq, nkv, scaling
        )

        self.assertTrue(torch.allclose(got, ref, atol=2e-2, rtol=2e-2))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for Triton")
    def test_triton_partial_merge_matches_fused(self):
        torch.manual_seed(3)
        device = torch.device("cuda")
        nq, nkv, head_dim = 8, 2, 64
        gpu_len, k_eff = 65, 13
        scaling = head_dim**-0.5

        q = torch.randn(nq, head_dim, dtype=torch.bfloat16, device=device)
        k_gpu = torch.randn(gpu_len, nkv, head_dim, dtype=torch.bfloat16, device=device)
        v_gpu = torch.randn(gpu_len, nkv, head_dim, dtype=torch.bfloat16, device=device)
        topk_scores = torch.randn(nq, k_eff, dtype=torch.bfloat16, device=device)
        v_topk = torch.randn(nq, k_eff, head_dim, dtype=torch.bfloat16, device=device)

        ref = merged_softmax_attention_per_head_v_triton(
            q, k_gpu, v_gpu, topk_scores, v_topk, nq, nkv, scaling
        )
        gpu_partial = gpu_dense_attention_partial_triton(
            q, k_gpu, v_gpu, nq, nkv, scaling
        )
        got = merge_gpu_partial_with_topk_triton(
            gpu_partial[0],
            gpu_partial[1],
            gpu_partial[2],
            topk_scores,
            v_topk,
        )

        self.assertTrue(torch.allclose(got, ref, atol=2e-2, rtol=2e-2))


if __name__ == "__main__":
    unittest.main()
