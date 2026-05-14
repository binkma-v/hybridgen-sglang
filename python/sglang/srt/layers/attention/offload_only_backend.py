"""Offload-only baseline attention backend.

This backend is the FlexGen-style baseline alongside `HybridKVCacheAttnBackend`:
it shares the same eviction / host KV-pool machinery (older KV is moved to host
memory once the GPU residency cap is exceeded), but performs **no CPU
computation**. Instead, for every decode step it pulls the entire host segment
of each request back to GPU over PCIe and runs a dense SDPA on the
concatenation of `[host_kv, gpu_kv]`. This isolates the "offload + full
load-back" cost so we can compare it against HybridGen's CPU top-k path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.hybrid_kvcache_backend import (
    HybridKVCacheAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class OffloadOnlyAttnBackend(HybridKVCacheAttnBackend):
    """Offload-only baseline: same eviction + host pool as HybridGen, but every
    decode step loads the **full** host KV segment back to GPU and runs dense
    attention there. No CPU-side QK^T, no top-k selection, no merged softmax."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # The feedback controller tunes topk_ratio / cpu_k_cap, neither of which
        # this backend uses; disable it unconditionally to keep the baseline
        # cost model clean (no analytical-estimator overhead either).
        self._feedback = None

    # ------------------------------------------------------------------
    # Decode override
    # ------------------------------------------------------------------

    def _run_hybrid_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
    ) -> torch.Tensor:
        """Per-request loop. For each request, load any host-resident KV for
        this layer back to GPU and run a single dense SDPA over the union of
        host and GPU segments."""

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        if layer.qk_head_dim != layer.v_head_dim:
            output = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            output = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        out_view = output.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens

        scaling = layer.scaling
        nq = layer.tp_q_head_num
        nkv = layer.tp_k_head_num

        device = q.device
        dtype = q.dtype
        use_gqa = nq != nkv

        for seq_idx in range(seq_lens.shape[0]):
            req_pool_idx = int(req_pool_indices[seq_idx].item())
            seq_len_kv = int(seq_lens[seq_idx].item())
            host_indices = self._host_segments.get(req_pool_idx)
            n_host = 0 if host_indices is None else int(host_indices.shape[0])

            per_req_q = q_view[seq_idx]  # (nq, qk_head_dim)
            # GPU-resident segment is positions [n_host, seq_len_kv);
            # positions [0, n_host) live on host (their GPU shadows may have
            # been released during eviction).
            per_req_tokens = req_to_token[req_pool_idx, n_host:seq_len_kv]
            per_req_k_gpu = k_cache[per_req_tokens]  # (S_gpu, nkv, qk_head_dim)
            per_req_v_gpu = v_cache[per_req_tokens]  # (S_gpu, nkv, v_head_dim)

            if n_host == 0:
                k_dense = per_req_k_gpu
                v_dense = per_req_v_gpu
            else:
                # Load the full host segment back to GPU. This is the
                # baseline's defining cost: PCIe traffic is O(n_host * nkv *
                # head_dim) per request per layer per decode step.
                k_host_seg = self._host_pool.k_buffer[
                    layer.layer_id, host_indices
                ]  # CPU (n_host, nkv, qk_head_dim)
                v_host_seg = self._host_pool.v_buffer[
                    layer.layer_id, host_indices
                ]  # CPU (n_host, nkv, v_head_dim)
                # Move to GPU in the layer's compute dtype. non_blocking only
                # has effect when the source is pinned; host pool buffers are
                # pinned, so this can overlap with kernel launches above.
                k_host_gpu = k_host_seg.to(
                    device=device, dtype=per_req_q.dtype, non_blocking=True
                )
                v_host_gpu = v_host_seg.to(
                    device=device, dtype=per_req_q.dtype, non_blocking=True
                )
                # Bring GPU segment into the same dtype as host segment.
                if per_req_q.dtype != per_req_k_gpu.dtype:
                    per_req_k_gpu = per_req_k_gpu.to(per_req_q.dtype)
                    per_req_v_gpu = per_req_v_gpu.to(per_req_q.dtype)
                # Order: host segment is older (positions [0, n_host)), GPU
                # segment is newer (positions [n_host, seq_len)). Concat in
                # chronological order for clarity (the actual attention is
                # permutation-invariant for non-causal SDPA).
                k_dense = torch.cat([k_host_gpu, per_req_k_gpu], dim=0)
                v_dense = torch.cat([v_host_gpu, per_req_v_gpu], dim=0)

            # SDPA expects (..., L, E). Lay K/V as (nkv, S, head_dim).
            k_dense = k_dense.movedim(0, 1)
            v_dense = v_dense.movedim(0, 1)
            if per_req_q.dtype != k_dense.dtype:
                k_dense = k_dense.to(per_req_q.dtype)
                v_dense = v_dense.to(per_req_q.dtype)
            out_seq = scaled_dot_product_attention(
                per_req_q.unsqueeze(0).unsqueeze(2),  # (1, nq, 1, qk_head_dim)
                k_dense.unsqueeze(0),                 # (1, nkv, S, qk_head_dim)
                v_dense.unsqueeze(0),                 # (1, nkv, S, v_head_dim)
                enable_gqa=use_gqa,
                scale=scaling,
                is_causal=False,
            ).squeeze(0).squeeze(1)                   # (nq, v_head_dim)
            out_view[seq_idx] = out_seq.to(dtype)

        return output
