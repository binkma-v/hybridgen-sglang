"""HybridGen CPU-side top-k selection and merged-softmax attention.

Ported from hybridgen/llama/hybridgen/pytorch_backend.py:
- compute_score_from_q (line 417)
- mha_gen_hybridgen merged softmax body (line 525)

The functions here are pure PyTorch and device-agnostic; the caller decides
which device each tensor lives on. `compute_topk_on_cpu` expects CPU tensors;
`merged_softmax_attention` expects GPU tensors.
"""

from __future__ import annotations

from typing import Optional, Tuple

import triton
import triton.language as tl
import torch


class CPUTopKWorkspace:
    """Reusable CPU scratch tensors for HybridGen top-k.

    The decode path calls top-k once per layer, so avoiding fresh score/top-k
    allocations removes a noticeable amount of Python allocator overhead.
    """

    def __init__(self) -> None:
        self.q_grouped: Optional[torch.Tensor] = None
        self.scores_grouped: Optional[torch.Tensor] = None
        self.topk_scores: Optional[torch.Tensor] = None
        self.topk_indices: Optional[torch.Tensor] = None

    @staticmethod
    def _ensure(
        tensor: Optional[torch.Tensor],
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            tensor is None
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            return torch.empty(shape, dtype=dtype, device=device)
        return tensor


@triton.jit(
    do_not_specialize=[
        "s_gpu",
        "k_eff",
    ]
)
def _merged_softmax_attention_per_head_v_kernel(
    q_ptr,
    k_gpu_ptr,
    v_gpu_ptr,
    cpu_scores_ptr,
    v_topk_ptr,
    out_ptr,
    s_gpu,
    k_eff,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    scaling: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_head = tl.program_id(0)
    q_per_kv = num_q_heads // num_kv_heads
    kv_head = q_head // q_per_kv

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    total_len = k_eff + s_gpu

    q_vals = tl.load(
        q_ptr + q_head * head_dim + offs_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    gpu_n = offs_n - k_eff
    gpu_mask = (offs_n >= k_eff) & (gpu_n < s_gpu)
    k_vals = tl.load(
        k_gpu_ptr
        + gpu_n[:, None] * (num_kv_heads * head_dim)
        + kv_head * head_dim
        + offs_d[None, :],
        mask=gpu_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    gpu_logits = tl.sum(k_vals * q_vals[None, :], axis=1) * scaling

    topk_mask = offs_n < k_eff
    topk_logits = tl.load(
        cpu_scores_ptr + q_head * k_eff + offs_n,
        mask=topk_mask,
        other=-float("inf"),
    ).to(tl.float32)
    logits = tl.where(topk_mask, topk_logits, gpu_logits)
    logits = tl.where(offs_n < total_len, logits, -float("inf"))

    max_logit = tl.max(logits, axis=0)
    weights = tl.exp(logits - max_logit)
    denom = tl.sum(weights, axis=0)

    topv_vals = tl.load(
        v_topk_ptr
        + (q_head * k_eff + offs_n[:, None]) * head_dim
        + offs_d[None, :],
        mask=topk_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    gpuv_vals = tl.load(
        v_gpu_ptr
        + gpu_n[:, None] * (num_kv_heads * head_dim)
        + kv_head * head_dim
        + offs_d[None, :],
        mask=gpu_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    vals = tl.where(topk_mask[:, None], topv_vals, gpuv_vals)
    acc = tl.sum(weights[:, None] * vals, axis=0) / denom

    tl.store(
        out_ptr + q_head * head_dim + offs_d,
        acc,
        mask=d_mask,
    )


def _maybe_repeat_kv(
    k: torch.Tensor, v: torch.Tensor, num_q_heads: int, num_kv_heads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand K/V along the head dim to match Q heads (GQA).

    K, V shape: (S, num_kv_heads, head_dim). Returns expanded (S, num_q_heads, head_dim).
    """
    if num_q_heads == num_kv_heads:
        return k, v
    repeat = num_q_heads // num_kv_heads
    assert num_q_heads == repeat * num_kv_heads, (
        f"num_q_heads ({num_q_heads}) must be a multiple of "
        f"num_kv_heads ({num_kv_heads})"
    )
    return (
        k.repeat_interleave(repeat, dim=1),
        v.repeat_interleave(repeat, dim=1),
    )


def compute_topk_on_cpu(
    q: torch.Tensor,
    k_host: torch.Tensor,
    top_k: int,
    num_q_heads: int,
    num_kv_heads: int,
    scaling: float,
    workspace: Optional[CPUTopKWorkspace] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute QK^T scores against host-resident K and pick top-k.

    Args:
        q: query for one sequence, shape (num_q_heads, head_dim) on CPU.
        k_host: host-resident K segment, shape (S_host, num_kv_heads, head_dim) on CPU.
        top_k: requested top-k count; clipped to S_host if larger.
        num_q_heads, num_kv_heads: head counts (GQA aware).
        scaling: 1/sqrt(head_dim) factor applied to Q before QK^T.

    Returns:
        topk_scores: (num_q_heads, k_eff) on CPU, where k_eff = min(top_k, S_host).
        topk_indices: (num_q_heads, k_eff) on CPU, indices into k_host's S dim.
    """
    s_host, n_kv, head_dim = k_host.shape
    assert n_kv == num_kv_heads, (
        f"k_host has {n_kv} kv-heads, expected {num_kv_heads}"
    )
    k_eff = min(top_k, s_host)
    if k_eff == 0:
        return (
            q.new_zeros(num_q_heads, 0),
            torch.zeros(num_q_heads, 0, dtype=torch.long, device=q.device),
        )

    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be a multiple of "
        f"num_kv_heads ({num_kv_heads})"
    )
    g = num_q_heads // num_kv_heads

    # GQA-aware QK^T without expanding K. Group q heads by kv head:
    # q (nq, d) -> (nkv, g, d), then bmm against
    # k.permute(1, 2, 0): (nkv, d, S) -> (nkv, g, S) -> (nq, S).
    q_grouped_view = q.view(num_kv_heads, g, head_dim)
    k_b = k_host.permute(1, 2, 0)
    if workspace is None:
        q_grouped = (q_grouped_view * scaling).contiguous()
        scores_grouped = torch.bmm(q_grouped, k_b)
        topk_scores, topk_indices = scores_grouped.reshape(
            num_q_heads, s_host
        ).topk(k_eff, dim=1, sorted=False)
        return topk_scores, topk_indices

    workspace.q_grouped = CPUTopKWorkspace._ensure(
        workspace.q_grouped,
        (num_kv_heads, g, head_dim),
        q.dtype,
        q.device,
    )
    workspace.scores_grouped = CPUTopKWorkspace._ensure(
        workspace.scores_grouped,
        (num_kv_heads, g, s_host),
        q.dtype,
        q.device,
    )
    workspace.topk_scores = CPUTopKWorkspace._ensure(
        workspace.topk_scores,
        (num_q_heads, k_eff),
        q.dtype,
        q.device,
    )
    workspace.topk_indices = CPUTopKWorkspace._ensure(
        workspace.topk_indices,
        (num_q_heads, k_eff),
        torch.long,
        q.device,
    )

    torch.mul(q_grouped_view, scaling, out=workspace.q_grouped)
    torch.bmm(workspace.q_grouped, k_b, out=workspace.scores_grouped)
    torch.topk(
        workspace.scores_grouped.reshape(num_q_heads, s_host),
        k_eff,
        dim=1,
        sorted=False,
        out=(workspace.topk_scores, workspace.topk_indices),
    )
    topk_scores, topk_indices = workspace.topk_scores, workspace.topk_indices
    return topk_scores, topk_indices


def merged_softmax_attention(
    q: torch.Tensor,
    k_gpu: torch.Tensor,
    v_gpu: torch.Tensor,
    cpu_topk_scores: Optional[torch.Tensor],
    v_topk: Optional[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    scaling: float,
) -> torch.Tensor:
    """Merged softmax over (CPU top-k logits, GPU dense logits) and matmul.

    Mirrors mha_gen_hybridgen lines 564-602 of the standalone hybridgen.

    Args:
        q: (num_q_heads, head_dim) — single decode token's query, on GPU.
        k_gpu: (S_gpu, num_kv_heads, head_dim) — recent K cache on GPU.
        v_gpu: (S_gpu, num_kv_heads, head_dim) — recent V cache on GPU.
        cpu_topk_scores: optional (num_q_heads, k_eff) — pre-scaled scores from
            compute_topk_on_cpu, already moved to GPU.
        v_topk: optional (k_eff, num_kv_heads, head_dim) — V vectors gathered from
            host at the top-k indices, already moved to GPU.
        num_q_heads, num_kv_heads, scaling: as in compute_topk_on_cpu.

    Returns:
        out: (num_q_heads, head_dim) on GPU — the merged attention output.
    """
    s_gpu, n_kv, head_dim = k_gpu.shape
    assert n_kv == num_kv_heads
    assert v_gpu.shape == k_gpu.shape

    # Expand recent K/V to Q heads.
    k_g, v_g = _maybe_repeat_kv(k_gpu, v_gpu, num_q_heads, num_kv_heads)

    # GPU-side QK^T -> (num_q_heads, S_gpu)
    q_b = q.unsqueeze(1) * scaling          # (num_q_heads, 1, head_dim)
    k_b = k_g.permute(1, 2, 0)               # (num_q_heads, head_dim, S_gpu)
    s_gpu_logits = torch.bmm(q_b, k_b).squeeze(1)  # (num_q_heads, S_gpu)

    if cpu_topk_scores is not None and cpu_topk_scores.numel() > 0:
        # cpu_topk_scores layout: (num_q_heads, k_eff). Both segments share
        # the same scaling because compute_topk_on_cpu already applied it.
        all_logits = torch.cat([cpu_topk_scores, s_gpu_logits], dim=1)
    else:
        all_logits = s_gpu_logits

    weights = torch.softmax(all_logits, dim=1)  # (num_q_heads, k_eff + S_gpu)

    # Combine V along the seq axis. v_topk first (matches the order of
    # cpu_topk_scores in `all_logits`), then v_g.
    if v_topk is not None and v_topk.numel() > 0:
        v_t, _ = _maybe_repeat_kv(v_topk, v_topk, num_q_heads, num_kv_heads)
        all_v = torch.cat([v_t, v_g], dim=0)  # (k_eff + S_gpu, num_q_heads, head_dim)
    else:
        all_v = v_g

    # all_v: (S_total, num_q_heads, head_dim) -> (num_q_heads, S_total, head_dim)
    all_v_t = all_v.permute(1, 0, 2)
    weights = weights.to(all_v_t.dtype)
    # bmm: (num_q_heads, 1, S_total) @ (num_q_heads, S_total, head_dim)
    out = torch.bmm(weights.unsqueeze(1), all_v_t).squeeze(1)
    return out  # (num_q_heads, head_dim)


def merged_softmax_attention_per_head_v(
    q: torch.Tensor,
    k_gpu: torch.Tensor,
    v_gpu: torch.Tensor,
    cpu_topk_scores: Optional[torch.Tensor],
    v_topk_per_head: Optional[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    scaling: float,
) -> torch.Tensor:
    """Merged softmax where CPU top-k V is already laid out per Q head.

    This is numerically equivalent to the shared-unique-column representation
    used by :func:`merged_softmax_attention`, but it avoids torch.unique,
    inverse-index construction, and scatter remapping in the decode path.

    Args:
        q: (num_q_heads, head_dim) on GPU.
        k_gpu/v_gpu: (S_gpu, num_kv_heads, head_dim) on GPU.
        cpu_topk_scores: optional (num_q_heads, k_eff) on GPU.
        v_topk_per_head: optional (num_q_heads, k_eff, head_dim) on GPU.

    Returns:
        out: (num_q_heads, head_dim) on GPU.
    """
    s_gpu, n_kv, _ = k_gpu.shape
    assert n_kv == num_kv_heads
    assert v_gpu.shape == k_gpu.shape

    k_g, v_g = _maybe_repeat_kv(k_gpu, v_gpu, num_q_heads, num_kv_heads)

    q_b = q.unsqueeze(1) * scaling
    k_b = k_g.permute(1, 2, 0)
    s_gpu_logits = torch.bmm(q_b, k_b).squeeze(1)

    if cpu_topk_scores is not None and cpu_topk_scores.numel() > 0:
        assert v_topk_per_head is not None
        all_logits = torch.cat([cpu_topk_scores, s_gpu_logits], dim=1)
        all_v_t = torch.cat([v_topk_per_head, v_g.permute(1, 0, 2)], dim=1)
    else:
        all_logits = s_gpu_logits
        all_v_t = v_g.permute(1, 0, 2)

    weights = torch.softmax(all_logits, dim=1).to(all_v_t.dtype)
    out = torch.bmm(weights.unsqueeze(1), all_v_t).squeeze(1)
    return out


def merged_softmax_attention_per_head_v_triton(
    q: torch.Tensor,
    k_gpu: torch.Tensor,
    v_gpu: torch.Tensor,
    cpu_topk_scores: Optional[torch.Tensor],
    v_topk_per_head: Optional[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    scaling: float,
) -> torch.Tensor:
    """Triton fused variant of :func:`merged_softmax_attention_per_head_v`.

    The kernel fuses GPU-segment QK, merged softmax over CPU top-k + GPU dense
    scores, and the final weighted V sum for single-token decode.
    """
    if (
        cpu_topk_scores is None
        or v_topk_per_head is None
        or not q.is_cuda
        or not k_gpu.is_cuda
        or not v_gpu.is_cuda
        or not cpu_topk_scores.is_cuda
        or not v_topk_per_head.is_cuda
    ):
        return merged_softmax_attention_per_head_v(
            q,
            k_gpu,
            v_gpu,
            cpu_topk_scores,
            v_topk_per_head,
            num_q_heads,
            num_kv_heads,
            scaling,
        )

    s_gpu, n_kv, head_dim = k_gpu.shape
    k_eff = int(cpu_topk_scores.shape[1])
    total_len = s_gpu + k_eff
    if (
        n_kv != num_kv_heads
        or q.shape != (num_q_heads, head_dim)
        or v_gpu.shape != k_gpu.shape
        or v_topk_per_head.shape != (num_q_heads, k_eff, head_dim)
        or total_len <= 0
        or total_len > 4096
        or head_dim > 256
        or num_q_heads % num_kv_heads != 0
    ):
        return merged_softmax_attention_per_head_v(
            q,
            k_gpu,
            v_gpu,
            cpu_topk_scores,
            v_topk_per_head,
            num_q_heads,
            num_kv_heads,
            scaling,
        )

    q = q.contiguous()
    k_gpu = k_gpu.contiguous()
    v_gpu = v_gpu.contiguous()
    cpu_topk_scores = cpu_topk_scores.contiguous()
    v_topk_per_head = v_topk_per_head.contiguous()

    # Use one compiled variant for the common decode range. Triton JIT cost is
    # high enough that separate 256/512/1024 variants hurt first-request
    # latency more than the smaller block helps steady-state throughput.
    block_n = 1024 if total_len <= 1024 else triton.next_power_of_2(total_len)
    block_d = triton.next_power_of_2(head_dim)
    num_warps = 8 if block_n >= 1024 else 4
    out = torch.empty_like(q)
    _merged_softmax_attention_per_head_v_kernel[(num_q_heads,)](
        q,
        k_gpu,
        v_gpu,
        cpu_topk_scores,
        v_topk_per_head,
        out,
        s_gpu,
        k_eff,
        num_q_heads,
        num_kv_heads,
        head_dim,
        float(scaling),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out
