"""Helpers for HybridGen-released device KV slots.

SGLang's token-to-KV allocators reserve slot 0 as a dummy/padding slot and
allocate real KV slots from 1. HybridGen uses that reserved value as an
in-band sentinel in req_to_token rows after a request-owned GPU KV slot has
been backed up to host and physically returned to the allocator.
"""

from __future__ import annotations

import torch

RELEASED_DEVICE_KV_SLOT = 0


def valid_device_indices(indices: torch.Tensor) -> torch.Tensor:
    """Return only still-resident GPU KV slots from a req_to_token slice."""
    if indices.numel() == 0:
        return indices
    return indices[indices != RELEASED_DEVICE_KV_SLOT]


def has_released_device_indices(indices: torch.Tensor) -> bool:
    """Whether a req_to_token slice contains HybridGen released-slot sentinels."""
    return (
        bool((indices == RELEASED_DEVICE_KV_SLOT).any().item())
        if indices.numel() > 0
        else False
    )


def count_released_device_indices(indices: torch.Tensor) -> int:
    """Count HybridGen released-slot sentinels in a req_to_token slice."""
    return (
        int((indices == RELEASED_DEVICE_KV_SLOT).sum().item())
        if indices.numel() > 0
        else 0
    )


def mark_device_indices_released(
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    start_pos: int,
    num_tokens: int,
) -> None:
    """Mark a contiguous logical token span as host-backed / GPU-released."""
    if num_tokens <= 0:
        return
    req_to_token[
        req_pool_idx, start_pos : start_pos + num_tokens
    ] = RELEASED_DEVICE_KV_SLOT
