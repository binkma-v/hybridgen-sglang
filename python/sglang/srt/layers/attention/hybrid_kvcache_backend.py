from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybridgen_feedback import (
    FeedbackController,
    FeedbackLatencyEstimator,
    FeedbackPolicy,
    LATENCY_MODE_ESTIMATED,
    ModelSpecs,
    SystemSpecs,
)
from sglang.srt.layers.attention.hybridgen_topk import (
    CPUTopKWorkspace,
    compute_topk_on_cpu,
    gpu_dense_attention_partial_triton,
    merge_gpu_partial_with_topk_triton,
    merged_softmax_attention_per_head_v_triton,
)
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.mem_cache.hybridgen_release import (
    mark_device_indices_released,
    valid_device_indices,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class HybridKVCacheAttnBackend(AttentionBackend):
    """Attention backend implementing the HybridGen scheme:
    keep recent KV on GPU, offload older KV to host, do CPU-side top-k selection
    and merged softmax over (GPU dense + CPU top-k).

    The backend keeps per-request host-segment metadata, backs up evicted KV to
    `MHATokenToKVPoolHost`, and physically releases request-owned GPU shadows
    during decode when the prefix is not protected by RadixCache.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        sa = model_runner.server_args
        self.topk_ratio: float = sa.hybridgen_topk_ratio
        self.cpu_k_cap: int = sa.hybridgen_cpu_k_cap
        # Initial values, used to reset adaptive state on each fresh request
        # (the original hybridgen runs one batch at a time, so adapt() always
        # started from these defaults; sglang's serving loop reuses one
        # backend across requests so we restore here).
        self._initial_topk_ratio: float = self.topk_ratio
        self._initial_cpu_k_cap: int = self.cpu_k_cap
        self.gpu_cache_factor: float = sa.hybridgen_gpu_cache_factor
        self.feedback_interval: int = sa.hybridgen_feedback_interval
        self.gpu_q_proj: bool = sa.hybridgen_gpu_q_proj
        self.host_size_gb: int = sa.hybridgen_host_size
        self.host_ratio: float = sa.hybridgen_host_ratio
        self.host_layout: str = sa.hybridgen_host_layout
        self.io_backend: str = sa.hybridgen_io_backend

        self.device = model_runner.device
        self.page_size = model_runner.server_args.page_size
        self._fallback = TorchNativeAttnBackend(model_runner)
        self._device_allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)

        # Feedback scheduler — analytical model by default; the backend
        # avoids the wall-clock measurement path so this stays free.
        self._feedback = self._build_feedback_controller(model_runner)

        # Host pool is created lazily on first forward to avoid touching CPU
        # memory if the backend is constructed but never invoked (e.g. CUDA
        # graph dry-runs, capability probing).
        self._host_pool: Optional[MHATokenToKVPoolHost] = None
        self._host_pool_init_attempted = False
        self._device_pool: Optional[MHATokenToKVPool] = None

        # Per-request host-segment bookkeeping.
        # Maps req_pool_idx -> CPU long tensor of host token slot ids,
        # chronologically ordered (oldest first). Storing as tensor (not
        # list) avoids rebuilding a tensor in every layer's forward_decode.
        self._host_segments: Dict[int, torch.Tensor] = {}
        # Per-request prompt length, tracked across chunked prefill chunks
        # (we keep the largest seq_len seen during prefill). Used to size
        # the GPU residency cap = gpu_cache_factor * prompt_len, matching
        # hybridgen/pytorch_backend.py:280:
        #     gpu_cache_max = max(1, min(int(prompt_len * factor), max_len))
        # factor=1.0 (default) keeps the whole prompt on GPU and only evicts
        # decode tokens beyond it; factor<1.0 evicts part of the prompt
        # during prefill; factor>1.0 leaves room for some decode.
        self._prompt_lens: Dict[int, int] = {}

        self._topk_workspace = CPUTopKWorkspace()
        self._q_to_kv_head_cache: Dict[tuple[int, int], torch.Tensor] = {}
        self._q_to_kv_topk_head_cache: Dict[tuple[int, int, int], torch.Tensor] = {}
        self._h2d_stream: Optional[torch.cuda.Stream] = None
        self._gpu_overlap_stream: Optional[torch.cuda.Stream] = None
        self._pinned_scores: Optional[torch.Tensor] = None
        self._pinned_topk_v: Optional[torch.Tensor] = None
        self._h2d_overlap_min_bytes = 1 << 20
        self._partial_overlap_min_cpu_tokens = 4096
        self._partial_overlap_min_gpu_tokens = 1024
        if torch.device(self.device).type == "cuda":
            self._h2d_stream = torch.cuda.Stream(device=self.device)
            self._gpu_overlap_stream = torch.cuda.Stream(device=self.device)

    # ------------------------------------------------------------------
    # Feedback scheduler bootstrap
    # ------------------------------------------------------------------

    def _build_feedback_controller(
        self, model_runner: ModelRunner
    ) -> Optional[FeedbackController]:
        if self.feedback_interval <= 0:
            return None
        try:
            cfg = model_runner.model_config
            hidden_size = int(getattr(cfg, "hidden_size", 0))
            num_heads = int(getattr(cfg, "num_attention_heads", 0))
            num_layers = int(getattr(cfg, "num_hidden_layers", 0))
            d_ffn = int(
                getattr(cfg, "intermediate_size", hidden_size * 4) or hidden_size * 4
            )
            kv_dtype = getattr(model_runner.token_to_kv_pool, "store_dtype", None)
            dtype_bytes = getattr(kv_dtype, "itemsize", 2) if kv_dtype else 2
            specs = ModelSpecs(
                B=1,
                D=hidden_size or 4096,
                N_HEADS=num_heads or 32,
                N_LAYERS=num_layers or 32,
                D_FFN=d_ffn,
                DTYPE_BYTES=dtype_bytes,
            )
            estimator = FeedbackLatencyEstimator(
                model_specs=specs,
                system_specs=SystemSpecs(),
                mode=LATENCY_MODE_ESTIMATED,
            )
            return FeedbackController(
                estimator=estimator,
                policy=FeedbackPolicy(),
                interval=self.feedback_interval,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "HybridKVCacheAttnBackend: feedback controller disabled (%s).",
                exc,
            )
            return None

    def _maybe_step_feedback(self, forward_batch: ForwardBatch) -> None:
        """After each decode step, ask the controller whether to nudge
        ``self.topk_ratio`` / ``self.cpu_k_cap``. Uses representative cache
        sizes from the largest in-flight request to avoid per-request churn."""
        if self._feedback is None or not self._host_segments:
            return
        # Pick the request with the largest host segment as the representative
        # working point — that's where the hybrid path costs the most.
        rep_idx = max(
            self._host_segments.keys(),
            key=lambda r: len(self._host_segments[r]),
        )
        cpu_len = len(self._host_segments[rep_idx])
        seq_lens = forward_batch.seq_lens.tolist()
        req_pool_indices = forward_batch.req_pool_indices.tolist()
        try:
            i = req_pool_indices.index(rep_idx)
            gpu_len = seq_lens[i] - cpu_len
        except ValueError:
            return
        effective_cpu_len = self._effective_cpu_k_len(cpu_len)
        new_ratio, new_cap = self._feedback.maybe_step(
            self.topk_ratio,
            self.cpu_k_cap,
            cpu_len,
            gpu_len,
            effective_cpu_len,
        )
        if new_ratio != self.topk_ratio or new_cap != self.cpu_k_cap:
            logger.info(
                "HybridKVCacheAttnBackend feedback: topk_ratio %.4f -> %.4f, "
                "cpu_k_cap %d -> %d (cpu_len=%d, effective_cpu_len=%d, gpu_len=%d).",
                self.topk_ratio, new_ratio, self.cpu_k_cap, new_cap,
                cpu_len, effective_cpu_len, gpu_len,
            )
        self.topk_ratio = new_ratio
        self.cpu_k_cap = new_cap

    # ------------------------------------------------------------------
    # Host pool lifecycle
    # ------------------------------------------------------------------

    def _ensure_host_pool(self, forward_batch: ForwardBatch) -> None:
        """Lazy-init the host KV pool from the device pool that the
        forward batch carries. No-op after the first successful call."""
        if self._host_pool_init_attempted:
            return
        self._host_pool_init_attempted = True

        device_pool = forward_batch.token_to_kv_pool
        # Only MHA models are supported for now; other pool types
        # (MLA / NSA / hybrid-linear) are out of scope for this stage.
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        if not isinstance(device_pool, MHATokenToKVPool):
            logger.warning(
                "HybridKVCacheAttnBackend: device pool %s is not MHA; "
                "host segment disabled, falling back to torch_native.",
                type(device_pool).__name__,
            )
            return

        try:
            from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost

            self._host_pool = MHATokenToKVPoolHost(
                device_pool,
                self.host_ratio,
                self.host_size_gb,
                self.page_size,
                self.host_layout,
            )
            self._device_pool = device_pool
            logger.info(
                "HybridKVCacheAttnBackend: host pool ready "
                "(size=%d slots, layout=%s, ratio=%.2f).",
                self._host_pool.size,
                self.host_layout,
                self.host_ratio,
            )
        except Exception as exc:  # noqa: BLE001 — surface any backend init issue
            logger.exception(
                "HybridKVCacheAttnBackend: failed to allocate host pool (%s). "
                "Falling back to GPU-only attention.",
                exc,
            )
            self._host_pool = None
            self._device_pool = None

    # ------------------------------------------------------------------
    # Host pool helpers
    # ------------------------------------------------------------------

    def _alloc_host(self, n_tokens: int) -> Optional[torch.Tensor]:
        """Allocate `n_tokens` host KV slots. Returns None if the pool is
        absent or out of capacity. The returned tensor lives on CPU."""
        if self._host_pool is None:
            return None
        # MHATokenToKVPoolHost requires alloc requests aligned to page_size.
        rounded = ((n_tokens + self.page_size - 1) // self.page_size) * self.page_size
        return self._host_pool.alloc(rounded)

    def _free_host(self, host_indices: torch.Tensor) -> None:
        if self._host_pool is None or host_indices is None:
            return
        self._host_pool.free(host_indices)

    def _effective_cpu_k_len(self, n_host: int) -> int:
        if self.cpu_k_cap > 0:
            return min(n_host, self.cpu_k_cap)
        return n_host

    def _q_to_kv_heads(self, num_q_heads: int, num_kv_heads: int) -> torch.Tensor:
        key = (num_q_heads, num_kv_heads)
        cached = self._q_to_kv_head_cache.get(key)
        if cached is not None:
            return cached
        assert num_q_heads % num_kv_heads == 0, (
            f"num_q_heads ({num_q_heads}) must be a multiple of "
            f"num_kv_heads ({num_kv_heads})"
        )
        repeat = num_q_heads // num_kv_heads
        mapping = torch.arange(num_q_heads, dtype=torch.long) // repeat
        self._q_to_kv_head_cache[key] = mapping
        return mapping

    def _q_to_kv_heads_for_topk(
        self, num_q_heads: int, num_kv_heads: int, top_k: int
    ) -> torch.Tensor:
        key = (num_q_heads, num_kv_heads, top_k)
        cached = self._q_to_kv_topk_head_cache.get(key)
        if cached is not None:
            return cached
        mapping = self._q_to_kv_heads(num_q_heads, num_kv_heads).repeat_interleave(
            top_k
        )
        self._q_to_kv_topk_head_cache[key] = mapping
        return mapping

    @staticmethod
    def _ensure_pinned_workspace(
        tensor: Optional[torch.Tensor],
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            tensor is None
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not tensor.is_pinned()
        ):
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        return tensor

    def _copy_topk_to_device_async(
        self,
        topk_scores_cpu: torch.Tensor,
        v_topk_host: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.cuda.Event]]:
        """Stage CPU top-k artefacts in pinned memory and copy on a side stream."""
        if self._h2d_stream is None or torch.device(device).type != "cuda":
            return (
                topk_scores_cpu.to(device=device, dtype=dtype, non_blocking=True),
                v_topk_host.to(device=device, dtype=dtype, non_blocking=True),
                None,
            )
        dtype_bytes = torch.empty((), dtype=dtype).element_size()
        transfer_bytes = (
            topk_scores_cpu.numel() + v_topk_host.numel()
        ) * dtype_bytes
        if transfer_bytes < self._h2d_overlap_min_bytes:
            return (
                topk_scores_cpu.to(device=device, dtype=dtype, non_blocking=True),
                v_topk_host.to(device=device, dtype=dtype, non_blocking=True),
                None,
            )

        if topk_scores_cpu.dtype != dtype:
            topk_scores_cpu = topk_scores_cpu.to(dtype=dtype)
        if v_topk_host.dtype != dtype:
            v_topk_host = v_topk_host.to(dtype=dtype)

        self._pinned_scores = self._ensure_pinned_workspace(
            self._pinned_scores, tuple(topk_scores_cpu.shape), dtype
        )
        self._pinned_topk_v = self._ensure_pinned_workspace(
            self._pinned_topk_v, tuple(v_topk_host.shape), dtype
        )
        self._pinned_scores.copy_(topk_scores_cpu)
        self._pinned_topk_v.copy_(v_topk_host)

        scores_gpu = torch.empty(
            topk_scores_cpu.shape, device=device, dtype=dtype
        )
        v_topk_gpu = torch.empty(v_topk_host.shape, device=device, dtype=dtype)
        copy_event = torch.cuda.Event()
        with torch.cuda.stream(self._h2d_stream):
            scores_gpu.copy_(self._pinned_scores, non_blocking=True)
            v_topk_gpu.copy_(self._pinned_topk_v, non_blocking=True)
            copy_event.record(self._h2d_stream)
        return scores_gpu, v_topk_gpu, copy_event

    def _should_overlap_gpu_partial(
        self, n_host_scored: int, n_gpu_tokens: int
    ) -> bool:
        # Splitting the original fused attention into dense-partial + merge adds
        # one kernel launch, one event wait, and global-memory partial traffic.
        # It only pays off when CPU top-k has enough work to hide that overhead.
        return (
            self._gpu_overlap_stream is not None
            and n_host_scored >= self._partial_overlap_min_cpu_tokens
            and n_gpu_tokens >= self._partial_overlap_min_gpu_tokens
        )

    def _can_release_device_slots(
        self, cache_protected_len: int, is_prefill: bool
    ) -> bool:
        """Whether HybridGen owns the GPU slots and can physically free them."""
        return (
            self._device_allocator is not None
            and not is_prefill
            and cache_protected_len == 0
        )

    def _release_device_slots(
        self,
        req_to_token: torch.Tensor,
        req_pool_idx: int,
        start_pos: int,
        device_indices: torch.Tensor,
    ) -> int:
        """Return GPU KV slots after their KV has been backed up to host.

        The request row is marked with 0 for released prefix positions so the
        scheduler-side cleanup path can distinguish already-released slots from
        still-GPU-resident KV and avoid double-freeing them.
        """
        span_len = int(device_indices.numel())
        if self._device_allocator is None or span_len == 0:
            return 0
        live_device_indices = valid_device_indices(device_indices).contiguous()
        if live_device_indices.numel() > 0:
            self._device_allocator.free(live_device_indices.to(dtype=torch.int64))
        mark_device_indices_released(req_to_token, req_pool_idx, start_pos, span_len)
        return int(live_device_indices.numel())

    def _release_existing_host_shadows(
        self,
        req_to_token: torch.Tensor,
        req_pool_idx: int,
        n_host: int,
        cache_protected_len: int,
        is_prefill: bool,
    ) -> int:
        """Release GPU shadows for host-backed prefix tokens after prefill.

        Prefill still delegates to torch_native, so host-backed prompt tokens
        must stay on GPU until decode begins. Once in decode, their host copies
        are authoritative for the hybrid path and the GPU shadow slots can be
        returned to the allocator for requests that do not share a protected
        RadixCache prefix.
        """
        if n_host <= 0 or not self._can_release_device_slots(
            cache_protected_len, is_prefill
        ):
            return 0
        shadow_slots = req_to_token[req_pool_idx, :n_host]
        live_shadow_slots = valid_device_indices(shadow_slots).contiguous()
        if live_shadow_slots.numel() == 0:
            return 0
        self._device_allocator.free(live_shadow_slots.to(dtype=torch.int64))
        mark_device_indices_released(req_to_token, req_pool_idx, 0, n_host)
        return int(live_shadow_slots.numel())

    def _backup_layer_to_host(
        self,
        layer_id: int,
        device_indices: torch.Tensor,
        host_indices: torch.Tensor,
    ) -> None:
        """Copy KV at given device-pool indices to the host pool."""
        if self._host_pool is None or self._device_pool is None:
            return
        # backup_from_device_all_layer copies all layers in one shot; for
        # per-layer eviction we expose a thin wrapper that the caller can
        # later replace with batched all-layer eviction.
        self._host_pool.backup_from_device_all_layer(
            self._device_pool,
            host_indices,
            device_indices,
            self.io_backend,
        )

    def _load_layer_from_host(
        self,
        layer_id: int,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
    ) -> None:
        """Copy KV at given host indices back to the device pool for a layer."""
        if self._host_pool is None or self._device_pool is None:
            return
        self._host_pool.load_to_device_per_layer(
            self._device_pool,
            host_indices,
            device_indices,
            layer_id,
            self.io_backend,
        )

    # ------------------------------------------------------------------
    # AttentionBackend interface
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._ensure_host_pool(forward_batch)
        if self._host_pool is not None:
            self._reset_finished_requests(forward_batch)
            self._maybe_evict_per_request(forward_batch)
        self._fallback.init_forward_metadata(forward_batch)

    # ------------------------------------------------------------------
    # Eviction & per-request lifecycle
    # ------------------------------------------------------------------

    def _reset_finished_requests(self, forward_batch: ForwardBatch) -> None:
        """Detect fresh prefills (req_pool_idx reused by a new request) and
        clear stale per-request state.

        Signal: in extend mode, when extend_seq_lens[i] == seq_lens[i], there
        is no cached prefix for this request — i.e. it's a brand-new request
        starting from scratch. Any state we're carrying for that idx came
        from a finished request and must be released.
        """
        if not forward_batch.forward_mode.is_extend():
            return
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            return
        req_pool_indices = forward_batch.req_pool_indices.tolist()
        seq_lens = forward_batch.seq_lens.tolist()
        extend_lens = extend_seq_lens.tolist()
        for i, req_pool_idx in enumerate(req_pool_indices):
            if extend_lens[i] != seq_lens[i]:
                continue
            stale_host = self._host_segments.pop(req_pool_idx, None)
            if stale_host is not None and stale_host.numel() > 0:
                self._free_host(stale_host)
            self._prompt_lens.pop(req_pool_idx, None)
        # Any extend-mode batch is the start of a (possibly chunked) prefill.
        # The feedback controller's topk_ratio / cpu_k_cap are global state that
        # would otherwise carry adaptation from a previous request. Reset to
        # initial values so each new request starts from a clean slate. Note
        # we cannot gate on `extend_lens[i] != seq_lens[i]` because sglang's
        # RadixCache lets new requests reuse a cached prefix, in which case
        # extend_lens < seq_lens even for a brand-new prompt.
        self.topk_ratio = self._initial_topk_ratio
        self.cpu_k_cap = self._initial_cpu_k_cap

    def _maybe_evict_per_request(self, forward_batch: ForwardBatch) -> None:
        """For each request whose GPU-resident KV exceeds gpu_cap, copy the
        oldest excess tokens' KV to the host pool and append their host
        indices to `_host_segments`. During decode, request-owned GPU slots are
        released after backup; during extend or cache-protected prefixes they
        remain as shadow copies for correctness."""
        if self._device_pool is None or self._host_pool is None:
            return

        req_pool_indices = forward_batch.req_pool_indices.tolist()
        seq_lens = forward_batch.seq_lens.tolist()
        req_to_token = forward_batch.req_to_token_pool.req_to_token

        is_prefill = bool(getattr(forward_batch.forward_mode, "is_extend", lambda: False)())
        cache_protected_lens = getattr(forward_batch, "cache_protected_lens", None)
        for i, req_pool_idx in enumerate(req_pool_indices):
            seq_len = seq_lens[i]
            cache_protected_len = (
                int(cache_protected_lens[i]) if cache_protected_lens is not None else -1
            )
            # Track the largest seq_len observed during prefill as the prompt
            # length. After decode begins we freeze it (so cap stays stable).
            existing = self._prompt_lens.get(req_pool_idx, 0)
            if is_prefill or existing == 0:
                self._prompt_lens[req_pool_idx] = max(existing, seq_len)

            prompt_len = self._prompt_lens[req_pool_idx]
            # gpu_cache_factor is the fraction of prompt KV kept on GPU.
            # cap = factor * prompt_len, with a minimum of 1 to avoid 0-cap.
            gpu_cap = max(int(self.gpu_cache_factor * prompt_len), 1)

            n_host = len(self._host_segments.get(req_pool_idx, ()))
            self._release_existing_host_shadows(
                req_to_token,
                req_pool_idx,
                n_host,
                cache_protected_len,
                is_prefill,
            )
            n_gpu = seq_len - n_host
            excess = n_gpu - gpu_cap
            if excess <= 0:
                continue

            # Page-align the eviction batch so the host allocator is happy.
            ps = self.page_size if self.page_size > 0 else 1
            evict_n = ((excess + ps - 1) // ps) * ps
            # Keep at least gpu_cap tokens resident. The previous page rounding
            # can otherwise over-evict the whole GPU segment when page_size > 1.
            max_evict_n = max(n_gpu - max(gpu_cap, 1), 0)
            evict_n = min(evict_n, max_evict_n)
            if ps > 1:
                evict_n = (evict_n // ps) * ps
            if evict_n <= 0:
                continue

            # GPU slot indices for positions [n_host, n_host + evict_n).
            # Keep the native dtype that req_to_token uses; the JIT hicache
            # kernel checks tensor descriptors strictly.
            gpu_slots = req_to_token[
                req_pool_idx, n_host : n_host + evict_n
            ].contiguous()

            host_slots_cpu = self._alloc_host(evict_n)
            if host_slots_cpu is None:
                logger.warning(
                    "HybridKVCacheAttnBackend: host pool exhausted; "
                    "skipping eviction of %d tokens for req_pool_idx=%d.",
                    evict_n,
                    req_pool_idx,
                )
                continue
            # _alloc_host may return more slots than requested (page rounding);
            # use only the first `evict_n`. Free the tail to avoid leaks.
            if host_slots_cpu.numel() > evict_n:
                tail = host_slots_cpu[evict_n:]
                self._free_host(tail)
                host_slots_cpu = host_slots_cpu[:evict_n]

            # When io_backend == "kernel", the JIT kernel needs both index
            # tensors on the device side; mirror sglang's HiCacheController.
            if self.io_backend == "kernel":
                host_slots_arg = host_slots_cpu.to(
                    gpu_slots.device, dtype=gpu_slots.dtype, non_blocking=True
                )
            else:
                host_slots_arg = host_slots_cpu.to(dtype=gpu_slots.dtype)

            try:
                self._host_pool.backup_from_device_all_layer(
                    self._device_pool,
                    host_slots_arg,
                    gpu_slots,
                    self.io_backend,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "HybridKVCacheAttnBackend: backup_from_device_all_layer "
                    "failed (%s); rolling back host slots.",
                    exc,
                )
                self._free_host(host_slots_cpu)
                continue

            # Physical GPU release is safe only after prefill, because
            # forward_extend still delegates to torch_native and needs the full
            # prefix on GPU. Also avoid releasing RadixCache-protected prefixes:
            # those slots are owned by the cache tree and may be shared.
            release_device = self._can_release_device_slots(
                cache_protected_len, is_prefill
            )

            # Append the new host slot ids (CPU long tensor) to this
            # request's segment.
            new_slots = host_slots_cpu.to("cpu", dtype=torch.long)
            existing = self._host_segments.get(req_pool_idx)
            if existing is None or existing.numel() == 0:
                self._host_segments[req_pool_idx] = new_slots
            else:
                self._host_segments[req_pool_idx] = torch.cat(
                    [existing, new_slots]
                )
            if release_device:
                self._release_device_slots(
                    req_to_token, req_pool_idx, n_host, gpu_slots
                )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self._fallback.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # Fast path: no request has been evicted yet — behave like torch_native.
        # `_host_segments` is populated by eviction; affected requests then use
        # the hybrid path.
        if not self._host_segments or self._host_pool is None:
            out = self._fallback.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        else:
            out = self._run_hybrid_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )
        # Run feedback only at the last layer of the decode step so all layers
        # in this step see consistent topk_ratio / cpu_k_cap.
        last_layer_id = (
            self._device_pool.end_layer - 1
            if self._device_pool is not None
            else layer.layer_id
        )
        if layer.layer_id == last_layer_id:
            self._maybe_step_feedback(forward_batch)
        return out

    def support_triton(self):
        return False

    # ------------------------------------------------------------------
    # Hybrid decode (per-request loop)
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
        """Per-request loop. For each request with a non-empty host segment,
        runs the hybrid path (CPU top-k + merged softmax). Other requests use
        a torch_native-equivalent dense SDPA path."""

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

        for seq_idx in range(seq_lens.shape[0]):
            req_pool_idx = int(req_pool_indices[seq_idx].item())
            seq_len_kv = int(seq_lens[seq_idx].item())
            host_indices = self._host_segments.get(req_pool_idx)
            n_host = 0 if host_indices is None else int(host_indices.shape[0])

            per_req_q = q_view[seq_idx]  # (nq, qk_head_dim)
            # GPU-resident segment is positions [n_host, seq_len_kv).
            # Tokens at [0, n_host) live on host and must be excluded from the
            # GPU-side dense attention to avoid double-counting against the
            # top-k selection. Their GPU shadows may already have been released.
            per_req_tokens = req_to_token[req_pool_idx, n_host:seq_len_kv]

            if n_host == 0:
                per_req_k = k_cache[per_req_tokens]  # (S_gpu, nkv, qk_head_dim)
                per_req_v = v_cache[per_req_tokens]  # (S_gpu, nkv, v_head_dim)
                # Dense fallback for this request — equivalent to torch_native.
                k_dense = per_req_k.movedim(0, 1)  # (nkv, S_gpu, qk_head_dim)
                v_dense = per_req_v.movedim(0, 1)
                if per_req_q.dtype != k_dense.dtype:
                    k_dense = k_dense.to(per_req_q.dtype)
                    v_dense = v_dense.to(per_req_q.dtype)
                use_gqa = nq != nkv
                # SDPA expects (..., L, E). Add batch + sequence dims.
                out_seq = scaled_dot_product_attention(
                    per_req_q.unsqueeze(0).unsqueeze(2),  # (1, nq, 1, qk_head_dim)
                    k_dense.unsqueeze(0),                 # (1, nkv, S_gpu, qk_head_dim)
                    v_dense.unsqueeze(0),                 # (1, nkv, S_gpu, v_head_dim)
                    enable_gqa=use_gqa,
                    scale=scaling,
                    is_causal=False,
                ).squeeze(0).squeeze(1)                   # (nq, v_head_dim)
                out_view[seq_idx] = out_seq.to(dtype)
                continue

            # Hybrid path for this request.
            n_host_total = int(host_indices.shape[0])
            if self.cpu_k_cap > 0 and n_host_total > self.cpu_k_cap:
                # Trim the host segment we score against — we keep the most
                # recent `cpu_k_cap` host tokens to bound CPU compute.
                host_idx_tensor = host_indices[-self.cpu_k_cap:]
            else:
                host_idx_tensor = host_indices
            n_host_scored = int(host_idx_tensor.shape[0])
            top_k = max(int(self.topk_ratio * n_host_scored), 1)

            gpu_partial = None
            gpu_partial_event = None
            if self._should_overlap_gpu_partial(n_host_scored, per_req_tokens.numel()):
                current_stream = torch.cuda.current_stream(device)
                with torch.cuda.stream(self._gpu_overlap_stream):
                    self._gpu_overlap_stream.wait_stream(current_stream)
                    per_req_k = k_cache[
                        per_req_tokens
                    ]  # (S_gpu, nkv, qk_head_dim)
                    per_req_v = v_cache[
                        per_req_tokens
                    ]  # (S_gpu, nkv, v_head_dim)
                    if per_req_q.dtype != per_req_k.dtype:
                        per_req_k = per_req_k.to(per_req_q.dtype)
                        per_req_v = per_req_v.to(per_req_q.dtype)
                    gpu_partial = gpu_dense_attention_partial_triton(
                        per_req_q,
                        per_req_k,
                        per_req_v,
                        nq,
                        nkv,
                        scaling,
                    )
                    gpu_partial_event = torch.cuda.Event()
                    gpu_partial_event.record(self._gpu_overlap_stream)

            # Read host K for the segment (CPU tensor, model dtype).
            k_host_seg = self._host_pool.k_buffer[layer.layer_id, host_idx_tensor]
            # CPU top-k. Keep model dtype throughout (no float32 upcast):
            # avoids an O(S * nkv * d) materialization per layer per step.
            q_cpu = per_req_q.detach().to("cpu")
            topk_scores_cpu, topk_local_idx_cpu = compute_topk_on_cpu(
                q_cpu, k_host_seg, top_k, nq, nkv, scaling, self._topk_workspace
            )
            k_eff = topk_scores_cpu.shape[1]

            # Gather each Q head's selected V directly. This keeps the same
            # sparse softmax set as the old unique+scatter representation, but
            # avoids building shared columns and per-head -inf masks.
            host_idx_per_head = torch.take(
                host_idx_tensor, topk_local_idx_cpu.reshape(-1)
            )
            kv_heads = self._q_to_kv_heads_for_topk(nq, nkv, k_eff)
            v_topk_host = self._host_pool.v_buffer[
                layer.layer_id, host_idx_per_head, kv_heads
            ].view(nq, k_eff, layer.v_head_dim)

            scores_gpu, v_topk_gpu, topk_copy_event = self._copy_topk_to_device_async(
                topk_scores_cpu, v_topk_host, device, per_req_q.dtype
            )

            if topk_copy_event is not None:
                torch.cuda.current_stream(device).wait_event(topk_copy_event)

            if gpu_partial_event is not None:
                torch.cuda.current_stream(device).wait_event(gpu_partial_event)

            if gpu_partial is not None:
                out_seq = merge_gpu_partial_with_topk_triton(
                    gpu_partial[0],
                    gpu_partial[1],
                    gpu_partial[2],
                    scores_gpu,
                    v_topk_gpu,
                )
            else:
                per_req_k = k_cache[per_req_tokens]  # (S_gpu, nkv, qk_head_dim)
                per_req_v = v_cache[per_req_tokens]  # (S_gpu, nkv, v_head_dim)
                if per_req_q.dtype != per_req_k.dtype:
                    per_req_k = per_req_k.to(per_req_q.dtype)
                    per_req_v = per_req_v.to(per_req_q.dtype)
                out_seq = merged_softmax_attention_per_head_v_triton(
                    per_req_q,
                    per_req_k,
                    per_req_v,
                    scores_gpu,
                    v_topk_gpu,
                    nq,
                    nkv,
                    scaling,
                )
            out_view[seq_idx] = out_seq

        return output
