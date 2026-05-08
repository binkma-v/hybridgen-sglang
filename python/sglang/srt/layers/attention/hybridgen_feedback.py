"""HybridGen feedback latency model and adaptation policy.

Ported from hybridgen/llama/hybridgen/estimator.py and the
``feedback_update`` method in hybridgen/llama/hybridgen/hybrid_llama.py
(line 540).

The estimator answers: "given current sequence sizes and current topk
settings, will the host-side branch be hidden by the device-side branch
or will it become the bottleneck?". The adaptation policy nudges
``topk_ratio`` and ``cpu_k_cap`` based on that answer.

This module is pure Python / PyTorch-free so it imports cleanly without
needing the rest of sglang. The backend instantiates a single
``FeedbackController`` and calls ``maybe_step`` after each forward decode.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

LATENCY_MODE_ESTIMATED = "estimated"
LATENCY_MODE_MEASURED = "measured"

# Constants pulled from the original estimator (FLOP accounting rules).
_SOFTMAX_FACTOR = 5
_FACTOR = 2  # mul + add per FMA


@dataclasses.dataclass
class ModelSpecs:
    """Coarse model dimensions used by the analytical latency model."""

    B: int = 1
    D: int = 5120
    N_HEADS: int = 40
    N_LAYERS: int = 40
    D_FFN: int = 20480
    DTYPE_BYTES: int = 2
    UNIT: int = 1  # transfer-size denominator (kept for parity with original)


@dataclasses.dataclass
class SystemSpecs:
    """Hardware coefficients. Defaults match the original A100 + Xeon
    baseline; users override via `--hybridgen-feedback-*` knobs."""

    cpu_power: float = 1.0e7   # FLOPs/s on the CPU side (effective)
    gpu_power: float = 1.0e9   # FLOPs/s on the GPU side (effective)
    pcie_speed: float = 1.6e9  # bytes/s host->device


# ---------------------------------------------------------------------------
# Analytical latency components (1:1 port from estimator.py)
# ---------------------------------------------------------------------------


def _per_token_cache_bytes(m: ModelSpecs) -> float:
    head_dim = m.D / m.N_HEADS
    return (m.N_HEADS * head_dim) * m.DTYPE_BYTES / m.UNIT


def _hybridgen_size(topk: float, seq_len: int, m: ModelSpecs) -> float:
    # K vector + logits — both proportional to topk * seq_len
    k_bytes = topk * seq_len * _per_token_cache_bytes(m)
    logits_bytes = topk * seq_len * m.N_HEADS * m.DTYPE_BYTES / m.UNIT
    return k_bytes + logits_bytes


def _ffn_flops(m: ModelSpecs) -> float:
    return m.B * _FACTOR * (m.D * m.D_FFN + m.D_FFN + m.D_FFN * m.D)


def _gpu_attn_flops(seq_len: float, m: ModelSpecs) -> float:
    """Merged-softmax GPU work, parameterized by effective seq length.
    Mirrors hybridgen_attn_gpu_flops with ratio folded into seq_len."""
    kv_update = 2 * _FACTOR * m.B * 1 * m.D * m.D
    q_proj = _FACTOR * m.B * 1 * m.D * m.D
    softmax = m.B * _SOFTMAX_FACTOR * m.N_HEADS * seq_len
    weight_sum = m.B * _FACTOR * 1 * seq_len * m.D
    out_proj = m.B * _FACTOR * 1 * m.D * m.D
    return kv_update + q_proj + softmax + weight_sum + out_proj


def _cpu_qkt_flops(seq_len: float, m: ModelSpecs) -> float:
    """CPU-side QK^T: B * N_HEADS * 2 * d_head * seq_len = B * 2 * D * seq_len."""
    return m.B * _FACTOR * m.D * seq_len


def host_side_latency(
    seq_len: int, topk: float, model: ModelSpecs, sys: SystemSpecs
) -> float:
    transfer = _hybridgen_size(topk, seq_len, model) * model.B / sys.pcie_speed
    compute = _cpu_qkt_flops(seq_len, model) / sys.cpu_power
    return transfer + compute


def device_side_latency(
    gpu_seq_len: int,
    cpu_seq_len: int,
    topk: float,
    model: ModelSpecs,
    sys: SystemSpecs,
) -> float:
    """GPU-side merged attention + FFN (per layer). The merged seq is
    `cpu_seq_len * topk + gpu_seq_len`."""
    new_length = cpu_seq_len * topk + gpu_seq_len
    device_compute = _gpu_attn_flops(new_length, model) + _ffn_flops(model)
    # Subtract redundant CPU-equivalent work that the GPU would otherwise do.
    redundant = _cpu_qkt_flops(cpu_seq_len * topk, model)
    device_compute = max(device_compute - redundant, 0.0)
    return device_compute / sys.gpu_power


# ---------------------------------------------------------------------------
# Stateful estimator + adaptation policy
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FeedbackLatencyEstimator:
    """Latency oracle. In ESTIMATED mode it answers from the analytical
    model; in MEASURED mode it returns the last recorded times."""

    model_specs: ModelSpecs
    system_specs: SystemSpecs = dataclasses.field(default_factory=SystemSpecs)
    mode: str = LATENCY_MODE_ESTIMATED

    measured_host_compute_latency: Optional[float] = None
    measured_host_transfer_latency: Optional[float] = None
    measured_device_latency: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mode not in (LATENCY_MODE_ESTIMATED, LATENCY_MODE_MEASURED):
            raise ValueError(f"unsupported feedback mode: {self.mode!r}")

    # measurement hooks (called by backend if it has timed forward steps)

    def reset_measurements(self) -> None:
        self.measured_host_compute_latency = None
        self.measured_host_transfer_latency = None
        self.measured_device_latency = None

    def record_host_compute_latency(self, t: float) -> None:
        self.measured_host_compute_latency = max(0.0, float(t))

    def record_host_transfer_latency(self, t: float) -> None:
        self.measured_host_transfer_latency = max(0.0, float(t))

    def record_device_latency(self, t: float) -> None:
        self.measured_device_latency = max(0.0, float(t))

    # latency queries

    def host_latency(self, seq_len: int, topk: float) -> float:
        if self.mode == LATENCY_MODE_MEASURED:
            tot, has = 0.0, False
            for v in (
                self.measured_host_compute_latency,
                self.measured_host_transfer_latency,
            ):
                if v is not None:
                    tot += v
                    has = True
            if has:
                return tot
        return host_side_latency(
            seq_len, topk, self.model_specs, self.system_specs
        )

    def device_latency(
        self, gpu_seq_len: int, cpu_seq_len: int, topk: float
    ) -> float:
        if (
            self.mode == LATENCY_MODE_MEASURED
            and self.measured_device_latency is not None
        ):
            return self.measured_device_latency
        return device_side_latency(
            gpu_seq_len,
            cpu_seq_len,
            topk,
            self.model_specs,
            self.system_specs,
        )


@dataclasses.dataclass
class FeedbackPolicy:
    """Caps + step sizes for the adaptation rule. Tunables come from the
    backend's CLI flags."""

    topk_min_ratio: float = 0.01
    topk_max_ratio: float = 0.5
    cpu_k_cap_min: int = 32
    grow_factor: float = 1.2
    shrink_factor: float = 0.8
    cap_shrink_factor: float = 0.5


def adapt(
    topk_ratio: float,
    cpu_k_cap: int,
    cpu_cache_len: int,
    host_lat: float,
    device_lat: float,
    policy: FeedbackPolicy,
) -> Tuple[float, int]:
    """Returns the updated (topk_ratio, cpu_k_cap) pair given the latest
    latency reading. Mirrors hybrid_llama.py:540 feedback_update."""
    if cpu_cache_len <= 0:
        return topk_ratio, cpu_k_cap

    if host_lat <= device_lat:
        # Host is hidden under device-side work — spend more on accuracy.
        if topk_ratio < policy.topk_max_ratio:
            topk_ratio = min(topk_ratio * policy.grow_factor, policy.topk_max_ratio)
        if cpu_k_cap:
            cpu_k_cap = max(cpu_k_cap + 1, int(cpu_k_cap * policy.grow_factor))
            cpu_k_cap = min(cpu_k_cap, cpu_cache_len)
            if cpu_k_cap >= cpu_cache_len:
                cpu_k_cap = 0
    else:
        # Host is the bottleneck — trim work. Shrink both top-k density and
        # CPU scan window; the cap reduction is the main lever for long
        # contexts because CPU QK^T cost scales with scanned K length.
        if topk_ratio > policy.topk_min_ratio:
            topk_ratio = max(topk_ratio * policy.shrink_factor, policy.topk_min_ratio)
        if cpu_k_cap == 0:
            cpu_k_cap = max(int(cpu_cache_len * policy.cap_shrink_factor), policy.cpu_k_cap_min)
        else:
            cpu_k_cap = max(int(cpu_k_cap * policy.cap_shrink_factor), policy.cpu_k_cap_min)
    return topk_ratio, cpu_k_cap


@dataclasses.dataclass
class FeedbackController:
    """Holds the estimator + policy + step counter that the backend touches."""

    estimator: FeedbackLatencyEstimator
    policy: FeedbackPolicy = dataclasses.field(default_factory=FeedbackPolicy)
    interval: int = 0   # 0 disables feedback
    _step: int = 0

    def maybe_step(
        self,
        topk_ratio: float,
        cpu_k_cap: int,
        cpu_cache_len: int,
        gpu_cache_len: int,
        effective_cpu_k_len: Optional[int] = None,
    ) -> Tuple[float, int]:
        if self.interval <= 0 or cpu_cache_len <= 0:
            return topk_ratio, cpu_k_cap
        self._step += 1
        if self._step % self.interval != 0:
            return topk_ratio, cpu_k_cap

        score_len = effective_cpu_k_len if effective_cpu_k_len is not None else cpu_cache_len
        score_len = max(0, min(score_len, cpu_cache_len))
        host_lat = self.estimator.host_latency(score_len, topk_ratio)
        device_lat = self.estimator.device_latency(
            gpu_cache_len, score_len, topk_ratio
        )
        return adapt(
            topk_ratio,
            cpu_k_cap,
            cpu_cache_len,
            host_lat,
            device_lat,
            self.policy,
        )
