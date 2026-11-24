"""Training throughput, MFU, and GPU memory estimator and step-level profiler.

Provides:

* :func:`estimate_training_memory`: analytical per-GPU memory breakdown (parameters,
  gradients, AdamW optimizer states, and activations) across ``single``, ``ddp``,
  ``fsdp`` / ``deepspeed_zero3``, ``deepspeed_zero2``, activation checkpointing, and CPU offload.
* :func:`estimate_cluster_mfu`: Model FLOPs Utilization across ``world_size`` GPUs.
* :class:`TrainingProfiler`: lightweight phase-level step profiler (``data``, ``forward``,
  ``backward``, ``optim``) with percentile summary and JSON export.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from disttrain.metrics import flops_per_token, peak_flops

BYTES_PER_GB = float(1 << 30)


@dataclass(frozen=True)
class MemoryEstimate:
    """Analytical per-GPU memory breakdown in bytes and gigabytes."""

    strategy: str
    world_size: int
    param_bytes: int
    grad_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    total_bytes: int
    per_gpu_gb: float
    vram_gb: float
    fits_in_vram: bool


def estimate_training_memory(
    n_params: int,
    n_layer: int,
    n_embd: int,
    n_head: int,
    seq_len: int,
    micro_batch_size: int,
    *,
    strategy: str = "ddp",
    world_size: int = 1,
    dtype: str = "bfloat16",
    activation_checkpointing: bool = False,
    cpu_offload: bool = False,
    vram_gb: float = 40.0,
) -> MemoryEstimate:
    """Estimate per-GPU peak training memory (ZeRO accounting + transformer activations)."""
    if n_params <= 0 or n_layer <= 0 or n_embd <= 0 or n_head <= 0 or seq_len <= 0 or micro_batch_size <= 0:
        raise ValueError("Model dimensions, seq_len, and micro_batch_size must be positive")
    if world_size < 1:
        raise ValueError("world_size must be >= 1")

    bytes_per_elem = 2 if dtype in ("bfloat16", "float16") else 4
    # Mixed precision keeps fp32 master weights (4B), fp32 grads (4B), and AdamW moments (8B).
    base_param = n_params * 4
    base_grad = n_params * 4
    base_optim = n_params * 8

    if strategy in ("single", "ddp"):
        param_bytes, grad_bytes, optim_bytes = base_param, base_grad, base_optim
    elif strategy == "deepspeed_zero2":
        param_bytes = base_param
        grad_bytes = math.ceil(base_grad / world_size)
        optim_bytes = math.ceil(base_optim / world_size)
    elif strategy in ("fsdp", "deepspeed_zero3"):
        param_bytes = math.ceil(base_param / world_size)
        grad_bytes = math.ceil(base_grad / world_size)
        optim_bytes = math.ceil(base_optim / world_size)
    else:
        raise ValueError(f"Unsupported strategy for memory estimation: {strategy!r}")

    if cpu_offload and strategy in ("fsdp", "deepspeed_zero2", "deepspeed_zero3"):
        optim_bytes = 0
        if strategy in ("fsdp", "deepspeed_zero3"):
            param_bytes = 0
            grad_bytes = 0

    # Standard transformer activation approximation (Megatron-LM selective/full formula):
    # ~ (10 * B * T * d + 2 * B * n_head * T) * bytes_per_elem per layer under FlashAttention.
    per_layer_act = int((10 * micro_batch_size * seq_len * n_embd + 2 * micro_batch_size * n_head * seq_len) * bytes_per_elem)
    if activation_checkpointing:
        # Store layer inputs (B * T * d * bytes) plus one active recomputed block.
        input_checkpoints = micro_batch_size * seq_len * n_embd * bytes_per_elem * n_layer
        activation_bytes = input_checkpoints + per_layer_act
    else:
        activation_bytes = per_layer_act * n_layer

    total_bytes = param_bytes + grad_bytes + optim_bytes + activation_bytes
    per_gpu_gb = round(total_bytes / BYTES_PER_GB, 3)
    return MemoryEstimate(
        strategy=strategy,
        world_size=world_size,
        param_bytes=param_bytes,
        grad_bytes=grad_bytes,
        optimizer_bytes=optim_bytes,
        activation_bytes=activation_bytes,
        total_bytes=total_bytes,
        per_gpu_gb=per_gpu_gb,
        vram_gb=vram_gb,
        fits_in_vram=per_gpu_gb <= vram_gb,
    )


def estimate_cluster_mfu(
    n_params: int,
    n_layer: int,
    seq_len: int,
    n_embd: int,
    tokens_per_sec: float,
    gpu_name: str,
    world_size: int = 1,
) -> float:
    """Compute Model FLOPs Utilization across ``world_size`` homogeneous GPUs."""
    peak = peak_flops(gpu_name) * world_size
    if peak <= 0 or tokens_per_sec <= 0:
        return 0.0
    return tokens_per_sec * flops_per_token(n_params, n_layer, seq_len, n_embd) / peak


@dataclass(frozen=True)
class StepProfile:
    """Per-step phase latency and throughput profile."""

    step: int
    data_ms: float
    forward_ms: float
    backward_ms: float
    optim_ms: float
    total_ms: float
    tokens: int
    tokens_per_sec: float
    mfu: float


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(p / 100.0 * len(ordered))) - 1))
    return ordered[idx]


class TrainingProfiler:
    """Lightweight CPU/GPU-safe phase profiler for distributed training loops."""

    def __init__(self, *, flops_per_tok: float = 0.0, peak_flops_total: float = 0.0, warmup_steps: int = 0) -> None:
        self.flops_per_tok = flops_per_tok
        self.peak_flops_total = peak_flops_total
        self.warmup_steps = warmup_steps
        self._current: dict[str, float] = {}
        self.records: list[StepProfile] = []

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._current[name] = self._current.get(name, 0.0) + elapsed_ms

    def finish_step(self, step: int, tokens: int) -> StepProfile:
        data_ms = self._current.get("data", 0.0)
        forward_ms = self._current.get("forward", 0.0)
        backward_ms = self._current.get("backward", 0.0)
        optim_ms = self._current.get("optim", 0.0)
        total_ms = max(data_ms + forward_ms + backward_ms + optim_ms, 1e-6)
        toks_per_sec = tokens / (total_ms / 1000.0)
        mfu = (toks_per_sec * self.flops_per_tok / self.peak_flops_total) if self.peak_flops_total > 0 else 0.0
        profile = StepProfile(
            step=step,
            data_ms=round(data_ms, 4),
            forward_ms=round(forward_ms, 4),
            backward_ms=round(backward_ms, 4),
            optim_ms=round(optim_ms, 4),
            total_ms=round(total_ms, 4),
            tokens=tokens,
            tokens_per_sec=round(toks_per_sec, 2),
            mfu=round(mfu, 4),
        )
        self._current = {}
        if len(self.records) >= self.warmup_steps or step >= self.warmup_steps:
            self.records.append(profile)
        return profile

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {"steps": 0, "mean_step_ms": 0.0, "p50_step_ms": 0.0, "p95_step_ms": 0.0, "mean_tokens_per_sec": 0.0, "mean_mfu": 0.0}
        totals = [r.total_ms for r in self.records]
        sum_total = sum(totals)
        return {
            "steps": len(self.records),
            "mean_step_ms": round(sum_total / len(self.records), 4),
            "p50_step_ms": round(_percentile(totals, 50), 4),
            "p95_step_ms": round(_percentile(totals, 95), 4),
            "mean_tokens_per_sec": round(sum(r.tokens_per_sec for r in self.records) / len(self.records), 2),
            "mean_mfu": round(sum(r.mfu for r in self.records) / len(self.records), 4),
            "phase_pct": {
                "data": round(100.0 * sum(r.data_ms for r in self.records) / sum_total, 2),
                "forward": round(100.0 * sum(r.forward_ms for r in self.records) / sum_total, 2),
                "backward": round(100.0 * sum(r.backward_ms for r in self.records) / sum_total, 2),
                "optim": round(100.0 * sum(r.optim_ms for r in self.records) / sum_total, 2),
            },
        }

    def export_json(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary(), "steps": [dataclasses.asdict(r) for r in self.records]}
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return dest
