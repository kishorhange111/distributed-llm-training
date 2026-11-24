"""Tests for training throughput, MFU, and GPU memory estimator and step profiler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from disttrain.metrics import flops_per_token, peak_flops
from disttrain.model import build_model
from disttrain.profiler import TrainingProfiler, estimate_cluster_mfu, estimate_training_memory


def test_memory_estimator_scales_across_strategies_and_offload() -> None:
    kwargs = dict(n_params=1_500_000_000, n_layer=48, n_embd=1600, n_head=25, seq_len=1024, micro_batch_size=4, world_size=8, vram_gb=40.0)
    ddp = estimate_training_memory(**kwargs, strategy="ddp")
    z2 = estimate_training_memory(**kwargs, strategy="deepspeed_zero2")
    z3 = estimate_training_memory(**kwargs, strategy="fsdp")
    z3_ckpt = estimate_training_memory(**kwargs, strategy="fsdp", activation_checkpointing=True)
    z3_offload = estimate_training_memory(**kwargs, strategy="fsdp", activation_checkpointing=True, cpu_offload=True)

    assert ddp.total_bytes > z2.total_bytes > z3.total_bytes > z3_ckpt.total_bytes > z3_offload.total_bytes
    assert z3_ckpt.activation_bytes < z3.activation_bytes
    assert z3_offload.optimizer_bytes == 0 and z3_offload.param_bytes == 0
    assert z3_ckpt.fits_in_vram is True


def test_estimate_cluster_mfu_matches_a100_math() -> None:
    mfu = estimate_cluster_mfu(
        n_params=124_000_000,
        n_layer=12,
        seq_len=1024,
        n_embd=768,
        tokens_per_sec=192_000,
        gpu_name="NVIDIA A100-SXM4-40GB",
        world_size=1,
    )
    assert 0.45 < mfu < 0.60
    assert estimate_cluster_mfu(124_000_000, 12, 1024, 768, 192_000, "unknown-cpu", 1) == 0.0


def test_training_profiler_records_phases_and_exports_json(tmp_path: Path) -> None:
    model = build_model("tiny")
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    fpt = flops_per_token(model.num_params(), model.cfg.n_layer, 16, model.cfg.n_embd)
    profiler = TrainingProfiler(flops_per_tok=fpt, peak_flops_total=peak_flops("A100"), warmup_steps=0)

    for step in range(3):
        with profiler.phase("data"):
            x = torch.randint(0, 256, (2, 17))
        with profiler.phase("forward"):
            _, loss = model(x[:, :-1], x[:, 1:])
        with profiler.phase("backward"):
            loss.backward()
        with profiler.phase("optim"):
            opt.step()
            opt.zero_grad(set_to_none=True)
        rec = profiler.finish_step(step=step, tokens=32)
        assert rec.total_ms > 0
        assert rec.tokens_per_sec > 0

    summary = profiler.summary()
    assert summary["steps"] == 3
    assert summary["p95_step_ms"] >= summary["p50_step_ms"] > 0
    assert sum(summary["phase_pct"].values()) == pytest.approx(100.0, abs=0.2)

    out = profiler.export_json(tmp_path / "profile.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["steps"] == 3
    assert len(data["steps"]) == 3
