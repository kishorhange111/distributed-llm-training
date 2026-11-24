"""Tests for modular distributed YAML configuration schemas and validator."""

from __future__ import annotations

from pathlib import Path

import pytest

from disttrain.config import (
    ConfigValidationError,
    DistributedJobConfig,
    ModelSchema,
    OptimSchema,
    ParallelSchema,
    PrecisionSchema,
    dump_job_config,
    load_job_config,
    to_deepspeed_json,
    to_train_config,
    validate_config,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize(
    "filename",
    ["ddp_gpt2_medium.yaml", "fsdp_zero3_gpt2_xl.yaml", "deepspeed_zero2.yaml", "deepspeed_zero3.yaml"],
)
def test_builtin_distributed_yaml_configs_are_valid(filename: str) -> None:
    cfg = load_job_config(CONFIG_DIR / filename)
    assert validate_config(cfg) == []
    assert cfg.global_tokens_per_step(world_size=4) > 0


def test_config_roundtrip_yaml(tmp_path: Path) -> None:
    cfg = DistributedJobConfig(
        name="custom-fsdp",
        model=ModelSchema(preset="gpt2-small", block_size=512, activation_checkpointing=True),
        parallel=ParallelSchema(strategy="fsdp", micro_batch_size=8, gradient_accumulation_steps=4, zero_stage=3),
        optim=OptimSchema(lr=3e-4, min_lr=3e-5, warmup_steps=50, max_steps=500),
        precision=PrecisionSchema(dtype="bfloat16", compile=True),
    )
    out = dump_job_config(cfg, tmp_path / "job.yaml")
    loaded = load_job_config(out)
    assert loaded == cfg


def test_validation_catches_inconsistent_strategies_and_hyperparams() -> None:
    bad = DistributedJobConfig(
        model=ModelSchema(preset="custom", block_size=0, n_embd=100, n_head=3),
        parallel=ParallelSchema(strategy="deepspeed_zero3", zero_stage=2, micro_batch_size=0),
        optim=OptimSchema(lr=1e-4, min_lr=2e-4, warmup_steps=200, max_steps=100, beta1=1.2, grad_clip=-1.0),
        precision=PrecisionSchema(dtype="bfloat16", grad_scaler=True),
    )
    errors = validate_config(bad)
    assert any("block_size" in e for e in errors)
    assert any("divisible" in e for e in errors)
    assert any("zero_stage must be 3" in e for e in errors)
    assert any("min_lr" in e for e in errors)
    assert any("warmup_steps" in e for e in errors)
    assert any("grad_scaler" in e for e in errors)


def test_ddp_rejects_cpu_offload_and_zero_stage() -> None:
    bad = DistributedJobConfig(parallel=ParallelSchema(strategy="ddp", cpu_offload=True, zero_stage=2))
    errors = validate_config(bad)
    assert any("cpu_offload" in e for e in errors)
    assert any("zero_stage must be 0" in e for e in errors)


def test_load_raises_on_invalid_yaml(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("parallel:\n  strategy: invalid_backend\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_job_config(bad_file)


def test_deepspeed_json_generation_zero2_and_zero3() -> None:
    z2 = load_job_config(CONFIG_DIR / "deepspeed_zero2.yaml")
    ds2 = to_deepspeed_json(z2)
    assert ds2["zero_optimization"]["stage"] == 2
    assert ds2["bf16"]["enabled"] is True
    assert "offload_param" not in ds2["zero_optimization"]

    z3 = load_job_config(CONFIG_DIR / "deepspeed_zero3.yaml")
    ds3 = to_deepspeed_json(z3)
    assert ds3["zero_optimization"]["stage"] == 3
    assert ds3["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert ds3["zero_optimization"]["offload_param"]["device"] == "cpu"
    assert ds3["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] is True


def test_to_train_config_maps_strategies_correctly() -> None:
    z2 = load_job_config(CONFIG_DIR / "deepspeed_zero2.yaml")
    tc2 = to_train_config(z2)
    assert tc2.strategy == "fsdp"
    assert tc2.reshard_after_forward is False

    z3 = load_job_config(CONFIG_DIR / "deepspeed_zero3.yaml")
    tc3 = to_train_config(z3)
    assert tc3.strategy == "fsdp"
    assert tc3.reshard_after_forward is True
    assert tc3.cpu_offload is True
