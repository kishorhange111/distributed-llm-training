"""Modular YAML configuration schemas and validator for DDP, FSDP2, and DeepSpeed ZeRO-2/ZeRO-3."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from disttrain.model import PRESETS
from disttrain.train import TrainConfig

VALID_STRATEGIES = frozenset({"single", "ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"})
VALID_DTYPES = frozenset({"bfloat16", "float16", "float32"})


class ConfigValidationError(ValueError):
    """Raised when a DistributedJobConfig fails schema or invariant checks."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Invalid distributed training configuration:\n- " + "\n- ".join(errors))


@dataclass
class ModelSchema:
    preset: str = "gpt2-small"
    block_size: int = 1024
    vocab_size: int | None = None
    n_layer: int | None = None
    n_head: int | None = None
    n_embd: int | None = None
    activation_checkpointing: bool = False


@dataclass
class ParallelSchema:
    strategy: str = "ddp"
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    reshard_after_forward: bool = True
    cpu_offload: bool = False
    zero_stage: int = 0
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: int = 200_000_000
    allgather_bucket_size: int = 200_000_000


@dataclass
class OptimSchema:
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    max_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


@dataclass
class PrecisionSchema:
    dtype: str = "bfloat16"
    grad_scaler: bool = False
    compile: bool = False


@dataclass
class DistributedJobConfig:
    """Composable top-level job config for DDP, FSDP2, and DeepSpeed ZeRO runs."""

    name: str = "default-job"
    model: ModelSchema = field(default_factory=ModelSchema)
    parallel: ParallelSchema = field(default_factory=ParallelSchema)
    optim: OptimSchema = field(default_factory=OptimSchema)
    precision: PrecisionSchema = field(default_factory=PrecisionSchema)
    train_data: str = ""
    val_data: str = ""
    out_dir: str = "runs/default"
    seed: int = 1337
    ckpt_interval: int = 100
    eval_interval: int = 250
    log_interval: int = 10

    def global_tokens_per_step(self, world_size: int = 1) -> int:
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        return self.parallel.micro_batch_size * self.parallel.gradient_accumulation_steps * self.model.block_size * world_size


def validate_config(cfg: DistributedJobConfig) -> list[str]:
    """Return a list of schema and invariant violations (empty list means valid)."""
    errors: list[str] = []

    if cfg.model.preset not in PRESETS and cfg.model.preset != "custom":
        errors.append(f"unknown model.preset {cfg.model.preset!r}; expected one of {sorted(PRESETS)} or 'custom'")
    if cfg.model.block_size <= 0:
        errors.append("model.block_size must be positive")
    if cfg.model.preset == "custom":
        for field_name in ("vocab_size", "n_layer", "n_head", "n_embd"):
            val = getattr(cfg.model, field_name)
            if val is None or val <= 0:
                errors.append(f"model.{field_name} must be a positive integer when preset='custom'")
    if cfg.model.n_embd is not None and cfg.model.n_head is not None and cfg.model.n_head > 0 and cfg.model.n_embd % cfg.model.n_head != 0:
        errors.append("model.n_embd must be divisible by model.n_head")

    if cfg.parallel.strategy not in VALID_STRATEGIES:
        errors.append(f"parallel.strategy {cfg.parallel.strategy!r} not in {sorted(VALID_STRATEGIES)}")
    if cfg.parallel.micro_batch_size < 1:
        errors.append("parallel.micro_batch_size must be >= 1")
    if cfg.parallel.gradient_accumulation_steps < 1:
        errors.append("parallel.gradient_accumulation_steps must be >= 1")
    if cfg.parallel.strategy == "deepspeed_zero2" and cfg.parallel.zero_stage != 2:
        errors.append("parallel.zero_stage must be 2 for strategy='deepspeed_zero2'")
    if cfg.parallel.strategy == "deepspeed_zero3" and cfg.parallel.zero_stage != 3:
        errors.append("parallel.zero_stage must be 3 for strategy='deepspeed_zero3'")
    if cfg.parallel.strategy in ("single", "ddp") and cfg.parallel.cpu_offload:
        errors.append(f"parallel.cpu_offload is not supported for strategy={cfg.parallel.strategy!r}")
    if cfg.parallel.strategy in ("single", "ddp") and cfg.parallel.zero_stage != 0:
        errors.append(f"parallel.zero_stage must be 0 for strategy={cfg.parallel.strategy!r}")

    if cfg.optim.lr <= 0:
        errors.append("optim.lr must be > 0")
    if cfg.optim.min_lr < 0 or cfg.optim.min_lr > cfg.optim.lr:
        errors.append("optim.min_lr must satisfy 0 <= min_lr <= lr")
    if cfg.optim.max_steps < 1:
        errors.append("optim.max_steps must be >= 1")
    if cfg.optim.warmup_steps < 0 or cfg.optim.warmup_steps > cfg.optim.max_steps:
        errors.append("optim.warmup_steps must satisfy 0 <= warmup_steps <= max_steps")
    if not (0.0 < cfg.optim.beta1 < 1.0):
        errors.append("optim.beta1 must be in (0, 1)")
    if not (0.0 < cfg.optim.beta2 < 1.0):
        errors.append("optim.beta2 must be in (0, 1)")
    if cfg.optim.grad_clip <= 0:
        errors.append("optim.grad_clip must be > 0")
    if cfg.optim.weight_decay < 0:
        errors.append("optim.weight_decay must be >= 0")

    if cfg.precision.dtype not in VALID_DTYPES:
        errors.append(f"precision.dtype {cfg.precision.dtype!r} not in {sorted(VALID_DTYPES)}")
    if cfg.precision.grad_scaler and cfg.precision.dtype != "float16":
        errors.append("precision.grad_scaler should only be enabled with precision.dtype='float16'")

    return errors


def validate_config_or_raise(cfg: DistributedJobConfig) -> DistributedJobConfig:
    """Validate ``cfg`` and raise :class:`ConfigValidationError` on any violation."""
    errors = validate_config(cfg)
    if errors:
        raise ConfigValidationError(errors)
    return cfg


def load_job_config(path: str | Path) -> DistributedJobConfig:
    """Load and validate a :class:`DistributedJobConfig` from a YAML file."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = DistributedJobConfig(
        name=raw.get("name", "default-job"),
        model=ModelSchema(**(raw.get("model") or {})),
        parallel=ParallelSchema(**(raw.get("parallel") or {})),
        optim=OptimSchema(**(raw.get("optim") or {})),
        precision=PrecisionSchema(**(raw.get("precision") or {})),
        train_data=raw.get("train_data", ""),
        val_data=raw.get("val_data", ""),
        out_dir=raw.get("out_dir", "runs/default"),
        seed=raw.get("seed", 1337),
        ckpt_interval=raw.get("ckpt_interval", 100),
        eval_interval=raw.get("eval_interval", 250),
        log_interval=raw.get("log_interval", 10),
    )
    return validate_config_or_raise(cfg)


def dump_job_config(cfg: DistributedJobConfig, path: str | Path) -> Path:
    """Serialize a validated :class:`DistributedJobConfig` to YAML."""
    validate_config_or_raise(cfg)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False), encoding="utf-8")
    return dest


def to_deepspeed_json(cfg: DistributedJobConfig) -> dict[str, Any]:
    """Generate a DeepSpeed ZeRO-2 or ZeRO-3 JSON config dictionary from ``cfg``."""
    validate_config_or_raise(cfg)
    if cfg.parallel.strategy not in ("deepspeed_zero2", "deepspeed_zero3"):
        raise ValueError(f"to_deepspeed_json requires a DeepSpeed strategy, got {cfg.parallel.strategy!r}")

    zero_cfg: dict[str, Any] = {
        "stage": cfg.parallel.zero_stage,
        "overlap_comm": cfg.parallel.overlap_comm,
        "contiguous_gradients": cfg.parallel.contiguous_gradients,
        "reduce_bucket_size": cfg.parallel.reduce_bucket_size,
        "allgather_bucket_size": cfg.parallel.allgather_bucket_size,
    }
    if cfg.parallel.cpu_offload:
        zero_cfg["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        if cfg.parallel.zero_stage == 3:
            zero_cfg["offload_param"] = {"device": "cpu", "pin_memory": True}
    if cfg.parallel.zero_stage == 3:
        zero_cfg["stage3_gather_16bit_weights_on_model_save"] = True

    ds_cfg: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": cfg.parallel.micro_batch_size,
        "gradient_accumulation_steps": cfg.parallel.gradient_accumulation_steps,
        "gradient_clipping": cfg.optim.grad_clip,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": cfg.optim.lr,
                "betas": [cfg.optim.beta1, cfg.optim.beta2],
                "weight_decay": cfg.optim.weight_decay,
            },
        },
        "zero_optimization": zero_cfg,
    }
    if cfg.precision.dtype == "bfloat16":
        ds_cfg["bf16"] = {"enabled": True}
    elif cfg.precision.dtype == "float16":
        ds_cfg["fp16"] = {"enabled": True, "auto_cast": True}
    return ds_cfg


def to_train_config(cfg: DistributedJobConfig) -> TrainConfig:
    """Bridge a :class:`DistributedJobConfig` to the native PyTorch :class:`TrainConfig`."""
    validate_config_or_raise(cfg)
    strategy_map = {
        "single": "single",
        "ddp": "ddp",
        "fsdp": "fsdp",
        "deepspeed_zero2": "fsdp",
        "deepspeed_zero3": "fsdp",
    }
    reshard = cfg.parallel.reshard_after_forward
    if cfg.parallel.strategy == "deepspeed_zero2":
        reshard = False
    elif cfg.parallel.strategy == "deepspeed_zero3":
        reshard = True
    return TrainConfig(
        preset=cfg.model.preset,
        block_size=cfg.model.block_size,
        activation_checkpointing=cfg.model.activation_checkpointing,
        strategy=strategy_map[cfg.parallel.strategy],
        reshard_after_forward=reshard,
        cpu_offload=cfg.parallel.cpu_offload,
        train_data=cfg.train_data,
        val_data=cfg.val_data,
        micro_batch_size=cfg.parallel.micro_batch_size,
        grad_accum=cfg.parallel.gradient_accumulation_steps,
        max_steps=cfg.optim.max_steps,
        lr=cfg.optim.lr,
        min_lr=cfg.optim.min_lr,
        warmup_steps=cfg.optim.warmup_steps,
        weight_decay=cfg.optim.weight_decay,
        beta1=cfg.optim.beta1,
        beta2=cfg.optim.beta2,
        grad_clip=cfg.optim.grad_clip,
        dtype="float32" if cfg.precision.dtype == "float32" else "bfloat16",
        compile=cfg.precision.compile,
        seed=cfg.seed,
        eval_interval=cfg.eval_interval,
        log_interval=cfg.log_interval,
        out_dir=cfg.out_dir,
        ckpt_interval=cfg.ckpt_interval,
    )
