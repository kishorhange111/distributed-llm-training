"""The training loop. One script for single-GPU, DDP and FSDP runs.

Launch examples::

    # one process (laptop, or one Colab GPU)
    python -m disttrain.train --config configs/gpt2_small_fineweb.yaml

    # 8 GPUs on one machine, sharded with FSDP
    torchrun --nproc_per_node=8 -m disttrain.train --config configs/gpt2_small_fineweb.yaml --strategy fsdp

    # any field can be overridden from the command line
    python -m disttrain.train --config configs/bench.yaml --micro_batch_size 16 --compile true

What one optimizer step does (global batch = micro_batch * grad_accum * world_size):

    for each of grad_accum micro-batches:
        forward + backward (gradients *accumulate* in .grad)
        - skip the gradient all-reduce on all but the last micro-batch (no_sync)
    clip global grad norm -> AdamW step -> zero grads -> LR schedule
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from disttrain import checkpoint
from disttrain.data import TokenWindows
from disttrain.distributed import DistInfo, all_reduce_mean, barrier, cleanup, seed_everything, setup
from disttrain.metrics import ThroughputMeter, flops_per_token, peak_flops
from disttrain.model import build_model
from disttrain.strategies import clip_grad_norm, wrap_model


@dataclass
class TrainConfig:
    """Every knob of a run. Values come from a YAML file and/or command-line flags."""

    # model
    preset: str = "gpt2-small"
    block_size: int = 1024
    activation_checkpointing: bool = False
    # parallelism
    strategy: str = "single"  # single | ddp | fsdp
    reshard_after_forward: bool = True  # FSDP: ZeRO-3 (True) vs ZeRO-2-like (False)
    cpu_offload: bool = False  # FSDP: keep shards in CPU RAM
    # data
    train_data: str = ""  # path to a uint16 .bin file; empty = synthetic random tokens (for benchmarks)
    val_data: str = ""
    # batch
    micro_batch_size: int = 16  # sequences per GPU per forward pass
    grad_accum: int = 1  # micro-batches per optimizer step
    # optimisation
    max_steps: int = 1000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # performance
    dtype: str = "bfloat16"  # bfloat16 | float32
    compile: bool = False
    # bookkeeping
    seed: int = 1337
    eval_interval: int = 250
    eval_batches: int = 20
    log_interval: int = 10
    out_dir: str = "runs/default"
    ckpt_interval: int = 0  # 0 disables checkpointing
    time_budget_minutes: float = 0  # stop cleanly (and checkpoint) after this long; 0 = no limit

    @property
    def tokens_per_step_per_rank(self) -> int:
        return self.micro_batch_size * self.grad_accum * self.block_size


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warm-up, then cosine decay from ``lr`` to ``min_lr`` at ``max_steps``.

    Warm-up avoids large, destabilising updates while Adam's moment estimates
    are still noisy. The cosine tail lets the model settle into a minimum.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only.

    Decaying biases and LayerNorm gains hurts quality, and embeddings/matmul
    weights (2-D tensors) are what actually need regularising. ``fused=True``
    runs the update as one CUDA kernel instead of one per parameter.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=device.type == "cuda")


def no_grad_sync(model: torch.nn.Module, strategy: str, enabled: bool):
    """Context that skips gradient synchronisation on non-final micro-batches.

    Without it, DDP/FSDP would all-reduce gradients after *every* micro-batch,
    multiplying communication by ``grad_accum`` for no benefit.
    """
    if not enabled:
        return contextlib.nullcontext()
    if strategy == "ddp":
        return model.no_sync()
    if strategy == "fsdp":
        model.set_requires_gradient_sync(False)
        return contextlib.nullcontext()
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: TokenWindows, cfg: TrainConfig, info: DistInfo, autocast) -> float:
    """Mean validation loss over a fixed set of batches, averaged across ranks."""
    model.eval()
    losses = []
    for x, y in data.sequential_batches(cfg.micro_batch_size, info.device, cfg.eval_batches):
        with autocast():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return all_reduce_mean(sum(losses) / max(len(losses), 1), info)


def train(cfg: TrainConfig) -> dict[str, Any]:
    """Run training; returns a summary (final losses, throughput, MFU, peak memory)."""
    info = setup()
    seed_everything(cfg.seed, info.rank)
    out = Path(cfg.out_dir)
    if info.is_main:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    # ---- model: seed identically on every rank so FSDP shards come from the same init
    torch.manual_seed(cfg.seed)
    model = build_model(cfg.preset, block_size=cfg.block_size, activation_checkpointing=cfg.activation_checkpointing)
    n_params = model.num_params()
    mcfg = model.cfg
    model = wrap_model(
        model,
        cfg.strategy,
        info,
        mixed_precision=cfg.dtype == "bfloat16",
        reshard_after_forward=cfg.reshard_after_forward,
        cpu_offload=cfg.cpu_offload,
    )
    # Keep a handle on the wrapped (DDP/FSDP) module: gradient-sync control, clipping and
    # checkpointing talk to it, while the forward pass goes through the compiled version.
    wrapped = model
    if cfg.compile:
        # Fuses elementwise ops and removes Python overhead: typically 1.3-2x faster on A100/H100.
        model = torch.compile(model)
    optimizer = build_optimizer(wrapped, cfg, info.device)

    # Under FSDP the MixedPrecisionPolicy already casts to bf16, so autocast is only for single/DDP.
    use_autocast = cfg.dtype == "bfloat16" and cfg.strategy != "fsdp"

    def autocast():
        return torch.autocast(info.device.type, dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()

    # ---- data: a real token file, or random tokens for pure-throughput benchmarks
    if cfg.train_data:
        train_data = TokenWindows(cfg.train_data, cfg.block_size, info.rank, info.world_size, cfg.seed)
        val_data = TokenWindows(cfg.val_data, cfg.block_size) if cfg.val_data else None
    else:
        train_data, val_data = None, None

    def next_batch():
        if train_data is not None:
            return train_data.batch(cfg.micro_batch_size, info.device)
        x = torch.randint(0, mcfg.vocab_size, (cfg.micro_batch_size, cfg.block_size + 1), device=info.device)
        return x[:, :-1], x[:, 1:]

    # ---- resume if a checkpoint exists
    start_step = 0
    if cfg.ckpt_interval and (ckpt := checkpoint.latest(out / "checkpoints")) is not None:
        start_step = checkpoint.load(ckpt, wrapped, optimizer) + 1
        if info.is_main:
            print(f"resumed from {ckpt} at step {start_step}", flush=True)

    gpu_name = torch.cuda.get_device_name(info.device) if info.device.type == "cuda" else "cpu"
    meter = ThroughputMeter(
        flops_per_token(n_params, mcfg.n_layer, cfg.block_size, mcfg.n_embd),
        peak_flops(gpu_name) * info.world_size,
    )
    if info.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(info.device)
    log_path = out / "log.jsonl"
    t0 = time.time()
    history: list[dict[str, Any]] = []
    val_loss = float("nan")
    loss_value = float("nan")

    step = start_step
    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ---- forward/backward over grad_accum micro-batches
        loss_sum = 0.0
        for micro in range(cfg.grad_accum):
            last = micro == cfg.grad_accum - 1
            if cfg.strategy == "fsdp" and last:
                wrapped.set_requires_gradient_sync(True)  # re-enable sync for the final micro-batch
            x, y = next_batch()
            with no_grad_sync(wrapped, cfg.strategy, enabled=not last), autocast():
                _, loss = model(x, y)
                # Divide so the accumulated gradient equals the gradient of the *mean* loss over the global batch.
                (loss / cfg.grad_accum).backward()
            loss_sum += loss.detach().float()

        grad_norm = clip_grad_norm(wrapped, cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # set_to_none frees gradient memory instead of zeroing it
        meter.step(cfg.tokens_per_step_per_rank * info.world_size)

        # ---- logging (rank 0 only, but the loss is averaged over all ranks)
        if step % cfg.log_interval == 0 or step == cfg.max_steps - 1:
            loss_value = all_reduce_mean(loss_sum / cfg.grad_accum, info)
            record = {
                "step": step,
                "loss": round(loss_value, 4),
                "lr": lr,
                "grad_norm": round(grad_norm, 3),
                "tokens_per_sec": round(meter.tokens_per_sec),
                "mfu": round(meter.mfu, 4),
                "elapsed_min": round((time.time() - t0) / 60, 2),
            }
            if info.device.type == "cuda":
                record["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(info.device) / 2**30, 2)
            if val_data is not None and (step % cfg.eval_interval == 0 or step == cfg.max_steps - 1):
                val_loss = evaluate(model, val_data, cfg, info, autocast)
                record["val_loss"] = round(val_loss, 4)
            history.append(record)
            if info.is_main:
                print(json.dumps(record), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

        # ---- checkpointing and time budget
        out_of_time = cfg.time_budget_minutes and (time.time() - t0) / 60 > cfg.time_budget_minutes
        if cfg.ckpt_interval and (step % cfg.ckpt_interval == 0 or out_of_time or step == cfg.max_steps - 1) and step > start_step:
            checkpoint.save(out / "checkpoints", wrapped, optimizer, step, {"preset": cfg.preset}, info.is_main)
        if out_of_time:
            if info.is_main:
                print(f"time budget reached at step {step}", flush=True)
            break

    summary = {
        "strategy": cfg.strategy,
        "world_size": info.world_size,
        "params_non_embedding": n_params,
        "final_step": step,
        "final_loss": loss_value,
        "final_val_loss": val_loss,
        "tokens_per_sec": meter.tokens_per_sec,
        "mfu": meter.mfu,
        "gpu": gpu_name,
        "peak_mem_gb": torch.cuda.max_memory_allocated(info.device) / 2**30 if info.device.type == "cuda" else None,
        "minutes": (time.time() - t0) / 60,
    }
    if info.is_main:
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    barrier(info)
    cleanup()
    return summary


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Build a TrainConfig from ``--config file.yaml`` plus ``--field value`` overrides."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML file with TrainConfig fields")
    # One flag per dataclass field, so every knob is overridable without editing YAML.
    # (Field types are strings because of `from __future__ import annotations`.)
    converters = {"bool": lambda s: s.lower() in ("1", "true", "yes"), "int": int, "float": float, "str": str}
    for f in dataclasses.fields(TrainConfig):
        parser.add_argument(f"--{f.name}", type=converters[str(f.type)], default=None)
    args = parser.parse_args(argv)
    values: dict[str, Any] = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    values.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    return TrainConfig(**values)


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
