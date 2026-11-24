# ADR 001: Parallelism Strategy Selection (DDP, FSDP2, DeepSpeed ZeRO-2/ZeRO-3)

## Status
Accepted

## Context
Large-language-model training jobs span single-GPU prototyping (Colab L4/A100), multi-GPU node runs (DDP), and multi-node sharded runs (FSDP2 / DeepSpeed ZeRO-2 & ZeRO-3). Misconfigured parallelism settings (e.g., ZeRO stage mismatches, CPU offload under DDP, or gradient accumulation without suppressed all-reduces) cause silent throughput degradation or OOM crashes hours into a run.

## Decision
1. **Declarative YAML schemas (`disttrain.config`)**: Separate `ModelSchema`, `ParallelSchema`, `OptimSchema`, and `PrecisionSchema` into validated dataclasses with `validate_config_or_raise()` invoked before job launch.
2. **Unified PyTorch FSDP2 & DeepSpeed config bridge**: Support native PyTorch FSDP2 (`fully_shard` per `Block`) while exporting validated DeepSpeed ZeRO-2 / ZeRO-3 JSON configs via `to_deepspeed_json()` and mapping ZeRO-2/ZeRO-3 semantics (`reshard_after_forward=False/True`) cleanly onto `TrainConfig`.
3. **Analytical memory estimation (`disttrain.profiler.estimate_training_memory`)**: Estimate per-GPU parameter, gradient, optimizer, and activation memory prior to launching GPU jobs so OOM configurations are caught locally on CPU.

## Consequences
- Invalid strategy/precision/offload combinations fail fast at config parse time.
- Engineers can switch between DDP, FSDP2, and DeepSpeed ZeRO-2/ZeRO-3 without rewriting training scripts.
