"""Parallelism strategies: how the model is replicated or sharded across GPUs.

Memory per GPU for a model with P parameters trained with AdamW in mixed
precision (the ZeRO paper's accounting, 16 bytes/param):

    weights fp32 4P + gradients fp32 4P + Adam moments 8P   (+ activations)

| strategy                   | what is split across N GPUs          | memory / GPU (states) | communication per step |
|----------------------------|--------------------------------------|-----------------------|------------------------|
| ``single``                 | nothing                              | 16P                   | none                   |
| ``ddp``                    | the *data*; every GPU holds a full copy | 16P                | all-reduce grads (2P)  |
| ``fsdp`` (ZeRO-3)          | params, grads and optimizer state    | 16P / N               | all-gather params + reduce-scatter grads (3P) |
| ``fsdp`` + ``reshard_after_forward=False`` (ZeRO-2) | grads + optimizer state | between the two | (2P), but params stay gathered after forward |
| ``fsdp`` + CPU offload     | as FSDP, with shards kept in CPU RAM | ~activations only     | + PCIe copies          |

DDP is fastest when the model fits; FSDP is what lets 7B+ models train at all.
This module uses FSDP2 (``torch.distributed.fsdp.fully_shard``), the
per-parameter DTensor-based API that replaces the older ``FullyShardedDataParallel``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from disttrain.distributed import DistInfo
from disttrain.model import GPT

Strategy = Literal["single", "ddp", "fsdp"]


def wrap_model(
    model: GPT,
    strategy: Strategy,
    info: DistInfo,
    *,
    mixed_precision: bool = True,
    reshard_after_forward: bool = True,
    cpu_offload: bool = False,
) -> nn.Module:
    """Move ``model`` to the right device and wrap it for the chosen strategy.

    Args:
        model: An *unwrapped* GPT.
        strategy: ``"single"``, ``"ddp"`` or ``"fsdp"``.
        info: This process's rank and device.
        mixed_precision: For FSDP, compute in bf16 but keep fp32 master weights and
            reduce gradients in fp32. (single/DDP use ``torch.autocast`` in the training loop instead.)
        reshard_after_forward: FSDP only. True = ZeRO-3 (free gathered params after
            forward, re-gather in backward: least memory). False = ZeRO-2-like (keep them: less communication).
        cpu_offload: FSDP only. Keep parameter/gradient/optimizer shards in pinned CPU memory.
    """
    if strategy == "single":
        return model.to(info.device)

    if strategy == "ddp":
        model = model.to(info.device)
        # DDP broadcasts rank 0's weights at construction, so all replicas start identical.
        # gradient_as_bucket_view avoids one gradient copy (saves 4P bytes).
        # (static_graph=True is deliberately not used: it is incompatible with no_sync()
        # gradient accumulation, which the training loop relies on.)
        return DDP(
            model,
            device_ids=[info.local_rank] if info.device.type == "cuda" else None,
            gradient_as_bucket_view=True,
        )

    if strategy == "fsdp":
        from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if mixed_precision else MixedPrecisionPolicy()
        kwargs = {"mp_policy": mp, "reshard_after_forward": reshard_after_forward}
        if cpu_offload:
            kwargs["offload_policy"] = CPUOffloadPolicy()
        else:
            model = model.to(info.device)
        # Shard each transformer block as its own unit. During forward only one block's
        # full parameters are gathered at a time, which keeps peak memory ~ one layer.
        for block in model.blocks:
            fully_shard(block, **kwargs)
        # The root call shards what's left (embeddings, final norm; the head is tied to wte).
        fully_shard(model, **kwargs)
        return model

    raise ValueError(f"unknown strategy {strategy!r}")


def unwrap(model: nn.Module) -> GPT:
    """Return the underlying GPT (DDP wraps it in ``.module``; FSDP2 modifies it in place)."""
    return model.module if isinstance(model, DDP) else model  # type: ignore[return-value]


def clip_grad_norm(model: nn.Module, max_norm: float) -> float:
    """Clip the global gradient norm.

    With FSDP each rank only holds a shard of each gradient. FSDP2 gradients
    are DTensors, so ``torch.nn.utils.clip_grad_norm_`` computes the *global*
    norm across shards correctly. That's why it's safe to call the same function
    for all strategies.
    """
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    # A DTensor norm must be materialised before converting to a Python float.
    if hasattr(norm, "full_tensor"):
        norm = norm.full_tensor()
    return float(norm)
