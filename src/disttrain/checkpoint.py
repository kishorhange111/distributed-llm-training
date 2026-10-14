"""Sharded, resumable checkpoints with ``torch.distributed.checkpoint`` (DCP).

With FSDP no single rank holds the whole model, so "gather everything on
rank 0 and torch.save it" would run rank 0 out of memory for large models.
DCP instead has every rank write *its own shards* in parallel, plus metadata
describing the layout. Loading can then re-shard to a *different* world size:
save on 8 GPUs, resume on 4.

The same code path works for single-GPU and DDP models (a world of one shard).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.optim import Optimizer

from disttrain.strategies import unwrap


def save(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    extra: dict[str, Any] | None = None,
    is_main: bool = True,
    keep_last: int = 2,
) -> None:
    """Write model + optimizer state for ``step`` into ``path/step_<step>/``.

    Only the newest ``keep_last`` checkpoints are kept (0 keeps all), so frequent
    checkpointing on a preemptible machine doesn't fill the disk. The ``latest``
    marker is written only after the save completes, so a crash mid-save
    leaves the previous checkpoint as the one to resume from.

    ``get_state_dict`` returns the right flavour of state dict for any wrapper
    (plain, DDP or FSDP2 sharded DTensors), so the caller doesn't need to know
    which strategy is in use.
    """
    model_sd, optim_sd = get_state_dict(unwrap(model), optimizer)
    target = Path(path) / f"step_{step:07d}"
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=str(target))
    if is_main:
        # Small human-readable sidecar file: step and config, so a run can be identified without loading it.
        (target / "meta.json").write_text(json.dumps({"step": step, **(extra or {})}, indent=2))
        (Path(path) / "latest").write_text(target.name)
        if keep_last:
            for old in sorted(Path(path).glob("step_*"))[:-keep_last]:
                shutil.rmtree(old, ignore_errors=True)


def latest(path: str | Path) -> Path | None:
    """Most recent checkpoint folder under ``path``, or None if there isn't one."""
    marker = Path(path) / "latest"
    return Path(path) / marker.read_text().strip() if marker.exists() else None


def load(checkpoint_dir: str | Path, model: nn.Module, optimizer: Optimizer) -> int:
    """Restore model + optimizer in place from ``checkpoint_dir``; returns the saved step.

    DCP loads *into* the existing (possibly sharded) tensors, so the model must
    already be wrapped with the strategy it will train with.
    """
    model_sd, optim_sd = get_state_dict(unwrap(model), optimizer)
    state = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=str(checkpoint_dir))
    set_state_dict(unwrap(model), optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"])
    return json.loads((Path(checkpoint_dir) / "meta.json").read_text())["step"]
