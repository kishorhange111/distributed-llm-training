"""Process-group setup and small collective helpers.

How a distributed run is launched: ``torchrun --nproc_per_node=N train.py`` starts
N copies of the script and sets environment variables in each one:

* ``WORLD_SIZE``: total number of processes (GPUs) in the job
* ``RANK``: this process's global id, 0 .. WORLD_SIZE-1
* ``LOCAL_RANK``: id within this machine, used to pick the GPU (``cuda:LOCAL_RANK``)
* ``MASTER_ADDR`` / ``MASTER_PORT``: where rank 0 listens so the others can rendezvous

Without these variables the code runs as an ordinary single process, so the
same script works on a laptop, on one Colab GPU, and on a multi-node cluster.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    """Where this process sits in the job."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        """Rank 0 does the logging and checkpoint bookkeeping, so output isn't duplicated N times."""
        return self.rank == 0


def setup(backend: str | None = None) -> DistInfo:
    """Initialise ``torch.distributed`` if launched by torchrun; otherwise run single-process.

    Args:
        backend: ``"nccl"`` for GPUs (the fast, GPU-to-GPU path), ``"gloo"`` for CPU.
            Chosen automatically when None.
    """
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistInfo(0, 0, 1, device)

    rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    use_cuda = torch.cuda.is_available() and backend != "gloo"
    backend = backend or ("nccl" if use_cuda else "gloo")
    if use_cuda:
        # Must happen *before* NCCL initialisation so each rank owns exactly one GPU.
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    return DistInfo(rank, local_rank, world, device)


def cleanup() -> None:
    """Tear down the process group (avoids hangs and warnings at exit)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed all RNGs. Each rank gets ``seed + rank`` so data sampling and dropout differ across ranks.

    Model *initialisation* must be identical on every rank. The training script
    re-seeds with the *same* seed just before building the model (and re-applies
    the per-rank seed afterwards). DDP additionally broadcasts rank 0's weights;
    FSDP2's ``fully_shard`` does not broadcast, each rank keeps its slice of the
    weights it initialised itself, so equal seeds are what make the shards agree.
    """
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)


def all_reduce_mean(value: float | torch.Tensor, info: DistInfo) -> float:
    """Average a scalar over all ranks, e.g. so the logged loss is the global loss, not rank 0's."""
    t = torch.as_tensor(value, dtype=torch.float32, device=info.device).clone()
    if info.world_size > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= info.world_size
    return float(t)


def stop_decision(local: bool, info: DistInfo) -> bool:
    """Rank 0's yes/no, broadcast to all ranks, so every rank leaves the training loop at the same step."""
    if info.world_size == 1:
        return local
    t = torch.tensor([1 if local else 0], device=info.device)
    dist.broadcast(t, src=0)
    return bool(t.item())


def barrier(info: DistInfo) -> None:
    """Wait until every rank reaches this point (no-op when single-process)."""
    if info.world_size > 1:
        dist.barrier()
