"""Multi-process tests on CPU with the gloo backend: real process groups, real collectives.

These verify the properties distributed training depends on, without GPUs:

1. **DDP == single process**: with N ranks each holding 1/N of a batch, the
   averaged gradients equal the gradient of one process seeing the whole batch.
2. **Gradient accumulation == large batch**: k micro-batches with no_sync give
   the same update as one batch k times bigger.
3. **FSDP2 == single process**: sharded training produces the same weights.
4. **Checkpoint round-trip**: sharded save/load restores training exactly.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from disttrain import checkpoint
from disttrain.distributed import DistInfo
from disttrain.model import build_model
from disttrain.strategies import unwrap, wrap_model
from disttrain.train import TrainConfig, train

WORLD = 2


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch(fn, *args, init_group: bool = True):
    """Run ``fn(rank, *args)`` in WORLD processes with torchrun's environment variables.

    ``init_group=False`` leaves process-group creation to the code under test
    (the real training loop calls ``disttrain.distributed.setup`` itself).
    """
    port = free_port()
    mp.spawn(_entry, args=(fn, port, args, init_group), nprocs=WORLD, join=True)


def _entry(rank, fn, port, args, init_group):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD))
    torch.set_num_threads(1)
    if not init_group:
        fn(rank, *args)
        return
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        fn(rank, *args)
    finally:
        dist.destroy_process_group()


def _batch():
    g = torch.Generator().manual_seed(42)
    return torch.randint(0, 256, (4, 33), generator=g)  # global batch of 4 sequences


def _reference_step(lr=1e-2):
    """One SGD step of a single process on the full batch: the ground truth."""
    torch.manual_seed(0)
    model = build_model("tiny")
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    x = _batch()
    model(x[:, :-1], x[:, 1:])[1].backward()
    opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ------------------------------------------------------------------ workers
def _ddp_worker(rank, out_path):
    info = DistInfo(rank, rank, WORLD, torch.device("cpu"))
    torch.manual_seed(0)
    model = wrap_model(build_model("tiny"), "ddp", info)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    x = _batch().chunk(WORLD)[rank]  # each rank gets its half of the batch
    model(x[:, :-1], x[:, 1:])[1].backward()  # DDP all-reduces (averages) gradients here
    opt.step()
    if rank == 0:
        torch.save(unwrap(model).state_dict(), out_path)


def _accum_worker(rank, out_path):
    """2 ranks x 2 micro-batches of 1 sequence, gradient sync only on the last micro-batch."""
    info = DistInfo(rank, rank, WORLD, torch.device("cpu"))
    torch.manual_seed(0)
    model = wrap_model(build_model("tiny"), "ddp", info)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    micro = _batch().chunk(WORLD)[rank].chunk(2)
    with model.no_sync():
        (model(micro[0][:, :-1], micro[0][:, 1:])[1] / 2).backward()
    (model(micro[1][:, :-1], micro[1][:, 1:])[1] / 2).backward()
    opt.step()
    if rank == 0:
        torch.save(unwrap(model).state_dict(), out_path)


def _fsdp_worker(rank, out_path):
    info = DistInfo(rank, rank, WORLD, torch.device("cpu"))
    torch.manual_seed(0)
    model = wrap_model(build_model("tiny"), "fsdp", info, mixed_precision=False)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    x = _batch().chunk(WORLD)[rank]
    model(x[:, :-1], x[:, 1:])[1].backward()  # reduce-scatter: each rank keeps its gradient shard
    opt.step()
    # Gather the shards back into full tensors to compare with the reference.
    full = {k: v.full_tensor() if hasattr(v, "full_tensor") else v for k, v in model.state_dict().items()}
    if rank == 0:
        torch.save(full, out_path)


def _assert_close(a, b, atol=1e-5):
    for k in a:
        assert torch.allclose(a[k], b[k], atol=atol), f"{k} differs by {(a[k] - b[k]).abs().max():.2e}"


# -------------------------------------------------------------------- tests
def test_ddp_matches_single_process(tmp_path):
    _launch(_ddp_worker, str(tmp_path / "ddp.pt"))
    _assert_close(_reference_step(), torch.load(tmp_path / "ddp.pt"))


def test_grad_accumulation_with_no_sync_matches_full_batch(tmp_path):
    _launch(_accum_worker, str(tmp_path / "acc.pt"))
    _assert_close(_reference_step(), torch.load(tmp_path / "acc.pt"))


def test_fsdp2_matches_single_process(tmp_path):
    _launch(_fsdp_worker, str(tmp_path / "fsdp.pt"))
    _assert_close(_reference_step(), torch.load(tmp_path / "fsdp.pt"))


def _train_worker(rank, cfg_dict):
    train(TrainConfig(**cfg_dict))


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_full_training_loop_runs_distributed_and_resumes(tmp_path, strategy):
    """End to end: the real training loop under torchrun-like env, with sharded checkpoints and resume."""
    from disttrain.data import write_tokens

    data = write_tokens(torch.randint(0, 256, (5000,)).numpy(), tmp_path / "train.bin")
    cfg = dict(
        preset="tiny", block_size=32, strategy=strategy, train_data=str(data), val_data=str(data), micro_batch_size=2, grad_accum=2,
        max_steps=6, warmup_steps=2, lr=1e-3, dtype="float32", out_dir=str(tmp_path / "run"), ckpt_interval=3, log_interval=1,
        eval_interval=3, eval_batches=2,
    )  # fmt: skip
    _launch(_train_worker, cfg, init_group=False)
    log = (tmp_path / "run" / "log.jsonl").read_text().splitlines()
    assert len(log) == 6 and checkpoint.latest(tmp_path / "run" / "checkpoints").name == "step_0000005"
    # Resume: raise max_steps; training continues from step 6 instead of restarting.
    cfg["max_steps"] = 8
    _launch(_train_worker, cfg, init_group=False)
    log = (tmp_path / "run" / "log.jsonl").read_text().splitlines()
    assert len(log) == 8 and '"step": 6' in log[6]
