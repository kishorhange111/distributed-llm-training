"""Tests for atomic checkpoint manager, resume, integrity verification, and retention."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from disttrain.checkpoint_manager import CheckpointManager
from disttrain.model import build_model


def test_atomic_save_and_exact_resume(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_model("tiny")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (2, 16))
    _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()
    opt.step()

    mgr = CheckpointManager(tmp_path / "ckpts", keep_last=2)
    saved_dir = mgr.save(model, opt, step=10, strategy="fsdp", rank=0, world_size=4, metrics={"val_loss": 3.42})
    assert saved_dir.name == "step_0000010"
    assert mgr.latest() == saved_dir
    assert mgr.verify_integrity(saved_dir)

    # Restore into a freshly initialized model and verify exact weight equality.
    restored_model = build_model("tiny")
    restored_opt = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    meta = mgr.load(restored_model, restored_opt)
    assert meta.step == 10
    assert meta.strategy == "fsdp"
    assert meta.world_size == 4
    assert meta.metrics["val_loss"] == pytest.approx(3.42)
    assert len(meta.shards) == 1 and meta.shards[0].total_bytes > 0
    for p1, p2 in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(p1, p2)


def test_integrity_check_detects_corrupted_shard(tmp_path: Path) -> None:
    model = build_model("tiny")
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    mgr = CheckpointManager(tmp_path / "ckpts")
    ckpt_dir = mgr.save(model, opt, step=1)

    shard = ckpt_dir / "shard_r0000.pt"
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    assert mgr.verify_integrity(ckpt_dir) is False
    with pytest.raises(ValueError, match="integrity check failed"):
        mgr.load(model, opt)


def test_cleanup_incomplete_ignores_and_removes_staging_dirs(tmp_path: Path) -> None:
    model = build_model("tiny")
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    mgr = CheckpointManager(tmp_path / "ckpts")
    good = mgr.save(model, opt, step=5)

    orphan = tmp_path / "ckpts" / "step_0000006.tmp.99999"
    orphan.mkdir()
    (orphan / "partial.pt").write_bytes(b"incomplete")

    assert mgr.latest() == good
    removed = mgr.cleanup_incomplete()
    assert orphan in removed
    assert not orphan.exists()


def test_retention_keeps_last_n_and_best_metric(tmp_path: Path) -> None:
    model = build_model("tiny")
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    mgr = CheckpointManager(tmp_path / "ckpts", keep_last=2, keep_best_metric="val_loss", lower_is_better=True)

    mgr.save(model, opt, step=10, metrics={"val_loss": 2.5})  # best!
    mgr.save(model, opt, step=20, metrics={"val_loss": 3.1})
    mgr.save(model, opt, step=30, metrics={"val_loss": 2.9})
    mgr.save(model, opt, step=40, metrics={"val_loss": 2.8})

    steps = [m.step for m in mgr.list_checkpoints()]
    # step 10 preserved as best; steps 30 & 40 preserved as keep_last=2; step 20 pruned.
    assert steps == [10, 30, 40]
