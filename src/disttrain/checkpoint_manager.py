"""Atomic checkpoint manager with crash-safe commit, resume, retention, and sharded metadata.

Complements the low-level DCP helpers in :mod:`disttrain.checkpoint` with:

* **Two-phase atomic commit**: writes into a temporary staging directory, computes a SHA-256
  digest, renames to the final ``step_<N>`` directory, and atomically replaces the ``latest``
  pointer so preemptions never corrupt the last good checkpoint.
* **Sharded state metadata**: records strategy, world size, per-shard byte counts, and metrics
  in ``meta.json`` for offline inspection and re-sharding audits.
* **Best + recency retention**: keeps the newest ``keep_last`` checkpoints while protecting the
  best checkpoint according to a tracked validation metric.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from disttrain.strategies import unwrap


@dataclass(frozen=True)
class ShardMetadata:
    """Metadata describing a single rank's checkpoint shard."""

    rank: int
    world_size: int
    num_tensors: int
    total_bytes: int
    sha256: str


@dataclass(frozen=True)
class CheckpointMetadata:
    """Structured sidecar metadata stored in ``meta.json``."""

    step: int
    strategy: str
    world_size: int
    created_at_utc: str
    metrics: dict[str, float] = field(default_factory=dict)
    shards: list[ShardMetadata] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        shards = [ShardMetadata(**s) for s in data.get("shards", [])]
        return cls(
            step=int(data["step"]),
            strategy=str(data.get("strategy", "single")),
            world_size=int(data.get("world_size", 1)),
            created_at_utc=str(data.get("created_at_utc", "")),
            metrics={k: float(v) for k, v in (data.get("metrics") or {}).items()},
            shards=shards,
            extra=dict(data.get("extra") or {}),
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _tensor_stats(state_dict: dict[str, Any]) -> tuple[int, int]:
    num_tensors = 0
    total_bytes = 0
    for val in state_dict.values():
        if isinstance(val, torch.Tensor):
            num_tensors += 1
            total_bytes += val.numel() * val.element_size()
    return num_tensors, total_bytes


class CheckpointManager:
    """Crash-safe checkpoint manager with atomic commits, integrity checks, and retention."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        keep_last: int = 3,
        keep_best_metric: str | None = None,
        lower_is_better: bool = True,
    ) -> None:
        if keep_last < 0:
            raise ValueError("keep_last must be >= 0")
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.keep_best_metric = keep_best_metric
        self.lower_is_better = lower_is_better

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        *,
        step: int,
        strategy: str = "single",
        rank: int = 0,
        world_size: int = 1,
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically persist model, optimizer, RNG state, and sharded metadata."""
        final_dir = self.root_dir / f"step_{step:07d}"
        staging_dir = self.root_dir / f"step_{step:07d}.tmp.{os.getpid()}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        base_model = unwrap(model)
        model_sd = base_model.state_dict()
        num_tensors, total_bytes = _tensor_stats(model_sd)

        state_file = staging_dir / f"shard_r{rank:04d}.pt"
        payload = {
            "step": step,
            "model": model_sd,
            "optimizer": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
        }
        torch.save(payload, state_file)
        digest = _sha256_file(state_file)

        shard_meta = ShardMetadata(
            rank=rank,
            world_size=world_size,
            num_tensors=num_tensors,
            total_bytes=total_bytes,
            sha256=digest,
        )
        meta = CheckpointMetadata(
            step=step,
            strategy=strategy,
            world_size=world_size,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            metrics=dict(metrics or {}),
            shards=[shard_meta],
            extra=dict(extra or {}),
        )
        (staging_dir / "meta.json").write_text(json.dumps(dataclasses.asdict(meta), indent=2), encoding="utf-8")

        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(staging_dir, final_dir)
        _atomic_write_text(self.root_dir / "latest", final_dir.name)
        self._prune()
        return final_dir

    def latest(self) -> Path | None:
        """Return the latest committed checkpoint directory, or None if none exist."""
        marker = self.root_dir / "latest"
        if marker.exists():
            candidate = self.root_dir / marker.read_text(encoding="utf-8").strip()
            if candidate.is_dir() and (candidate / "meta.json").exists():
                return candidate
        committed = sorted(p for p in self.root_dir.glob("step_*") if p.is_dir() and ".tmp." not in p.name and (p / "meta.json").exists())
        return committed[-1] if committed else None

    def load(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        *,
        checkpoint_dir: str | Path | None = None,
        rank: int = 0,
        restore_rng: bool = True,
        verify: bool = True,
    ) -> CheckpointMetadata:
        """Restore model (and optionally optimizer and RNG state) from a checkpoint."""
        target = Path(checkpoint_dir) if checkpoint_dir is not None else self.latest()
        if target is None:
            raise FileNotFoundError(f"No committed checkpoints found under {self.root_dir}")
        if verify and not self.verify_integrity(target, rank=rank):
            raise ValueError(f"Checkpoint integrity check failed for {target}")

        meta = CheckpointMetadata.from_dict(json.loads((target / "meta.json").read_text(encoding="utf-8")))
        payload = torch.load(target / f"shard_r{rank:04d}.pt", map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(payload["model"])
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if restore_rng and "rng_state" in payload:
            torch.set_rng_state(payload["rng_state"])
        return meta

    def verify_integrity(self, checkpoint_dir: str | Path, *, rank: int = 0) -> bool:
        """Verify that the shard file matches its SHA-256 checksum recorded in ``meta.json``."""
        target = Path(checkpoint_dir)
        meta_path = target / "meta.json"
        shard_path = target / f"shard_r{rank:04d}.pt"
        if not meta_path.exists() or not shard_path.exists():
            return False
        meta = CheckpointMetadata.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        expected = next((s.sha256 for s in meta.shards if s.rank == rank), None)
        if expected is None:
            return False
        return _sha256_file(shard_path) == expected

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """Return metadata for all committed checkpoints sorted by step ascending."""
        items: list[CheckpointMetadata] = []
        for path in sorted(self.root_dir.glob("step_*")):
            if not path.is_dir() or ".tmp." in path.name:
                continue
            meta_file = path / "meta.json"
            if meta_file.exists():
                items.append(CheckpointMetadata.from_dict(json.loads(meta_file.read_text(encoding="utf-8"))))
        return sorted(items, key=lambda m: m.step)

    def cleanup_incomplete(self) -> list[Path]:
        """Delete leftover ``*.tmp.*`` staging directories from interrupted saves."""
        removed: list[Path] = []
        for path in self.root_dir.glob("*.tmp.*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed.append(path)
        return removed

    def _prune(self) -> None:
        if self.keep_last == 0:
            return
        committed = sorted(
            (p for p in self.root_dir.glob("step_*") if p.is_dir() and ".tmp." not in p.name and (p / "meta.json").exists()),
            key=lambda p: p.name,
        )
        protected: set[Path] = set(committed[-self.keep_last :])
        best = self._best_checkpoint(committed)
        if best is not None:
            protected.add(best)
        for path in committed:
            if path not in protected:
                shutil.rmtree(path, ignore_errors=True)

    def _best_checkpoint(self, dirs: list[Path]) -> Path | None:
        if not self.keep_best_metric:
            return None
        scored: list[tuple[float, Path]] = []
        for d in dirs:
            meta = CheckpointMetadata.from_dict(json.loads((d / "meta.json").read_text(encoding="utf-8")))
            if self.keep_best_metric in meta.metrics:
                scored.append((meta.metrics[self.keep_best_metric], d))
        if not scored:
            return None
        scored.sort(key=lambda pair: pair[0], reverse=not self.lower_is_better)
        return scored[0][1]
