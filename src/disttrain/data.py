"""Token data pipeline: pre-tokenise once into a flat binary file, then sample windows from it.

Why a memory-mapped ``uint16`` file instead of a ``Dataset`` of strings?

* Tokenising on the fly would make the GPU wait for the CPU.
* GPT-2's vocabulary (50,257) fits in 16 bits, so 1B tokens take 2 GB on disk,
  and ``np.memmap`` reads only the pages actually used. Nothing is loaded into RAM up front.
* Every rank opens the same file. Ranks draw *different* random windows
  (different seeds), so together they see N times more data per step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenWindows:
    """Random (input, target) windows from a flat token file.

    The target is the input shifted by one token: at every position the model
    learns to predict the next token.

    Args:
        path: ``.bin`` file of ``uint16`` token ids (see :func:`write_tokens`).
        block_size: Sequence length T.
        rank, world_size: Used to give each rank its own random stream.
        seed: Base seed, so runs are reproducible.
    """

    def __init__(self, path: str | Path, block_size: int, rank: int = 0, world_size: int = 1, seed: int = 0):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.tokens) <= block_size + 1:
            raise ValueError(f"{path} has {len(self.tokens)} tokens, need more than block_size + 1 = {block_size + 1}")
        self.block_size = block_size
        # Different seed per rank, so no two ranks train on the same windows in the same step.
        self.rng = np.random.default_rng(seed * 1000 + rank)
        self.world_size = world_size

    def __len__(self) -> int:
        return len(self.tokens)

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``batch_size`` windows and move them to ``device``."""
        starts = self.rng.integers(0, len(self.tokens) - self.block_size - 1, size=batch_size)
        # int64 because embedding lookups and cross-entropy need long indices.
        x = np.stack([self.tokens[s : s + self.block_size] for s in starts]).astype(np.int64)
        y = np.stack([self.tokens[s + 1 : s + 1 + self.block_size] for s in starts]).astype(np.int64)
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        if device.type == "cuda":
            # Pinned memory + non_blocking lets the host->GPU copy overlap with compute.
            return xt.pin_memory().to(device, non_blocking=True), yt.pin_memory().to(device, non_blocking=True)
        return xt.to(device), yt.to(device)

    def sequential_batches(self, batch_size: int, device: torch.device, max_batches: int):
        """Deterministic, non-overlapping batches for validation (the same every evaluation)."""
        step = self.block_size * batch_size
        for i in range(min(max_batches, (len(self.tokens) - 1) // step)):
            chunk = np.asarray(self.tokens[i * step : (i + 1) * step + 1], dtype=np.int64)
            x = torch.from_numpy(chunk[:-1].reshape(batch_size, self.block_size))
            y = torch.from_numpy(chunk[1:].reshape(batch_size, self.block_size))
            yield x.to(device), y.to(device)


def write_tokens(token_ids: np.ndarray | list[int], path: str | Path) -> Path:
    """Save token ids as a flat ``uint16`` file that :class:`TokenWindows` can read."""
    arr = np.asarray(token_ids)
    if arr.size and arr.max() >= 2**16:
        raise ValueError("token id does not fit in uint16; use a vocabulary smaller than 65,536")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype(np.uint16).tofile(path)
    return path
