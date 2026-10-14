"""Throughput accounting: tokens/second and Model FLOPs Utilisation (MFU).

MFU = (FLOPs the model *needs* per second) / (peak FLOPs the hardware *offers*).
It is the standard way to say how well a training setup uses its GPUs (PaLM
paper, Appendix B). Good large-scale runs reach 40-60 %.

Training FLOPs per token for a decoder-only transformer:

    6 * N            matmuls: forward (2N) + backward (4N), N = non-embedding params
  + 12 * L * T * d   attention scores and weighted sum (forward + backward)

Activation recomputation is *not* counted. It is overhead, so checkpointing
correctly lowers MFU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Dense bf16 peak throughput (no sparsity) from NVIDIA data sheets, in FLOP/s.
PEAK_BF16_FLOPS = {
    "A100": 312e12,
    "H100": 989e12,  # SXM; the PCIe card is lower
    "L4": 121e12,
    "T4": 65e12,  # fp16 (T4 has no bf16)
    "G4": 0.0,  # unknown/unsupported: MFU is reported as 0
}


def flops_per_token(n_params: int, n_layer: int, block_size: int, n_embd: int) -> float:
    """Training FLOPs per token (see module docstring)."""
    return 6 * n_params + 12 * n_layer * block_size * n_embd


def peak_flops(device_name: str) -> float:
    """Look up the peak bf16 FLOP/s for a GPU name such as 'NVIDIA A100-SXM4-40GB'."""
    for key, value in PEAK_BF16_FLOPS.items():
        if key in device_name.upper():
            return value
    return 0.0


@dataclass
class ThroughputMeter:
    """Measures tokens/s and MFU over a window of steps, skipping warm-up steps.

    The first steps include CUDA kernel autotuning and ``torch.compile``
    compilation, which would make throughput look much worse than steady state.
    """

    flops_per_token: float
    peak_flops_total: float  # peak of *all* GPUs in the job
    warmup_steps: int = 3
    _tokens: int = 0
    _start: float | None = None
    _steps: int = 0
    history: list[float] = field(default_factory=list)

    def step(self, tokens: int) -> None:
        """Call once per optimizer step with the number of tokens processed globally."""
        self._steps += 1
        if self._steps == self.warmup_steps:
            self._start = time.perf_counter()  # start the clock after warm-up
        elif self._steps > self.warmup_steps:
            self._tokens += tokens

    @property
    def tokens_per_sec(self) -> float:
        if self._start is None or self._tokens == 0:
            return 0.0
        return self._tokens / (time.perf_counter() - self._start)

    @property
    def mfu(self) -> float:
        if not self.peak_flops_total:
            return 0.0
        return self.tokens_per_sec * self.flops_per_token / self.peak_flops_total
