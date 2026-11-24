"""LoRA and QLoRA parameter-efficient fine-tuning adapters and target-module selector.

Provides pure-PyTorch LoRA and simulated 4-bit/8-bit QLoRA adapters that work
identically on CPU and GPU:

* :class:`LoRAConfig`: rank, alpha, dropout, target module patterns, and QLoRA quantization config.
* :func:`select_target_modules`: deterministic ``nn.Linear`` selector that skips tied output heads.
* :func:`inject_lora` / :func:`merge_lora`: in-place adapter injection and lossless merge for inference.
* :func:`lora_state_dict` / :func:`trainable_param_summary`: adapter-only serialization and accounting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for LoRA / QLoRA adapter injection."""

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("qkv", "proj")
    exclude_modules: tuple[str, ...] = ("lm_head",)
    quant_bits: int | None = None  # None = standard LoRA; 4 or 8 = QLoRA simulated quantization
    quant_block_size: int = 64
    double_quant: bool = False
    compute_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.r < 1:
            raise ValueError("LoRA rank r must be >= 1")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be > 0")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.quant_bits not in (None, 4, 8):
            raise ValueError("quant_bits must be None, 4, or 8")
        if self.quant_block_size < 1:
            raise ValueError("quant_block_size must be >= 1")
        if not self.target_modules:
            raise ValueError("target_modules must be non-empty")

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    @property
    def is_qlora(self) -> bool:
        return self.quant_bits == 4


def quantize_weight_blockwise(weight: torch.Tensor, bits: int, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Symmetric block-wise quantization to signed ``bits``-bit integers (4-bit or 8-bit)."""
    qmax = (1 << (bits - 1)) - 1
    orig_shape = tuple(weight.shape)
    flat = weight.detach().float().reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(blocks / scales), -qmax, qmax).to(torch.int8)
    return q, scales.squeeze(1), orig_shape


def dequantize_weight_blockwise(q: torch.Tensor, scales: torch.Tensor, orig_shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Reconstruct a floating-point weight matrix from block-wise quantized tensors."""
    total = math.prod(orig_shape)
    deq = (q.float() * scales.unsqueeze(1)).reshape(-1)[:total]
    return deq.view(orig_shape).to(dtype=dtype)


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen ``nn.Linear`` (supports optional QLoRA quantization)."""

    def __init__(self, base: nn.Linear, cfg: LoRAConfig) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = cfg.r
        self.alpha = cfg.alpha
        self.scaling = cfg.scaling
        self.quant_bits = cfg.quant_bits
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        if cfg.quant_bits is not None:
            q, scales, orig_shape = quantize_weight_blockwise(base.weight, cfg.quant_bits, cfg.quant_block_size)
            self.register_buffer("qweight", q)
            self.register_buffer("qscales", scales)
            self._orig_shape = orig_shape
            self.weight = None
        else:
            self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
            self._orig_shape = tuple(base.weight.shape)

        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty((cfg.r, self.in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, cfg.r), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _base_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self.weight is not None:
            return self.weight.to(dtype=dtype)
        return dequantize_weight_blockwise(self.qweight, self.qscales, self._orig_shape, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_w = self._base_weight(x.dtype)
        out = F.linear(x, base_w, self.bias)
        adapter = F.linear(F.linear(self.dropout(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return out + adapter * self.scaling

    @torch.no_grad()
    def to_merged_linear(self) -> nn.Linear:
        """Fold ``scaling * lora_B @ lora_A`` into a standard ``nn.Linear``."""
        dtype = self.lora_A.dtype
        merged_w = self._base_weight(dtype) + (self.lora_B @ self.lora_A) * self.scaling
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None, device=merged_w.device, dtype=dtype)
        linear.weight.copy_(merged_w)
        if self.bias is not None and linear.bias is not None:
            linear.bias.copy_(self.bias)
        return linear


def select_target_modules(
    model: nn.Module,
    target_modules: Sequence[str],
    exclude_modules: Sequence[str] = ("lm_head",),
) -> list[str]:
    """Return dotted names of all ``nn.Linear`` submodules matching ``target_modules``."""
    targets = set(target_modules)
    excludes = set(exclude_modules)
    matched: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if name in excludes or leaf in excludes:
            continue
        if name in targets or leaf in targets:
            matched.append(name)
    return matched


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Freeze base weights and replace targeted ``nn.Linear`` layers with :class:`LoRALinear`."""
    matched = select_target_modules(model, cfg.target_modules, cfg.exclude_modules)
    if not matched:
        raise ValueError(f"No nn.Linear modules matched target_modules={cfg.target_modules!r}")

    for param in model.parameters():
        param.requires_grad = False

    modules_by_name = dict(model.named_modules())
    for name in matched:
        base = modules_by_name[name]
        assert isinstance(base, nn.Linear)
        _set_submodule(model, name, LoRALinear(base, cfg))
    return matched


def merge_lora(model: nn.Module) -> nn.Module:
    """Replace all :class:`LoRALinear` modules in ``model`` with merged ``nn.Linear`` layers."""
    lora_paths = [name for name, mod in model.named_modules() if isinstance(mod, LoRALinear)]
    modules_by_name = dict(model.named_modules())
    for name in lora_paths:
        adapter = modules_by_name[name]
        assert isinstance(adapter, LoRALinear)
        _set_submodule(model, name, adapter.to_merged_linear())
    return model


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA adapter parameters (``lora_A`` and ``lora_B``)."""
    return {name: param.detach().clone() for name, param in model.named_parameters() if "lora_" in name}


def trainable_param_summary(model: nn.Module) -> dict[str, Any]:
    """Report trainable vs total parameter counts and percentage."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (100.0 * trainable / total) if total else 0.0
    return {"trainable_params": trainable, "total_params": total, "trainable_pct": round(pct, 4)}
