"""Tests for LoRA / QLoRA adapter config, target-module selector, injection, and merge."""

from __future__ import annotations

import pytest
import torch

from disttrain.lora import (
    LoRAConfig,
    LoRALinear,
    inject_lora,
    lora_state_dict,
    merge_lora,
    select_target_modules,
    trainable_param_summary,
)
from disttrain.model import build_model


def test_select_target_modules_matches_linear_and_excludes_lm_head() -> None:
    model = build_model("tiny")
    attn_targets = select_target_modules(model, ("qkv", "proj"))
    assert "blocks.0.attn.qkv" in attn_targets
    assert "blocks.0.attn.proj" in attn_targets
    assert "blocks.0.mlp.proj" in attn_targets
    assert "lm_head" not in attn_targets


def test_lora_zero_init_preserves_exact_logits() -> None:
    torch.manual_seed(0)
    model = build_model("tiny").eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        base_logits, _ = model(x)

    adapted = inject_lora(model, LoRAConfig(r=4, alpha=8.0, target_modules=("qkv", "fc")))
    assert len(adapted) == 4  # 2 layers x (qkv, fc)
    with torch.no_grad():
        lora_logits, _ = model(x)
    assert torch.allclose(base_logits, lora_logits, atol=1e-6)


def test_lora_freezes_base_and_trains_only_adapters() -> None:
    torch.manual_seed(1)
    model = build_model("tiny")
    inject_lora(model, LoRAConfig(r=4, alpha=8.0, target_modules=("qkv",)))
    summary = trainable_param_summary(model)
    assert 0 < summary["trainable_params"] < summary["total_params"]
    assert summary["trainable_pct"] < 15.0

    wte_before = model.wte.weight.detach().clone()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randint(0, 256, (2, 16))
    for _ in range(3):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert torch.equal(model.wte.weight, wte_before)
    sd = lora_state_dict(model)
    assert len(sd) == 4  # 2 layers x (lora_A, lora_B)
    assert all("lora_" in k for k in sd)


def test_merge_lora_matches_unmerged_outputs() -> None:
    torch.manual_seed(2)
    model = build_model("tiny").eval()
    inject_lora(model, LoRAConfig(r=4, alpha=16.0, target_modules=("qkv", "fc")))
    # Set non-zero lora_B so adapter delta is active.
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            torch.nn.init.normal_(mod.lora_B, std=0.05)

    x = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        unmerged, _ = model(x)

    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    with torch.no_grad():
        merged, _ = model(x)
    assert torch.allclose(unmerged, merged, atol=1e-5)


@pytest.mark.parametrize("quant_bits", [4, 8])
def test_qlora_quantized_forward_backward_and_merge(quant_bits: int) -> None:
    torch.manual_seed(3)
    cfg = LoRAConfig(r=4, alpha=8.0, target_modules=("qkv", "fc"), quant_bits=quant_bits, quant_block_size=32)
    assert cfg.is_qlora is (quant_bits == 4)
    model = build_model("tiny")
    inject_lora(model, cfg)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randint(0, 256, (2, 16))
    _, loss = model(x[:, :-1], x[:, 1:])
    assert loss is not None and torch.isfinite(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()

    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())


def test_lora_config_validation() -> None:
    with pytest.raises(ValueError):
        LoRAConfig(r=0)
    with pytest.raises(ValueError):
        LoRAConfig(quant_bits=3)
    with pytest.raises(ValueError):
        LoRAConfig(target_modules=())
