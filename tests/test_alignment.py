"""Tests for SFT, DPO, and GRPO losses and trainer step abstractions."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from disttrain.alignment import (
    DPOStep,
    GRPOStep,
    SFTStep,
    compute_group_advantages,
    dpo_loss,
    grpo_loss,
    sequence_logprobs,
)
from disttrain.model import build_model


def test_sequence_logprobs_respects_ignore_index() -> None:
    logits = torch.zeros((2, 4, 10))
    labels = torch.tensor([[-100, -100, 1, 2], [-100, 3, 4, 5]])
    seq_lp, counts = sequence_logprobs(logits, labels, ignore_index=-100)
    assert counts.tolist() == [2.0, 3.0]
    assert seq_lp[0].item() == pytest.approx(2 * math.log(0.1), rel=1e-5)
    assert seq_lp[1].item() == pytest.approx(3 * math.log(0.1), rel=1e-5)


def test_sft_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = build_model("tiny")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    trainer = SFTStep(model, opt, ignore_index=-100)

    x = torch.randint(0, 256, (4, 16))
    labels = x.clone()
    labels[:, :6] = -100  # mask prompt tokens

    first = trainer.step(x, labels)
    for _ in range(15):
        last = trainer.step(x, labels)
    assert last.loss < first.loss
    assert last.grad_norm >= 0.0


def test_dpo_loss_at_init_and_step_increases_margin() -> None:
    # When policy == reference, margin is 0 and loss is ln(2).
    lp = torch.tensor([-5.0, -3.0])
    loss, metrics = dpo_loss(lp, lp, lp, lp, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), rel=1e-5)
    assert metrics["reward_margin"] == pytest.approx(0.0)

    torch.manual_seed(1)
    policy = build_model("tiny")
    ref = copy.deepcopy(policy)
    opt = torch.optim.AdamW(policy.parameters(), lr=2e-3)
    step = DPOStep(policy, ref, opt, beta=0.1)

    chosen = torch.randint(0, 256, (2, 12))
    rejected = torch.randint(0, 256, (2, 12))
    first = step.step(chosen[:, :-1], chosen[:, 1:], rejected[:, :-1], rejected[:, 1:])
    for _ in range(8):
        last = step.step(chosen[:, :-1], chosen[:, 1:], rejected[:, :-1], rejected[:, 1:])
    assert last.extras["reward_margin"] > first.extras["reward_margin"]
    assert last.loss < first.loss


def test_grpo_group_advantages_and_step() -> None:
    rewards = torch.tensor([1.0, 3.0, 2.0, 2.0, 0.0, 4.0])
    adv = compute_group_advantages(rewards, group_size=2)
    assert adv[0].item() == pytest.approx(-1.0, abs=1e-5)
    assert adv[1].item() == pytest.approx(1.0, abs=1e-5)
    assert adv[2].item() == pytest.approx(0.0, abs=1e-5)

    # Identical policy/old/ref should produce zero KL and zero clip fraction.
    logps = torch.tensor([-2.0, -2.5, -1.8, -3.0], requires_grad=True)
    loss, metrics = grpo_loss(logps, logps.detach(), logps.detach(), torch.tensor([0.0, 1.0, 0.5, 1.5]), group_size=2)
    assert torch.isfinite(loss)
    assert metrics["mean_kl"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["clip_fraction"] == pytest.approx(0.0)

    torch.manual_seed(2)
    policy = build_model("tiny")
    ref = copy.deepcopy(policy)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    step = GRPOStep(policy, ref, opt, group_size=2, kl_beta=0.05)

    batch = torch.randint(0, 256, (4, 12))
    rewards_b = torch.tensor([0.0, 1.0, 0.2, 0.9])
    m = step.step(batch[:, :-1], batch[:, 1:], rewards_b)
    assert math.isfinite(m.loss)
    assert m.extras["mean_kl"] >= 0.0
