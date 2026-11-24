"""SFT, DPO, and GRPO preference-alignment losses and trainer step abstractions.

Provides clean, testable post-training objectives for decoder-only language models:

* :func:`sft_loss` / :class:`SFTStep`: supervised fine-tuning with assistant-only masking.
* :func:`dpo_loss` / :class:`DPOStep`: Direct Preference Optimization (Rafailov et al., 2023).
* :func:`grpo_loss` / :class:`GRPOStep`: Group Relative Policy Optimization (DeepSeekMath / R1)
  with group-normalized advantages, clipped surrogate ratio, and k3 KL penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StepMetrics:
    """Structured metrics returned by a single post-training optimizer step."""

    loss: float
    grad_norm: float
    extras: dict[str, float] = field(default_factory=dict)


def sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sequence summed log-probabilities over non-masked target tokens.

    Args:
        logits: ``(B, T, V)`` float tensor.
        labels: ``(B, T)`` int64 tensor with ``ignore_index`` on masked positions.

    Returns:
        ``(seq_logps, token_counts)``, each of shape ``(B,)``.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    mask = labels != ignore_index
    safe_labels = labels.masked_fill(~mask, 0)
    per_token = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    per_token = per_token * mask.to(per_token.dtype)
    return per_token.sum(dim=-1), mask.sum(dim=-1).to(per_token.dtype)


def sft_loss(logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = -100) -> torch.Tensor:
    """Token-level cross-entropy loss over unmasked assistant tokens."""
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=ignore_index)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Direct Preference Optimization loss and implicit reward telemetry."""
    if beta <= 0:
        raise ValueError("beta must be > 0")
    if not (0.0 <= label_smoothing < 0.5):
        raise ValueError("label_smoothing must be in [0, 0.5)")

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    margin = chosen_rewards - rejected_rewards

    losses = -(1.0 - label_smoothing) * F.logsigmoid(margin) - label_smoothing * F.logsigmoid(-margin)
    loss = losses.mean()
    metrics = {
        "chosen_reward": float(chosen_rewards.detach().mean().item()),
        "rejected_reward": float(rejected_rewards.detach().mean().item()),
        "reward_margin": float(margin.detach().mean().item()),
        "reward_accuracy": float((margin.detach() > 0).float().mean().item()),
    }
    return loss, metrics


def compute_group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-8) -> torch.Tensor:
    """Normalize scalar rewards within each prompt group of size ``group_size``."""
    if group_size < 2:
        raise ValueError("group_size must be >= 2 for group-relative normalization")
    if rewards.numel() % group_size != 0:
        raise ValueError(f"rewards length ({rewards.numel()}) must be divisible by group_size ({group_size})")
    grouped = rewards.float().view(-1, group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, unbiased=False, keepdim=True)
    return ((grouped - mean) / (std + eps)).reshape(-1)


def grpo_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    rewards: torch.Tensor,
    *,
    group_size: int,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Group Relative Policy Optimization objective with PPO clipping and k3 KL penalty."""
    if clip_eps <= 0:
        raise ValueError("clip_eps must be > 0")
    if kl_beta < 0:
        raise ValueError("kl_beta must be >= 0")

    advantages = compute_group_advantages(rewards, group_size).detach()
    log_ratio = policy_logps - old_logps.detach()
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    surrogate = torch.minimum(unclipped, clipped)

    # Unbiased k3 KL estimator: exp(log_ref - log_pol) - (log_ref - log_pol) - 1 >= 0
    ref_diff = ref_logps.detach() - policy_logps
    kl = torch.exp(ref_diff) - ref_diff - 1.0

    loss = (-surrogate + kl_beta * kl).mean()
    clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
    metrics = {
        "mean_reward": float(rewards.detach().float().mean().item()),
        "mean_kl": float(kl.detach().mean().item()),
        "clip_fraction": float(clip_frac.detach().item()),
    }
    return loss, metrics


class SFTStep:
    """Single optimizer step for supervised fine-tuning."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, *, grad_clip: float = 1.0, ignore_index: int = -100) -> None:
        self.model = model
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.ignore_index = ignore_index

    def step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> StepMetrics:
        self.model.train()
        logits, _ = self.model(input_ids)
        loss = sft_loss(logits, labels, ignore_index=self.ignore_index)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip))
        self.optimizer.step()
        return StepMetrics(loss=float(loss.detach().item()), grad_norm=grad_norm)


class DPOStep:
    """Single optimizer step for Direct Preference Optimization."""

    def __init__(
        self,
        policy: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        grad_clip: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        self.policy = policy
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.grad_clip = grad_clip
        self.ignore_index = ignore_index

    def step(
        self,
        chosen_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> StepMetrics:
        self.policy.train()
        self.ref_model.eval()
        pol_c_logits, _ = self.policy(chosen_ids)
        pol_r_logits, _ = self.policy(rejected_ids)
        pol_c_lp, _ = sequence_logprobs(pol_c_logits, chosen_labels, ignore_index=self.ignore_index)
        pol_r_lp, _ = sequence_logprobs(pol_r_logits, rejected_labels, ignore_index=self.ignore_index)

        with torch.no_grad():
            ref_c_logits, _ = self.ref_model(chosen_ids)
            ref_r_logits, _ = self.ref_model(rejected_ids)
            ref_c_lp, _ = sequence_logprobs(ref_c_logits, chosen_labels, ignore_index=self.ignore_index)
            ref_r_lp, _ = sequence_logprobs(ref_r_logits, rejected_labels, ignore_index=self.ignore_index)

        loss, extras = dpo_loss(
            pol_c_lp,
            pol_r_lp,
            ref_c_lp,
            ref_r_lp,
            beta=self.beta,
            label_smoothing=self.label_smoothing,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip))
        self.optimizer.step()
        return StepMetrics(loss=float(loss.detach().item()), grad_norm=grad_norm, extras=extras)


class GRPOStep:
    """Single optimizer step for Group Relative Policy Optimization."""

    def __init__(
        self,
        policy: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        group_size: int = 4,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
        grad_clip: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        self.policy = policy
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.grad_clip = grad_clip
        self.ignore_index = ignore_index

    def step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        rewards: torch.Tensor,
        old_logps: torch.Tensor | None = None,
    ) -> StepMetrics:
        self.policy.train()
        self.ref_model.eval()
        pol_logits, _ = self.policy(input_ids)
        pol_lp, _ = sequence_logprobs(pol_logits, labels, ignore_index=self.ignore_index)
        if old_logps is None:
            old_logps = pol_lp.detach()

        with torch.no_grad():
            ref_logits, _ = self.ref_model(input_ids)
            ref_lp, _ = sequence_logprobs(ref_logits, labels, ignore_index=self.ignore_index)

        loss, extras = grpo_loss(
            pol_lp,
            old_logps,
            ref_lp,
            rewards,
            group_size=self.group_size,
            clip_eps=self.clip_eps,
            kl_beta=self.kl_beta,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip))
        self.optimizer.step()
        return StepMetrics(loss=float(loss.detach().item()), grad_norm=grad_norm, extras=extras)
