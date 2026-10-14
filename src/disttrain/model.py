"""A GPT-2-style decoder-only transformer, written to be easy to read and to shard.

Architecture (same as GPT-2, with a few modern defaults):

    tokens -> token embedding + learned position embedding
           -> N x Block( x + Attn(LN(x)) ; x + MLP(LN(x)) )      <- "pre-norm" residual blocks
           -> final LayerNorm -> linear head (weights tied to the embedding) -> logits

Design choices that matter for distributed training:

* Each :class:`Block` is a self-contained ``nn.Module``. FSDP shards the model
  one block at a time ("wrapping unit"), and activation checkpointing
  recomputes one block at a time, so a clean block boundary is required.
* Attention uses ``F.scaled_dot_product_attention``, which dispatches to
  FlashAttention / memory-efficient kernels on GPU: O(T) memory instead of O(T^2).
* Weight tying (embedding == output head) saves ``vocab x d_model`` parameters
  (38.6M for GPT-2's vocabulary at d=768), a third of the 124M model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    """Model hyper-parameters. Defaults match GPT-2 small (124M parameters)."""

    vocab_size: int = 50304  # GPT-2's 50257 rounded up to a multiple of 128: faster matmuls
    block_size: int = 1024  # maximum context length
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # pre-training on lots of data rarely needs dropout
    bias: bool = False  # no bias in Linear/LayerNorm: slightly faster, same quality
    activation_checkpointing: bool = False  # trade ~30% extra compute for much less activation memory


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token only attends to earlier tokens."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "embedding size must split evenly across heads"
        # One matmul produces queries, keys and values for all heads at once (3x fewer kernel launches).
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time (sequence length), channels (n_embd)
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): each head attends independently.
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # is_causal=True applies the triangular mask inside the fused kernel (no T x T mask tensor).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads side by side
        return self.proj(y)


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU, project back."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    """One transformer layer with pre-norm residual connections.

    Pre-norm (LayerNorm *before* each sub-layer) keeps the residual stream
    un-normalised, which makes deep models train stably without warm-up tricks.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """The full language model. ``forward(idx, targets)`` returns (logits, loss)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # learned absolute positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying (see module docstring)
        self.apply(self._init_weights)
        # GPT-2 trick: scale the init of residual output projections by 1/sqrt(2 * n_layer),
        # so the residual stream's variance doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 initialisation: N(0, 0.02) weights, zero biases."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count; position embeddings are excluded by convention when non_embedding=True."""
        n = sum(p.numel() for p in self.parameters())
        return n - self.wpe.weight.numel() if non_embedding else n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert self.cfg.block_size >= T, f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            if self.cfg.activation_checkpointing and self.training:
                # Don't keep this block's intermediate activations; recompute them in the backward pass.
                # use_reentrant=False is the recommended implementation (works with FSDP and torch.compile).
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            # Cross-entropy in float32 even under bf16 autocast: the softmax over 50k classes needs the precision.
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = 50) -> torch.Tensor:
        """Sample a continuation autoregressively (for qualitative checks during training)."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                # Keep only the k most likely tokens; avoids sampling from the long tail of junk.
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx


# Named presets used by configs and benchmarks.
PRESETS: dict[str, dict] = {
    "tiny": dict(n_layer=2, n_head=2, n_embd=64, block_size=64, vocab_size=256),  # unit tests
    "gpt2-small": dict(n_layer=12, n_head=12, n_embd=768),  # 124M
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1.5B
}


def build_model(preset: str, **overrides) -> GPT:
    """Create a model from a preset name, e.g. ``build_model("gpt2-small", block_size=512)``."""
    return GPT(GPTConfig(**{**PRESETS[preset], **overrides}))
