# ADR 002: Sequence Packing, Assistant-Only Loss Masking, and Post-Training Alignment

## Status
Accepted

## Context
Instruction-tuning (SFT) and preference-alignment (DPO / GRPO) datasets contain highly variable sequence lengths. Naive padding wastes 40–70% of GPU FLOPs on pad tokens, while computing cross-entropy over system/user prompts degrades instruction-following quality.

## Decision
1. **Packed streaming dataloader (`disttrain.packed_data`)**: Pack multiple tokenized examples into fixed `seq_len` windows with per-document `position_ids` resets and `doc_ids` tracking.
2. **Block-diagonal causal mask (`build_packed_causal_mask`)**: Prevent cross-document attention leakage inside packed sequences while keeping padding positions fully masked.
3. **Assistant-only masking (`encode_chat_turns`)**: Mask all non-assistant target positions with `ignore_index` so gradients flow solely from assistant completions and EOS tokens.
4. **Composable PEFT & alignment (`disttrain.lora`, `disttrain.alignment`)**: Provide pure-PyTorch LoRA/QLoRA adapters (with zero-init equivalence and lossless inference merge) alongside `SFTStep`, `DPOStep`, and `GRPOStep` (group-normalized advantages + PPO clipping + unbiased k3 KL penalty).

## Consequences
- Packing efficiency approaches 90–100% on short/medium instruction datasets.
- Pre-training, SFT, DPO, and GRPO share one model/adapter foundation and are 100% testable on CPU.
