"""Convert a trained checkpoint into a Hugging Face ``GPT2LMHeadModel``.

Why: once the model is in the standard format, any tool in the ecosystem can
use it (``generate``, ``pipeline``, vLLM, and the evaluation harness in
https://github.com/kishorhange111/llm-evaluation-harness for HellaSwag/ARC).

Differences between our GPT and HF GPT-2 that the converter handles:

* HF stores Linear layers as ``Conv1D`` with transposed weights ``(in, out)``.
* HF GPT-2 always has biases; our model is bias-free, so HF biases are set to 0.
* Our fused attention projection is called ``qkv``; HF calls it ``c_attn``.

Usage:  python -m disttrain.export runs/gpt2_fineweb out/hf_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from disttrain.model import GPT, GPTConfig


def to_hf(model: GPT):
    """Return an equivalent ``transformers.GPT2LMHeadModel`` (same logits for the same input)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    c = model.cfg
    hf = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=c.vocab_size, n_positions=c.block_size, n_embd=c.n_embd, n_layer=c.n_layer, n_head=c.n_head,
            activation_function="gelu_new", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, tie_word_embeddings=True,
        )
    )  # fmt: skip
    sd = model.state_dict()
    with torch.no_grad():
        # Start from all-zero biases (ours has none), then copy weights over.
        for name, p in hf.named_parameters():
            if name.endswith(".bias"):
                p.zero_()
        hf.transformer.wte.weight.copy_(sd["wte.weight"])
        hf.transformer.wpe.weight.copy_(sd["wpe.weight"])
        hf.transformer.ln_f.weight.copy_(sd["ln_f.weight"])
        for i, blk in enumerate(hf.transformer.h):
            p = f"blocks.{i}."
            blk.ln_1.weight.copy_(sd[p + "ln1.weight"])
            blk.ln_2.weight.copy_(sd[p + "ln2.weight"])
            # .t(): nn.Linear stores (out, in), HF Conv1D stores (in, out).
            blk.attn.c_attn.weight.copy_(sd[p + "attn.qkv.weight"].t())
            blk.attn.c_proj.weight.copy_(sd[p + "attn.proj.weight"].t())
            blk.mlp.c_fc.weight.copy_(sd[p + "mlp.fc.weight"].t())
            blk.mlp.c_proj.weight.copy_(sd[p + "mlp.proj.weight"].t())
    return hf


def load_trained(run_dir: str | Path) -> GPT:
    """Load the latest DCP checkpoint of a run into a plain (unsharded) GPT on CPU."""
    import torch.distributed.checkpoint as dcp

    from disttrain import checkpoint

    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    from disttrain.model import PRESETS

    model = GPT(GPTConfig(**{**PRESETS[cfg["preset"]], "block_size": cfg["block_size"]}))
    ckpt = checkpoint.latest(run_dir / "checkpoints")
    state = {"model": model.state_dict()}
    dcp.load(state, checkpoint_id=str(ckpt))  # DCP re-assembles the shards into full tensors
    model.load_state_dict(state["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    hf = to_hf(load_trained(args.run_dir))
    hf.save_pretrained(args.out_dir)
    AutoTokenizer.from_pretrained("openai-community/gpt2").save_pretrained(args.out_dir)
    print("saved", args.out_dir)


if __name__ == "__main__":
    main()
