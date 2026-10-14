"""disttrain: GPT pre-training that scales from one GPU to many with DDP or FSDP2.

Modules:
    model        GPT-2-style transformer (block-structured for sharding/checkpointing)
    distributed  torchrun process-group setup and collective helpers
    strategies   single / DDP / FSDP2 wrapping, mixed precision, CPU offload
    data         memory-mapped token windows, per-rank sampling
    train        the training loop (grad accumulation, no_sync, LR schedule, logging)
    checkpoint   sharded, resumable checkpoints with torch.distributed.checkpoint
    metrics      tokens/s and Model FLOPs Utilisation (MFU)
"""

__version__ = "0.1.0"
