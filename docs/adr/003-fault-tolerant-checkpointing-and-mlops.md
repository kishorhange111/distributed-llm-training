# ADR 003: Fault-Tolerant Checkpointing, TorchElastic, and Automated Post-Train Eval Gates

## Status
Accepted

## Context
Preemptible cloud GPUs (Colab, GKE Spot, AWS EC2 Spot) can terminate workers mid-checkpoint. A naive overwrite of `latest` or partial shard writes can corrupt recovery state, and unmonitored training runs risk promoting regressed models.

## Decision
1. **Two-phase atomic checkpoints (`disttrain.checkpoint_manager`)**: Write shards and `meta.json` (including SHA-256 checksums and per-rank byte statistics) into `step_<N>.tmp.<pid>`, atomically rename into place, and atomically update the `latest` pointer via `os.replace`.
2. **Best + recency retention**: Prune older checkpoints beyond `keep_last` while always protecting the best checkpoint by tracked validation metric (e.g., lowest `val_loss`).
3. **Kubernetes PyTorchJob & TorchElastic (`deploy/k8s/`)**: Provide production Kubeflow `PyTorchJob` manifests for both static FSDP2 jobs and `c10d` TorchElastic spot-node jobs with shared PVC mounts.
4. **Rank-0 MLflow tracking & post-train eval gates (`disttrain.callbacks`)**: Log params, metrics, and artifacts exclusively on rank 0 (with deterministic offline JSON fallback in CI), and run `PostTrainEvalHook` automatically on training completion to enforce `val_loss`, perplexity, and throughput gates.

## Consequences
- Preemptions never corrupt the last committed checkpoint; orphan `.tmp.*` staging directories are cleaned automatically.
- Quality and throughput regressions are caught automatically before model export or promotion.
