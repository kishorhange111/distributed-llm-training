"""Throughput and memory ablation: what each optimisation buys on one GPU.

Each configuration trains for a few dozen steps on synthetic tokens (data
loading is excluded, so we measure only the model, precision and strategy) in
a fresh subprocess, so GPU memory stats and compile caches never leak between
runs. An out-of-memory error is recorded as a result: "this setup cannot
train this model" is exactly what the table should show.

Usage:  python scripts/benchmark.py --preset gpt2-medium --out results/bench_gpt2_medium.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# (label, overrides). Ordered so each row adds one technique on top of a baseline.
SUITE = [
    ("fp32 eager", dict(dtype="float32")),
    ("bf16 autocast", dict(dtype="bfloat16")),
    ("bf16 + torch.compile", dict(dtype="bfloat16", compile=True)),
    ("bf16 + compile + activation ckpt", dict(dtype="bfloat16", compile=True, activation_checkpointing=True)),
    ("FSDP2 (bf16 mixed precision)", dict(strategy="fsdp", dtype="bfloat16")),
    ("FSDP2 + CPU offload", dict(strategy="fsdp", dtype="bfloat16", cpu_offload=True)),
]


def run_one(preset: str, micro_batch: int, steps: int, overrides: dict, torchrun: bool) -> dict:
    """Train briefly in a subprocess and return its summary.json (or the failure reason)."""
    with tempfile.TemporaryDirectory() as tmp:
        args = [
            "-m", "disttrain.train", "--preset", preset, "--micro_batch_size", str(micro_batch),
            "--max_steps", str(steps), "--warmup_steps", "2", "--log_interval", "5", "--out_dir", tmp,
        ]  # fmt: skip
        for k, v in overrides.items():
            args += [f"--{k}", str(v)]
        # FSDP needs a process group, so launch it through torchrun even for one GPU.
        cmd = ([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1"] + args[1:]) if torchrun else [sys.executable] + args
        proc = subprocess.run(cmd, capture_output=True, text=True)
        summary_path = Path(tmp) / "summary.json"
        if proc.returncode == 0 and summary_path.exists():
            return json.loads(summary_path.read_text())
        err = proc.stderr[-2000:]
        return {"error": "out of memory" if "OutOfMemoryError" in err or "CUDA out of memory" in err else err.strip().splitlines()[-1:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="gpt2-medium")
    parser.add_argument("--micro_batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out", default="results/benchmark.json")
    parser.add_argument("--only", nargs="*", help="run only these labels")
    args = parser.parse_args()

    rows = []
    for label, overrides in SUITE:
        if args.only and label not in args.only:
            continue
        res = run_one(args.preset, args.micro_batch, args.steps, overrides, torchrun=overrides.get("strategy") == "fsdp")
        rows.append({"config": label, **overrides, **res})
        print(json.dumps(rows[-1]), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"preset": args.preset, "micro_batch": args.micro_batch, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
