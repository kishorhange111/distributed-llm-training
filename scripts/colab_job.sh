#!/usr/bin/env bash
# Run a job on a Google Colab GPU runtime (A100 recommended).
#
#   colab new -s train --gpu A100
#   colab upload -s train scripts/colab_job.sh /content/colab_job.sh
#   echo 'import subprocess; subprocess.run(["bash", "/content/colab_job.sh", "bench"])' | colab exec -s train
#
# Jobs:  bench   throughput/memory ablation (gpt2-medium and gpt2-xl)
#        train   prepare FineWeb-Edu (1B tokens) and pre-train GPT-2 small, then export to Hugging Face format
# Results are packed into /content/results.tgz.
# Results are packed on exit even if a step fails, so a crash keeps what already finished.
set -uo pipefail

REPO=${REPO:-https://github.com/kishorhange111/distributed-llm-training}
BRANCH=${BRANCH:-main}
DIR=/content/distributed-llm-training
trap 'tar -czf /content/results.tgz -C "$DIR" results 2>/dev/null; echo "packed results ($(date -u +%T))"' EXIT
set -e

if [ ! -d "$DIR" ]; then git clone --depth 1 -b "$BRANCH" "$REPO" "$DIR"; else git -C "$DIR" fetch --depth 1 origin "$BRANCH" && git -C "$DIR" reset --hard FETCH_HEAD; fi
cd "$DIR"
pip install -q -e ".[data]"
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p results
set +e

for job in "$@"; do
  echo "=== $job ($(date -u +%T))"
  case "$job" in
    bench)
      python scripts/benchmark.py --preset gpt2-medium --micro_batch 8 --out results/bench_gpt2_medium.json | tee results/bench_gpt2_medium.log
      python scripts/benchmark.py --preset gpt2-xl --micro_batch 4 --out results/bench_gpt2_xl.json | tee results/bench_gpt2_xl.log
      ;;
    train)
      [ -f data/fineweb/train.bin ] || python scripts/prepare_fineweb.py --tokens 1050000000 --out data/fineweb
      [ -f data/fineweb/train.bin ] || { echo "FAILED: data preparation"; exit 1; }
      python -m disttrain.train --config configs/gpt2_small_fineweb.yaml 2>&1 | tee results/train_gpt2_small.log
      # Copy whatever the run produced, even if it stopped early (the log is the result).
      cp runs/gpt2_small_fineweb/{log.jsonl,summary.json,config.json} results/ 2>/dev/null
      python -m disttrain.export runs/gpt2_small_fineweb /content/hf_gpt2_small_fineweb
      ;;
    *) echo "unknown job $job"; exit 1 ;;
  esac
done
echo "DONE $(date -u +%T)"
