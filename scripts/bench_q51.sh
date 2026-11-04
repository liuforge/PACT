#!/bin/bash
# Q-51 (RTX 5090): one clean efficiency run of every trained and foundation model (ETTm1, H=96, batch 32), started
# automatically once the Time-MoE jobs of this machine are finished and the GPU has no other process.
cd /mnt/proj/pact-code
export PYTHONPATH=/mnt/proj/pact-code/fm_site:$PYTHONPATH OMP_NUM_THREADS=4
until grep -q "^done" logs/queue_fm_v2_q51.log 2>/dev/null && [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; do sleep 120; done
mkdir -p bench && touch bench/BENCH_RUNNING
/mnt/proj/qwen38-inference/env-vllm032/bin/python benchmark_efficiency.py --root_path data --emb_root embeddings --fm 1 \
  --out bench/efficiency_ETTm1_q51.json > logs/bench_q51.log 2>&1
rm -f bench/BENCH_RUNNING
echo "[$(date)] benchmark done" >> logs/bench_q51.log
