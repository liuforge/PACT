#!/bin/bash
# Phase 1 on Q-52 (RTX 5090 32GB): ECL embeddings (two processes share the GPU), then TimeCMA and v5 on ECL.
source /mnt/proj/pact-code/env.sh
mkdir -p logs jobs
for i in 0 1; do
  $PY build_embeddings.py --dataset ECL --root_path data --seq_len 96 --model hf_models/gpt2 --tag gpt2 --batch_size 256 --num_shards 2 --shard $i > logs/emb_ECL_gpt2_shard$i.log 2>&1 &
done
wait
$PY build_embeddings.py --dataset ECL --root_path data --seq_len 96 --model hf_models/gpt2 --tag gpt2 --num_shards 2 --merge 1 > logs/emb_ECL_gpt2_merge.log 2>&1
X="--root_path data --num_workers 4"
{ $PY tools/make_jobs.py --group timecma --datasets ECL --seeds 2024,2025,2026 --extra "$X";
  $PY tools/make_jobs.py --group v5 --datasets ECL --seeds 2024,2025,2026 --extra "$X"; } > jobs/baselines_q52.txt
$PY tools/queue.py jobs/baselines_q52.txt --gpus 0 --per_gpu 2 > logs/queue_baselines.log 2>&1
