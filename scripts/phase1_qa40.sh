#!/bin/bash
# Phase 1 on Q-a40 (A40 48GB): embeddings for all datasets except ECL, then pilot, TimeCMA and v5 re-runs.
source /mnt/proj/pact-code/env.sh
mkdir -p logs jobs
emb() { $PY build_embeddings.py --dataset $1 --root_path data --seq_len $2 --model hf_models/gpt2 --tag $3 --random_init $4 --batch_size 256 > logs/emb_$1_$3.log 2>&1; }
(
  emb ILI 36 gpt2 0
  for ds in ETTh1 ETTm2 Weather ETTh2 ETTm1; do emb $ds 96 gpt2 0; done
  for ds in ETTh1 ETTh2 ETTm2 Weather; do emb $ds 96 gpt2-randinit 1; done
  emb ILI 36 gpt2-randinit 1
) &
scripts/wait_store.sh embeddings/ETTh1/gpt2_L96.npy embeddings/ETTm2/gpt2_L96.npy embeddings/Weather/gpt2_L96.npy embeddings/ILI/gpt2_L36.npy
X="--root_path data --num_workers 2"
$PY tools/make_jobs.py --group pilot --extra "$X" > jobs/pilot.txt
$PY tools/queue.py jobs/pilot.txt --gpus 0 --per_gpu 4 > logs/queue_pilot.log 2>&1
scripts/wait_store.sh embeddings/ETTh2/gpt2_L96.npy embeddings/ETTm1/gpt2_L96.npy
D=ETTh1,ETTh2,ETTm1,ETTm2,Weather,ILI
{ $PY tools/make_jobs.py --group timecma --datasets $D --seeds 2024,2025,2026 --extra "$X";
  $PY tools/make_jobs.py --group v5 --datasets $D --seeds 2024,2025,2026 --extra "$X"; } > jobs/baselines_qa40.txt
$PY tools/queue.py jobs/baselines_qa40.txt --gpus 0 --per_gpu 4 > logs/queue_baselines.log 2>&1
wait
