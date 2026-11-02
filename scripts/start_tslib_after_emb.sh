#!/bin/bash
# Q-a40: start the TSLib baseline queue once this machine's embedding stores are finished.
# (Do not SIGSTOP GPU jobs on these HAMi-virtualised GPUs: a stopped context stalls every other process.)
source /mnt/proj/pact-code/env.sh
scripts/wait_store.sh embeddings/{ETTh1,ETTh2,ETTm1,ETTm2,Weather}/gpt2_L96.npy embeddings/ILI/gpt2_L36.npy \
  embeddings/{ETTh1,ETTh2,ETTm2,Weather}/gpt2-randinit_L96.npy embeddings/ILI/gpt2-randinit_L36.npy
$PY tools/queue.py jobs/tslib_qa40.txt --gpus 0 --per_gpu 2 > logs/queue_tslib.log 2>&1
