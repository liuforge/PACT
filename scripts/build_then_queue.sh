#!/bin/bash
# Build one or more embedding stores (each argument string before "--" is passed to build_embeddings.py; a store
# that already exists is skipped by build_embeddings.py itself), then run a job file.
#   scripts/build_then_queue.sh "<build args>" ["<build args>" ...] -- <job file> <gpu> <per_gpu>
source /mnt/proj/pact-code/env.sh
builds=()
while [ "$1" != "--" ]; do builds+=("$1"); shift; done
shift
J=$1 G=${2:-0} PER=${3:-3}
for b in "${builds[@]}"; do
  echo "[$(date '+%F %T')] build: $b"
  CUDA_VISIBLE_DEVICES=$G $PY build_embeddings.py $b --root_path data --seq_len 96 > logs/emb_$(basename $J .txt)_$(echo $b | md5sum | cut -c1-6).log 2>&1 \
    || { echo "build failed: $b"; exit 1; }
done
echo "[$(date '+%F %T')] stores done"
OMP_NUM_THREADS=4 $PY tools/queue.py $J --gpus $G --per_gpu $PER > logs/queue_$(basename $J .txt).log 2>&1
echo "[$(date '+%F %T')] runs done"
