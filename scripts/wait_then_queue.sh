#!/bin/bash
# Run a job file once a file exists (e.g. a copied embedding store marked complete); then exits.
#   scripts/wait_then_queue.sh <file to wait for> <job file> <per_gpu> <gpus>
source /mnt/proj/pact-code/env.sh
F=$1 J=$2 PER=$3 GPUS=$4
while [ ! -e "$F" ]; do sleep 120; done
echo "[$(date '+%F %T')] $F present; starting $J"
OMP_NUM_THREADS=4 $PY tools/queue.py $J --gpus $GPUS --per_gpu $PER > logs/queue_$(basename $J .txt).log 2>&1
echo "[$(date '+%F %T')] $J finished"
