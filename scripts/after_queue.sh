#!/bin/bash
# Run a job file once the queue of another job file on this host has finished (chains work without idle gaps).
#   scripts/after_queue.sh <job file to wait for> <job file to run> <per_gpu> <gpus>
source /mnt/proj/pact-code/env.sh
W=$1 J=$2 PER=$3 GPUS=$4
while pgrep -f "tools/queue.py $W" > /dev/null; do sleep 300; done
echo "[$(date '+%F %T')] $W finished; starting $J"
OMP_NUM_THREADS=4 $PY tools/queue.py $J --gpus $GPUS --per_gpu $PER > logs/queue_$(basename $J .txt).log 2>&1
echo "[$(date '+%F %T')] $J finished"
