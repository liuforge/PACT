#!/bin/bash
# Keeps a host busy with useful backlog work (servers idle for 24 h are reclaimed).
# Every 30 minutes: if no project job (train.py / eval_foundation.py / build_*embeddings.py) is running, run the next
# backlog job file with tools/queue.py; finished jobs are skipped by train.py (results.json exists). Exits when every
# backlog file has been processed. Only project processes are inspected; nothing else on the host is touched.
#   scripts/idle_guard.sh <per_gpu> <gpus> jobs/backlog_a.txt [jobs/backlog_b.txt ...]
#   (PY can be overridden for hosts with another Python environment)
cd /mnt/proj/pact-code
[ -f env.sh ] && source env.sh
PY=${PY_GUARD:-${PY:-/mnt/proj/qwen38-inference/env-vllm032/bin/python}}
PER=$1 GPUS=$2
shift 2
log=logs/idle_guard.log
busy() { pgrep -f "python[^ ]* (train\.py|eval_foundation\.py|build_(ts_)?embeddings\.py)" > /dev/null; }
gpu_ok() { nvidia-smi -L > /dev/null 2>&1; }  # e.g. a pod that lost GPU access: wait instead of failing the queue
echo "$(date '+%F %T') guard started: $*" >> $log
for jf in "$@"; do
  while busy || ! gpu_ok; do sleep 1800; done
  [ -s "$jf" ] || { echo "$(date '+%F %T') skip missing $jf" >> $log; continue; }
  stem=$(basename "$jf" .txt)
  echo "$(date '+%F %T') idle -> starting $jf" >> $log
  OMP_NUM_THREADS=4 $PY tools/queue.py "$jf" --gpus "$GPUS" --per_gpu "$PER" > logs/queue_${stem}.log 2>&1
  echo "$(date '+%F %T') finished $jf: $(tail -n 1 logs/queue_${stem}.log)" >> $log
done
echo "$(date '+%F %T') guard done (backlog exhausted)" >> $log
