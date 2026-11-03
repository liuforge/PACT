#!/bin/bash
# Q-a40: retry OOM failures with lower concurrency once the first pass of each queue has finished.
source /mnt/proj/pact-code/env.sh
export PYTHONPATH=/mnt/proj/pact-code/fm_site:$PYTHONPATH
w() { until grep -q "^done" "$1" 2>/dev/null; do sleep 120; done; }
( w logs/queue_fm.log
  sed 's/--batch_size [0-9]*/--batch_size 8/' jobs/fm_timemoe.txt > jobs/fm_timemoe_b8.txt
  $PY tools/queue.py jobs/fm_timemoe_b8.txt --gpus 0 --per_gpu 1 > logs/queue_fm_timemoe_b8.log 2>&1 ) &
( w logs/queue_baselines_r2.log; $PY tools/queue.py jobs/baselines_qa40_r2.txt --gpus 0 --per_gpu 3 > logs/queue_baselines_r3.log 2>&1 ) &
( w logs/queue_tslib_r2.log; $PY tools/queue.py jobs/tslib_qa40_r2.txt --gpus 0 --per_gpu 3 > logs/queue_tslib_r3.log 2>&1 ) &
wait
echo "[$(date)] retries done"
