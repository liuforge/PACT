#!/bin/bash
# Q-51: L=512 comparison with the final structure, after the C1/C5 queues of this machine.
source /mnt/proj/pact-code/env.sh
X="--root_path data --num_workers 0"
DS="ETTh1,ETTh2,ETTm1,ETTm2,Weather"
until grep -q "C5 done" logs/c5.log 2>/dev/null && grep -q "^done" logs/queue_timecma_norm.log 2>/dev/null; do sleep 120; done
$PY tools/make_jobs.py --group long --datasets $DS --extra "$X" > jobs/long_v2.txt
$PY tools/queue.py jobs/long_v2.txt --gpus 0 --per_gpu 6 > logs/queue_long_v2.log 2>&1
$PY tools/choose.py --root results --prefix L512_pact__ > chosen/long.json
{ $PY tools/make_jobs.py --group long2 --datasets $DS --seeds 2024,2025,2026 --chosen chosen/long.json --extra "$X"
  for s in 2025 2026; do $PY tools/make_jobs.py --group long --datasets $DS --extra "$X" | grep -v "L512_pact__" | sed "s/--seed 2024/--seed $s/"; done
} > jobs/long2_v2.txt
$PY tools/queue.py jobs/long2_v2.txt --gpus 0 --per_gpu 6 > logs/queue_long2_v2.log 2>&1
echo "[$(date)] long done"
