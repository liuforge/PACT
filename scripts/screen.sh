#!/bin/bash
# After main1 (grid) finishes: choose hp on validation, then screen the structural fixes (validation only).
source /mnt/proj/pact-code/env.sh
role=$1; DS=$2; GPUS=$3; PER=$4
X="--root_path data --num_workers 0"
until grep -q "^done" logs/queue_main1v2_$role.log 2>/dev/null; do sleep 60; done
while ps -eo args | awk '$2 == "train.py"' | grep -q -- "--tag pact__"; do sleep 60; done
[ -f chosen/pact_$role.json ] && cp chosen/pact_$role.json chosen/pact_${role}_v1.json
$PY tools/choose.py --root results --prefix pact__ > chosen/pact_$role.json
$PY tools/make_jobs.py --group screen --datasets $DS --chosen chosen/pact_$role.json --extra "$X" > jobs/screen_$role.txt
$PY tools/queue.py jobs/screen_$role.txt --gpus $GPUS --per_gpu $PER > logs/queue_screen_$role.log 2>&1
echo "[$(date)] screening done"
