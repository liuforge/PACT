#!/bin/bash
# Run follow-up groups once scripts/pipeline.sh <role> has finished:  bash scripts/after_pipeline.sh <role> <datasets> <gpus> <per_gpu>
source /mnt/proj/pact-code/env.sh
role=$1; DS=$2; GPUS=$3; PER=$4
until grep -q "] done" logs/pipeline_$role.log 2>/dev/null; do sleep 300; done
X="--root_path data --num_workers 0"
$PY tools/make_jobs.py --group fewshot --datasets $DS --seeds 2024,2025,2026 --chosen chosen/pact_$role.json --extra "$X" > jobs/fewshot_$role.txt
$PY tools/queue.py jobs/fewshot_$role.txt --gpus $GPUS --per_gpu $PER > logs/queue_fewshot_$role.log 2>&1
echo "[$(date)] follow-ups done"
