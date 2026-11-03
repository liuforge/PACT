#!/bin/bash
# Train checkpoints with the chosen configuration and project the prototypes.  bash scripts/viz.sh <datasets> <chosen.json>
source /mnt/proj/pact-code/env.sh
DS=$1; CH=$2
mkdir -p viz
$PY tools/make_jobs.py --group viz --datasets $DS --chosen $CH --extra "--root_path data --num_workers 0" > jobs/viz_$(hostname | cut -c1-12).txt
$PY tools/queue.py jobs/viz_$(hostname | cut -c1-12).txt --gpus 0 --per_gpu 3 > logs/queue_viz.log 2>&1
for ds in ${DS//,/ }; do
  for run in results/$ds/viz/*_s2024; do
    for br in seasonal trend; do $PY extract_prototypes.py --run $run --root_path data --branch $br --out viz/${ds}_${br}.npz >> logs/viz_extract.log 2>&1; done
  done
done
echo "[$(date)] viz done"
