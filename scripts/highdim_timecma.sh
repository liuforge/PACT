#!/bin/bash
# TimeCMA on an added high-dimensional benchmark: raw-value prompt store (its own template), then the ECL settings of
# the official scripts with three seeds.
#   scripts/highdim_timecma.sh <dataset> <per_gpu>
set -u
D=$1 PER=${2:-2}
source /mnt/proj/pact-code/env.sh
cd /mnt/proj/pact-code
log() { echo "[$(date '+%F %T')] $*"; }
if [ ! -f embeddings/$D/gpt2_L96.json ]; then
  log "building raw-value gpt2 store"
  $PY build_embeddings.py --dataset $D --root_path data --seq_len 96 --model hf_models/gpt2 --batch_size 256 \
    > logs/emb_${D}_gpt2raw.log 2>&1 || { log "embedding failed"; exit 1; }
fi
$PY tools/make_jobs.py --group timecma --datasets $D --seeds 2024,2025,2026 --extra "--root_path data --num_workers 0" > jobs/timecma_$D.txt
log "TimeCMA: $(wc -l < jobs/timecma_$D.txt) jobs"
$PY tools/queue.py jobs/timecma_$D.txt --gpus 0 --per_gpu $PER > logs/queue_timecma_$D.log 2>&1
log "$D TimeCMA done"
