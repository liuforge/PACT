#!/bin/bash
# PACT on an added high-dimensional benchmark with the full protocol of the low-dimensional datasets, as a check of
# the compute-reduced protocol: fp32, at most 50 epochs, patience 10, grid over lr x d_model x BDCA FFN on every
# horizon (seed 2024), then the selected configuration with seeds 2025 and 2026. Tags: pactfull__<hp>.
#   scripts/highdim_full.sh <dataset> <per_gpu> [gpus]
set -u
D=$1 PER=${2:-2} G=${3:-0}
source /mnt/proj/pact-code/env.sh
cd /mnt/proj/pact-code
mkdir -p chosen jobs logs
X="--root_path data --num_workers 0"
log() { echo "[$(date '+%F %T')] $*"; }
$PY tools/make_jobs.py --group main1 --datasets $D --extra "$X" | sed "s/--tag pact__/--tag pactfull__/" > jobs/full1_$D.txt
log "grid: $(wc -l < jobs/full1_$D.txt) jobs"
$PY tools/queue.py jobs/full1_$D.txt --gpus $G --per_gpu $PER > logs/queue_full1_$D.log 2>&1
$PY tools/choose.py --root results --prefix pactfull__ > chosen/pactfull_all.tmp.json
$PY - <<PY
import json
c = json.load(open("chosen/pactfull_all.tmp.json"))["$D"]
json.dump({"$D": c}, open("chosen/pactfull_$D.json", "w"), indent=1)
print("chosen", c)
PY
$PY tools/make_jobs.py --group main2 --datasets $D --seeds 2025,2026 --chosen chosen/pactfull_$D.json --extra "$X" \
  | sed "s/--tag pact__/--tag pactfull__/" > jobs/full2_$D.txt
log "seeds: $(wc -l < jobs/full2_$D.txt) jobs"
$PY tools/queue.py jobs/full2_$D.txt --gpus $G --per_gpu $PER > logs/queue_full2_$D.log 2>&1
log "$D full protocol done"
