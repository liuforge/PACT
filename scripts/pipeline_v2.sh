#!/bin/bash
# Second pass of the per-machine pipeline after adding the BDCA-FFN switch to the validation grid:
# complete main1 -> re-choose -> invalidate downstream runs whose choice changed -> stage 2 -> few-shot.
set -u
source /mnt/proj/pact-code/env.sh
role=$1
mkdir -p logs jobs chosen
X="--root_path data --num_workers 0"
case $role in
  dev)      DS="ETTh1,ETTh2,ILI";      GPUS=0;   PER=6; LLM_DS="ETTh1,ETTh2,ILI"; ABL_DS="ETTh2,ILI"; ABL_SEEDS=2024,2025,2026; SWEEP_DS="ETTh2"; FS_DS="ETTh1,ETTh2" ;;
  explore)  DS="ETTm1,ETTm2,Weather";  GPUS=0,1; PER=4; LLM_DS="ETTm2,Weather";   ABL_DS="ETTm2,Weather"; ABL_SEEDS=2024,2025,2026; SWEEP_DS="Weather"; FS_DS="ETTm1,ETTm2,Weather" ;;
  training) DS="ECL";                  GPUS=0;   PER=3; LLM_DS="";                ABL_DS="ECL"; ABL_SEEDS=2024; SWEEP_DS=""; FS_DS="" ;;
  *) echo "unknown role $role"; exit 1 ;;
esac
echo "[$(date)] main1 (grid incl. BDCA FFN)"
$PY tools/make_jobs.py --group main1 --datasets $DS --extra "$X" > jobs/main1v2_all_$role.txt
$PY tools/requeue.py jobs/main1v2_all_$role.txt jobs/main1v2_$role.txt   # skip finished and still-running jobs
$PY tools/queue.py jobs/main1v2_$role.txt --gpus $GPUS --per_gpu $PER > logs/queue_main1v2_$role.log 2>&1
# jobs left running by the first pass must finish before choosing
while ps -eo args | awk '$2 == "train.py"' | grep -q -- "--tag pact__"; do sleep 120; done
[ -f chosen/pact_$role.json ] && cp chosen/pact_$role.json chosen/pact_${role}_v1.json
$PY tools/choose.py --root results --prefix pact__ > chosen/pact_$role.json
$PY tools/invalidate.py chosen/pact_${role}_v1.json chosen/pact_$role.json
echo "[$(date)] stage 2"
CH="--chosen chosen/pact_$role.json"
{
  $PY tools/make_jobs.py --group main2 --datasets $DS --seeds 2024,2025,2026 $CH --extra "$X"
  [ -n "$LLM_DS" ] && $PY tools/make_jobs.py --group llm --datasets $LLM_DS --seeds 2024,2025,2026 $CH --extra "$X"
  $PY tools/make_jobs.py --group ablation --datasets $ABL_DS --seeds $ABL_SEEDS $CH --extra "$X"
  if [ -n "$SWEEP_DS" ]; then
    $PY tools/make_jobs.py --group kproto --datasets $SWEEP_DS --seeds 2024,2025,2026 $CH --extra "$X"
    $PY tools/make_jobs.py --group kernel --datasets $SWEEP_DS --seeds 2024,2025,2026 $CH --extra "$X"
  fi
  [ -n "$FS_DS" ] && $PY tools/make_jobs.py --group fewshot --datasets $FS_DS --seeds 2024,2025,2026 $CH --extra "$X"
} > jobs/stage2v2_$role.txt
$PY tools/queue.py jobs/stage2v2_$role.txt --gpus $GPUS --per_gpu $PER > logs/queue_stage2v2_$role.log 2>&1
echo "[$(date)] done"
