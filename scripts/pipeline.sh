#!/bin/bash
# Per-machine PACT pipeline:  bash scripts/pipeline.sh <role>
#   1. prompt embeddings for the role's datasets   2. main1 (hp grid, seed 2024)   3. choose hp on validation
#   4. main2 (seeds 2025/2026) + LLM controls + ablations (+ sweeps)
# Roles own disjoint datasets, so every results.json of a dataset is produced on one machine.
set -u
source /mnt/proj/pact-code/env.sh
role=$1
mkdir -p logs jobs chosen
X="--root_path data --num_workers 0"
case $role in
  dev)      DS="ETTh1,ETTh2,ILI";      GPUS=0;   PER=6; EMB_PROCS=1; LLM_DS="ETTh1,ETTh2,ILI"; ABL_DS="ETTh2,ILI"; ABL_SEEDS=2024,2025,2026; SWEEP_DS="ETTh2" ;;
  explore)  DS="ETTm1,ETTm2,Weather";  GPUS=0,1; PER=4; EMB_PROCS=2; LLM_DS="ETTm2,Weather";   ABL_DS="ETTm2,Weather"; ABL_SEEDS=2024,2025,2026; SWEEP_DS="Weather" ;;
  training) DS="ECL";                  GPUS=0;   PER=3; EMB_PROCS=3; LLM_DS="";                ABL_DS="ECL"; ABL_SEEDS=2024; SWEEP_DS="" ;;
  *) echo "unknown role $role"; exit 1 ;;
esac
L_OF() { [ "$1" = ILI ] && echo 36 || echo 96; }
emb() {  # emb <dataset> <tag-suffix flags...>
  local ds=$1; shift
  local L=$(L_OF $ds)
  $PY build_embeddings.py --dataset $ds --root_path data --seq_len $L --model hf_models/gpt2 --batch_size 256 "$@"
}
emb_sharded() {  # one store over EMB_PROCS processes on the first GPU, then merge
  local ds=$1; shift
  if [ $EMB_PROCS -gt 1 ] && [ $ds = ECL -o $ds = Weather ]; then
    for i in $(seq 0 $((EMB_PROCS - 1))); do emb $ds "$@" --num_shards $EMB_PROCS --shard $i > logs/emb_${ds}_$(echo "$@" | tr -d ' -')_s$i.log 2>&1 & done
    wait
    emb $ds "$@" --num_shards $EMB_PROCS --merge 1 > logs/emb_${ds}_merge.log 2>&1
  else
    emb $ds "$@" > logs/emb_${ds}_$(echo "$@" | tr -d ' -').log 2>&1
  fi
}
echo "[$(date)] embeddings"
for ds in ${DS//,/ }; do
  [ -f embeddings/$ds/gpt2-norm_L$(L_OF $ds).json ] || emb_sharded $ds --value_norm window
done
for ds in ${LLM_DS//,/ }; do  # controls: random-init GPT-2 (normalised prompt) and raw-value prompt
  [ -f embeddings/$ds/gpt2-randinit-norm_L$(L_OF $ds).json ] || emb_sharded $ds --value_norm window --random_init 1
  [ -f embeddings/$ds/gpt2_L$(L_OF $ds).json ] || emb_sharded $ds
done
echo "[$(date)] main1"
$PY tools/make_jobs.py --group main1 --datasets $DS --extra "$X" > jobs/main1_$role.txt
$PY tools/queue.py jobs/main1_$role.txt --gpus $GPUS --per_gpu $PER > logs/queue_main1_$role.log 2>&1
$PY tools/choose.py --root results --prefix pact__ > chosen/pact_$role.json
echo "[$(date)] main2 + controls + ablations"
C="--chosen chosen/pact_$role.json --extra $X"
{
  $PY tools/make_jobs.py --group main2 --datasets $DS --seeds 2024,2025,2026 --chosen chosen/pact_$role.json --extra "$X"
  [ -n "$LLM_DS" ] && $PY tools/make_jobs.py --group llm --datasets $LLM_DS --seeds 2024,2025,2026 --chosen chosen/pact_$role.json --extra "$X"
  $PY tools/make_jobs.py --group ablation --datasets $ABL_DS --seeds $ABL_SEEDS --chosen chosen/pact_$role.json --extra "$X"
  if [ -n "$SWEEP_DS" ]; then
    $PY tools/make_jobs.py --group kproto --datasets $SWEEP_DS --seeds 2024,2025,2026 --chosen chosen/pact_$role.json --extra "$X"
    $PY tools/make_jobs.py --group kernel --datasets $SWEEP_DS --seeds 2024,2025,2026 --chosen chosen/pact_$role.json --extra "$X"
  fi
} > jobs/stage2_$role.txt
$PY tools/queue.py jobs/stage2_$role.txt --gpus $GPUS --per_gpu $PER > logs/queue_stage2_$role.log 2>&1
echo "[$(date)] done"
