#!/bin/bash
# C5: semantic prompts.  bash scripts/c5.sh <datasets> <chosen.json> <per_gpu>
source /mnt/proj/pact-code/env.sh
DS=$1; CH=$2; PER=$3
X="--root_path data --num_workers 0"
for ds in ${DS//,/ }; do
  L=96; [ $ds = ILI ] && L=36
  $PY build_embeddings.py --dataset $ds --root_path data --seq_len $L --model hf_models/gpt2 --prompt variable > logs/emb_${ds}_varstatic.log 2>&1
  [ -f embeddings/$ds/gpt2-var-norm_L$L.json ] || $PY build_embeddings.py --dataset $ds --root_path data --seq_len $L --model hf_models/gpt2 --prompt variable_values --value_norm window --batch_size 256 > logs/emb_${ds}_varnorm.log 2>&1
done
$PY tools/make_jobs.py --group llm --datasets $DS --seeds 2024,2025,2026 --chosen $CH --extra "$X" | grep -E "tag llm_var" > jobs/c5_$(hostname | cut -c1-12).txt
$PY tools/queue.py jobs/c5_$(hostname | cut -c1-12).txt --gpus 0 --per_gpu $PER > logs/queue_c5.log 2>&1
echo "[$(date)] C5 done"
