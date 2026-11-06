#!/bin/bash
# Random-initialised GPT-2 control on a high-dimensional dataset: build the store (same window-standardised prompts,
# untrained weights), then run the paired llm_randinit jobs.   scripts/randinit_store_runs.sh <dataset> <job file> <gpu> <per_gpu>
source /mnt/proj/pact-code/env.sh
D=$1 J=$2 G=${3:-0} PER=${4:-3}
if [ ! -f embeddings/$D/gpt2-randinit-norm_L96.json ]; then
  CUDA_VISIBLE_DEVICES=$G $PY build_embeddings.py --dataset $D --root_path data --seq_len 96 --model hf_models/gpt2 \
    --value_norm window --random_init 1 --batch_size 256 > logs/emb_${D}_randinit.log 2>&1 || { echo "embedding failed"; exit 1; }
fi
echo "[$(date '+%F %T')] store done"
OMP_NUM_THREADS=4 $PY tools/queue.py $J --gpus $G --per_gpu $PER > logs/queue_$(basename $J .txt).log 2>&1
echo "[$(date '+%F %T')] runs done"
