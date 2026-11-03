#!/bin/bash
# Run one third of the remaining ETTm1/ETTm2/Weather stage-2 jobs:  bash scripts/takeover3.sh <part i/3> [wait_queue_log]
source /mnt/proj/pact-code/env.sh
PART=$1; WAIT=${2:-}
[ -n "$WAIT" ] && until grep -q "^done" "$WAIT" 2>/dev/null; do sleep 120; done
while ps -eo args | awk '$2 == "build_embeddings.py"' | grep -q .; do sleep 60; done   # never two builders on one store
emb() { $PY build_embeddings.py --dataset $1 --root_path data --seq_len 96 --model hf_models/gpt2 --batch_size 256 "${@:2}" >> logs/emb_takeover3.log 2>&1; }
for ds in ETTm1 ETTm2 Weather; do [ -f embeddings/$ds/gpt2-norm_L96.json ] || emb $ds --value_norm window; done
for ds in ETTm2 Weather; do
  [ -f embeddings/$ds/gpt2-randinit-norm_L96.json ] || emb $ds --value_norm window --random_init 1
  [ -f embeddings/$ds/gpt2_L96.json ] || emb $ds
done
$PY tools/queue.py jobs/stage2_rest3.txt --gpus 0 --per_gpu 6 --part $PART > logs/queue_stage2_rest3_p${PART%/*}.log 2>&1
echo "[$(date)] takeover $PART done"
