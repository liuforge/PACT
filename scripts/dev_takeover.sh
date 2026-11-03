#!/bin/bash
source /mnt/proj/pact-code/env.sh
emb() { $PY build_embeddings.py --dataset $1 --root_path data --seq_len 96 --model hf_models/gpt2 --batch_size 256 "${@:2}" >> logs/emb_takeover.log 2>&1; }
for ds in ETTm1 ETTm2 Weather; do [ -f embeddings/$ds/gpt2-norm_L96.json ] || emb $ds --value_norm window; done
for ds in ETTm2 Weather; do
  [ -f embeddings/$ds/gpt2-randinit-norm_L96.json ] || emb $ds --value_norm window --random_init 1
  [ -f embeddings/$ds/gpt2_L96.json ] || emb $ds
done
$PY tools/queue.py jobs/stage2_rest.txt --gpus 0 --per_gpu 6 --part 1/2 > logs/queue_stage2_rest_p1.log 2>&1
echo "[$(date)] takeover done"
