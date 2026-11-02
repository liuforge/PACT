#!/bin/bash
# Q-a40: zero-shot foundation-model evaluation (TimesFM-3, Toto-2 22M/313M), contexts 96 and 512.
source /mnt/proj/pact-code/env.sh
export PYTHONPATH=/mnt/proj/pact-code/fm_site:$PYTHONPATH
scripts/wait_store.sh embeddings/{ETTh1,ETTh2,ETTm1,ETTm2,Weather}/gpt2_L96.npy embeddings/ILI/gpt2_L36.npy \
  embeddings/{ETTh1,ETTh2,ETTm2,Weather}/gpt2-randinit_L96.npy embeddings/ILI/gpt2-randinit_L36.npy
: > jobs/fm.txt
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather ILI ECL; do
  bs=32; [ $ds = ECL ] && bs=2
  for ctx in 96 512; do
    for mc in toto2:Toto-2.0-22m toto2:Toto-2.0-313m timesfm3:timesfm-3.0-pytorch; do
      echo "python eval_foundation.py --model ${mc%%:*} --ckpt hf_models/${mc##*:} --dataset $ds --root_path data --context $ctx --batch_size $bs" >> jobs/fm.txt
    done
  done
done
$PY tools/queue.py jobs/fm.txt --gpus 0 --per_gpu 1 > logs/queue_fm.log 2>&1
