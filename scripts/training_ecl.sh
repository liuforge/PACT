#!/bin/bash
# Training (A800): ECL embeddings -> reduced hp grid (H 96/336, d_model 128) -> choose -> structural screening.
# The choice for H=192 follows H=96 and H=720 follows H=336 (ECL runs are expensive).
source /mnt/proj/pact-code/env.sh
X="--root_path data --num_workers 0"
until [ $(grep -l '"finished": true' embeddings/ECL/gpt2-norm_L96.part*of3.npy.progress 2>/dev/null | wc -l) -eq 3 ]; do sleep 120; done
[ -f embeddings/ECL/gpt2-norm_L96.npy ] || $PY build_embeddings.py --dataset ECL --root_path data --seq_len 96 --model hf_models/gpt2 --value_norm window --num_shards 3 --merge 1 > logs/emb_ECL_merge.log 2>&1
$PY tools/make_jobs.py --group main1 --datasets ECL --extra "$X" | grep -E -- "--pred_len (96|336) " | grep -- "--d_model 128" > jobs/main1_ecl.txt
$PY tools/queue.py jobs/main1_ecl.txt --gpus 0 --per_gpu 3 > logs/queue_main1_ecl.log 2>&1
$PY tools/choose.py --root results --prefix pact__ > chosen/pact_training_2h.json
$PY - <<'PY'
import json
c = json.load(open("chosen/pact_training_2h.json"))
e = c["ECL"]; e["192"] = e["96"]; e["720"] = e["336"]
json.dump(c, open("chosen/pact_training.json", "w"), indent=1)
PY
$PY tools/make_jobs.py --group screen --datasets ECL --chosen chosen/pact_training.json --extra "$X" > jobs/screen_training.txt
$PY tools/queue.py jobs/screen_training.txt --gpus 0 --per_gpu 3 > logs/queue_screen_training.log 2>&1
echo "[$(date)] ECL screening done"
