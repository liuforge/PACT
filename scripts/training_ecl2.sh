#!/bin/bash
# Training (A800), ECL with the compute-reduced setting (all PACT runs on ECL):
#   bf16 autocast, batch 32, at most 20 epochs, patience 5; grid on H 96/336 with d_model 128;
#   choice for 192 follows 96 and 720 follows 336; ablations on H 96/336; LLM controls with one seed.
source /mnt/proj/pact-code/env.sh
X="--root_path data --num_workers 0"
E="--amp 1 --batch_size 32 --epochs 20 --patience 5"
$PY tools/make_jobs.py --group main1 --datasets ECL --extra "$X" | grep -E -- "--pred_len (96|336) " | grep -- "--d_model 128" | sed "s/\$/ $E/" > jobs/main1_ecl2.txt
$PY tools/queue.py jobs/main1_ecl2.txt --gpus 0 --per_gpu 3 > logs/queue_main1_ecl2.log 2>&1
$PY tools/choose.py --root results --prefix pact__ > chosen/pact_training_2h.json
$PY - <<'PY'
import json
c = json.load(open("chosen/pact_training_2h.json"))
e = c["ECL"]; e["192"] = e["96"]; e["720"] = e["336"]
json.dump(c, open("chosen/pact_training.json", "w"), indent=1)
PY
[ -f embeddings/ECL/gpt2-var_static.npy ] || $PY build_embeddings.py --dataset ECL --root_path data --seq_len 96 --model hf_models/gpt2 --prompt variable > logs/emb_ECL_varstatic.log 2>&1
CH="--chosen chosen/pact_training.json"
{
  $PY tools/make_jobs.py --group main2 --datasets ECL --seeds 2024,2025,2026 $CH --extra "$X"
  $PY tools/make_jobs.py --group ablation --datasets ECL --seeds 2024 $CH --extra "$X" | grep -E -- "--pred_len (96|336) "
  $PY tools/make_jobs.py --group llm --datasets ECL --seeds 2024 $CH --extra "$X" | grep -E "tag llm_(gpt2|shuffled|learned|none|varstatic) "
} | sed "s/\$/ $E/" > jobs/stage2_ecl2.txt
$PY tools/queue.py jobs/stage2_ecl2.txt --gpus 0 --per_gpu 3 > logs/queue_stage2_ecl2.log 2>&1
echo "[$(date)] ECL done"
