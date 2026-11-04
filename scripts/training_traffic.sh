#!/bin/bash
# Training (A800), Traffic (862 variates) with the ECL compute-reduced setting: bf16 autocast, at most 20 epochs,
# patience 5; batch 16 (memory); grid over lr on H 96/336 with d_model 128 and the BDCA FFN (both chosen on ECL),
# 192 follows 96 and 720 follows 336; main table with three seeds; ablation (BDCA, one-way attention, PDPL) and
# anchor controls (none, learned, mismatched) on H 96/336 with one seed; the full model is the selected grid run.
# Waits for the merged prompt-embedding store (embeddings/Traffic/gpt2-norm_L96.npy).
source /mnt/proj/pact-code/env.sh
X="--root_path data --num_workers 0"
E="--amp 1 --epochs 20 --patience 5"
until [ -f embeddings/Traffic/gpt2-norm_L96.json ]; do sleep 120; done
$PY tools/make_jobs.py --group main1 --datasets Traffic --extra "$X" | grep -E -- "--pred_len (96|336) " | grep -- "--d_model 128" | grep -- "--bdca_ffn 1" | sed "s/\$/ $E/" > jobs/main1_traffic.txt
$PY tools/queue.py jobs/main1_traffic.txt --gpus 0 --per_gpu 2 > logs/queue_main1_traffic.log 2>&1
$PY tools/choose.py --root results --prefix pact__ > chosen/pact_training_traffic_2h.json
$PY - <<'PY'
import json
c = json.load(open("chosen/pact_training_traffic_2h.json"))
t = c["Traffic"]; t["192"] = t["96"]; t["720"] = t["336"]
json.dump({"Traffic": t}, open("chosen/pact_traffic.json", "w"), indent=1)
PY
CH="--chosen chosen/pact_traffic.json"
{
  $PY tools/make_jobs.py --group main2 --datasets Traffic --seeds 2024,2025,2026 $CH --extra "$X"
  $PY tools/make_jobs.py --group main2 --datasets Traffic --seeds 2025 $CH --extra "$X" | grep -E -- "--pred_len (192|720) " | sed "s/--seed 2025/--seed 2024/"
  $PY tools/make_jobs.py --group ablation --datasets Traffic --seeds 2024 $CH --extra "$X" | grep -E -- "--pred_len (96|336) " | grep -E "tag abl_(no_bdca|xattn|no_pdpl) "
  $PY tools/make_jobs.py --group llm --datasets Traffic --seeds 2024 $CH --extra "$X" | grep -E -- "--pred_len (96|336) " | grep -E "tag llm_(none|learned|shuffled) "
} | sed "s/\$/ $E/" > jobs/stage2_traffic.txt
$PY tools/queue.py jobs/stage2_traffic.txt --gpus 0 --per_gpu 3 > logs/queue_stage2_traffic.log 2>&1
echo "[$(date)] Traffic done"
