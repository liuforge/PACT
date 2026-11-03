#!/bin/bash
# Training (A800), ECL: wait for the reduced grid, choose (192 <- 96, 720 <- 336), then stage 2.
source /mnt/proj/pact-code/env.sh
X="--root_path data --num_workers 0"
until grep -q "^done" logs/queue_main1_ecl.log 2>/dev/null; do sleep 120; done
while ps -eo args | awk '$2 == "train.py"' | grep -q -- "--tag pact__"; do sleep 120; done
$PY tools/choose.py --root results --prefix pact__ > chosen/pact_training_2h.json
$PY - <<'PY'
import json
c = json.load(open("chosen/pact_training_2h.json"))
e = c["ECL"]; e["192"] = e["96"]; e["720"] = e["336"]
json.dump(c, open("chosen/pact_training.json", "w"), indent=1)
PY
$PY build_embeddings.py --dataset ECL --root_path data --seq_len 96 --model hf_models/gpt2 --prompt variable > logs/emb_ECL_varstatic.log 2>&1
CH="--chosen chosen/pact_training.json"
{
  $PY tools/make_jobs.py --group main2 --datasets ECL --seeds 2024,2025,2026 $CH --extra "$X"
  $PY tools/make_jobs.py --group ablation --datasets ECL --seeds 2024 $CH --extra "$X"
  $PY tools/make_jobs.py --group llm --datasets ECL --seeds 2024 $CH --extra "$X" | grep -E "tag llm_(gpt2|shuffled|learned|none|varstatic) "
} > jobs/stage2_ecl.txt
$PY tools/queue.py jobs/stage2_ecl.txt --gpus 0 --per_gpu 3 > logs/queue_stage2_ecl.log 2>&1
echo "[$(date)] ECL stage 2 done"
