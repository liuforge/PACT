#!/bin/bash
# PACT on an added high-dimensional benchmark (Solar / PEMS08), same compute-reduced protocol as ECL and Traffic:
# bf16 autocast, at most 20 epochs, patience 5; grid over lr on the first and third horizon with d_model 128 and the
# BDCA FFN (chosen on ECL); the second horizon follows the first and the fourth follows the third; main table with
# three seeds; ablation (BDCA, one-way attention, PDPL) and anchor controls (none, learned, mismatched) on the two
# grid horizons with three seeds; the full model is the selected configuration.
#   scripts/highdim_pact.sh <dataset> <per_gpu>
set -u
D=$1 PER=${2:-3}
source /mnt/proj/pact-code/env.sh
cd /mnt/proj/pact-code
X="--root_path data --num_workers 0"
E="--amp 1 --epochs 20 --patience 5"
log() { echo "[$(date '+%F %T')] $*"; }
if [ ! -f embeddings/$D/gpt2-norm_L96.json ]; then
  log "building gpt2-norm store"
  $PY build_embeddings.py --dataset $D --root_path data --seq_len 96 --model hf_models/gpt2 --value_norm window \
    --batch_size 256 > logs/emb_${D}_gpt2norm.log 2>&1 || { log "embedding failed"; exit 1; }
fi
read H1 H2 H3 H4 <<< $($PY -c "import sys; sys.path.insert(0,'tools'); from make_jobs import EXTRA_DATASETS as E; print(*E['$D']['H'])")
log "grid on H=$H1,$H3"
$PY tools/make_jobs.py --group main1 --datasets $D --extra "$X" | grep -E -- "--pred_len ($H1|$H3) " \
  | grep -- "--d_model 128" | grep -- "--bdca_ffn 1" | sed "s/\$/ $E/" > jobs/main1_$D.txt
$PY tools/queue.py jobs/main1_$D.txt --gpus 0 --per_gpu $PER > logs/queue_main1_$D.log 2>&1
$PY tools/choose.py --root results --prefix pact__ > chosen/pact_all_$D.tmp.json
$PY - <<PY
import json
c = json.load(open("chosen/pact_all_$D.tmp.json"))["$D"]
c["$H2"] = c["$H1"]; c["$H4"] = c["$H3"]
json.dump({"$D": c}, open("chosen/pact_$D.json", "w"), indent=1)
print("chosen", c)
PY
CH="--chosen chosen/pact_$D.json"
{
  $PY tools/make_jobs.py --group main2 --datasets $D --seeds 2024,2025,2026 $CH --extra "$X"
  $PY tools/make_jobs.py --group main2 --datasets $D --seeds 2025 $CH --extra "$X" | grep -E -- "--pred_len ($H2|$H4) " | sed "s/--seed 2025/--seed 2024/"
  $PY tools/make_jobs.py --group ablation --datasets $D --seeds 2024,2025,2026 $CH --extra "$X" | grep -E -- "--pred_len ($H1|$H3) " | grep -E "tag abl_(no_bdca|xattn|no_pdpl) "
  $PY tools/make_jobs.py --group llm --datasets $D --seeds 2024,2025,2026 $CH --extra "$X" | grep -E -- "--pred_len ($H1|$H3) " | grep -E "tag llm_(none|learned|shuffled) "
} | sed "s/\$/ $E/" > jobs/stage2_$D.txt
log "stage 2: $(wc -l < jobs/stage2_$D.txt) jobs"
$PY tools/queue.py jobs/stage2_$D.txt --gpus 0 --per_gpu $PER > logs/queue_stage2_$D.log 2>&1
log "$D PACT done"
