#!/bin/zsh
# Share Training's ECL stage-2 queue with Explore: once Training has written jobs/stage2_ecl2.txt (after its grid)
# and Explore is free, Explore runs the same jobs in reverse order; finished ECL results.json are relayed to
# Training every 10 minutes so that its forward queue skips them (train.py skips runs whose results.json exists).
S="-o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no -T"
PY=/mnt/proj/qwen38-inference/env-vllm032/bin/python
until ssh ${=S} Training 'test -s /mnt/proj/pact-code/jobs/stage2_ecl2.txt' 2>/dev/null; do sleep 300; done
until ssh ${=S} Explore 'test -s /mnt/proj/pact-code/embeddings/ECL/gpt2-norm_L96.npy && ! ps aux | grep -q "[s]tage2_rest3_p2rev"' 2>/dev/null; do sleep 300; done
ssh ${=S} Training 'cat /mnt/proj/pact-code/jobs/stage2_ecl2.txt' 2>/dev/null | grep -v -- "llm_varstatic" | tail -r \
  | ssh ${=S} Explore "cd /mnt/proj/pact-code && cat > jobs/stage2_ecl2_rev.txt && touch logs/ecl_rev.start && \
      OMP_NUM_THREADS=4 nohup $PY tools/queue.py jobs/stage2_ecl2_rev.txt --gpus 0,1 --per_gpu 2 > logs/queue_stage2_ecl2_rev.log 2>&1 < /dev/null &"
echo "$(date +%H:%M) Explore started $(ssh ${=S} Explore 'wc -l < /mnt/proj/pact-code/jobs/stage2_ecl2_rev.txt' 2>/dev/null) ECL jobs"
for i in $(seq 1 300); do
  sleep 600
  ssh ${=S} Explore 'cd /mnt/proj/pact-code && find results/ECL -name results.json -newer logs/ecl_rev.start | tar cf - -T -' 2>/dev/null \
    | ssh ${=S} Training 'cd /mnt/proj/pact-code && tar xf - --skip-old-files 2>/dev/null; echo "$(date +%H:%M) synced"' 2>/dev/null | tail -1
  ssh ${=S} Explore 'ps aux | grep -q "[s]tage2_ecl2_rev"' 2>/dev/null || break
done
