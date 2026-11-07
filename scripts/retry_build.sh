#!/bin/bash
# Run build_embeddings.py until it succeeds (progress is checkpointed, so a retry resumes); waits 10 minutes
# after a failure such as CUDA OOM on a shared GPU.   scripts/retry_build.sh <gpu> <build args...>
source /mnt/proj/pact-code/env.sh
G=$1; shift
for i in $(seq 1 100); do
  CUDA_VISIBLE_DEVICES=$G $PY build_embeddings.py "$@" && { echo "[$(date '+%F %T')] build ok"; exit 0; }
  echo "[$(date '+%F %T')] attempt $i failed; retry in 10 min"
  sleep 600
done
exit 1
