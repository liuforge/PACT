#!/bin/bash
# usage: scripts/wait_store.sh embeddings/ETTh1/gpt2_L96.npy [...]  -- block until every store is finished
for f in "$@"; do
  until [ -f "$f" ] && [ -f "${f%.npy}.json" ] && { [ ! -f "$f.progress" ] || grep -q '"finished": true' "$f.progress"; }; do sleep 60; done
done
