#!/bin/zsh
# Collect the four Traffic embedding shards on Training, merge them there, then copy the merged store to Explore.
S="-o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no -o ConnectTimeout=30 -T"
HOSTS=(dev Q-51 Q-52 Q-a40)
D=/mnt/proj/pact-code/embeddings/Traffic
for i in 0 1 2 3; do
  h=${HOSTS[$((i + 1))]}
  p=gpt2-norm_L96.part${i}of4.npy
  until ssh ${=S} $h "grep -q '\"finished\": true' $D/$p.progress" < /dev/null 2>/dev/null; do sleep 300; done
  ssh ${=S} $h "cd $D && tar cf - $p $p.progress $p.meta.json" < /dev/null 2>/dev/null \
    | ssh ${=S} Training "mkdir -p $D && cd $D && tar xf -" 2>/dev/null
  echo "$(date +%H:%M) shard $i from $h: $(ssh ${=S} Training "ls -la $D/$p | awk '{print \$5}'" < /dev/null 2>/dev/null)"
done
ssh ${=S} Training 'source /mnt/proj/pact-code/env.sh && $PY build_embeddings.py --dataset Traffic --root_path data --seq_len 96 --model hf_models/gpt2 --value_norm window --num_shards 4 --merge 1 2>&1 | tail -1' < /dev/null
ssh ${=S} Training "cd $D && tar cf - gpt2-norm_L96.npy gpt2-norm_L96.json" < /dev/null 2>/dev/null \
  | ssh ${=S} Explore "mkdir -p $D.tmp && cd $D.tmp && tar xf - && mkdir -p $D && mv $D.tmp/* $D/ && rmdir $D.tmp && ls -la $D" 2>/dev/null
echo "$(date +%H:%M) merged store on Training and Explore"
