#!/bin/zsh
# Second collector: shards 0-2 of 4 (dev, Q-51, Q-52) and shard 3 of 4 rebuilt as shards 6+7 of 8 on dev and Q-51
# (the A40 was too slow). Everything is gathered on Training, merged there and copied to Explore.
S="-o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no -o ConnectTimeout=30 -T"
D=/mnt/proj/pact-code/embeddings/Traffic
fetch() {  # host part
  until ssh ${=S} $1 "grep -q '\"finished\": true' $D/$2.progress" < /dev/null 2>/dev/null; do sleep 300; done
  ssh ${=S} $1 "cd $D && tar cf - $2 $2.progress $2.meta.json" < /dev/null 2>/dev/null > /tmp/traffic_part.tar
  ssh ${=S} Training "mkdir -p $D && cd $D && tar xf -" < /tmp/traffic_part.tar 2>/dev/null
  rm -f /tmp/traffic_part.tar
  echo "$(date +%H:%M) $2 from $1: $(ssh ${=S} Training "stat -c %s $D/$2" < /dev/null 2>/dev/null)"
}
ssh ${=S} Training "rm -f $D/gpt2-norm_L96.part2of4.npy*" < /dev/null 2>/dev/null  # possibly truncated by the first collector
fetch Q-52 gpt2-norm_L96.part2of4.npy
fetch dev gpt2-norm_L96.part6of8.npy
fetch Q-51 gpt2-norm_L96.part7of8.npy
ssh ${=S} Training 'source /mnt/proj/pact-code/env.sh && $PY - <<"PY"
import json, numpy as np
from numpy.lib.format import open_memmap
d = "embeddings/Traffic/gpt2-norm_L96."
a, b = (np.load(d + f"part{i}of8.npy", mmap_mode="r") for i in (6, 7))
out = open_memmap(d + "part3of4.npy", mode="w+", dtype=np.float16, shape=(a.shape[0] + b.shape[0],) + a.shape[1:])
out[:a.shape[0]] = a
out[a.shape[0]:] = b
out.flush()
json.dump(dict(done=int(out.shape[0]), finished=True), open(d + "part3of4.npy.progress", "w"))
json.dump(json.load(open(d + "part6of8.npy.meta.json")), open(d + "part3of4.npy.meta.json", "w"))
print("part3of4", out.shape)
PY
$PY build_embeddings.py --dataset Traffic --root_path data --seq_len 96 --model hf_models/gpt2 --value_norm window --num_shards 4 --merge 1 2>&1 | tail -1
rm -f embeddings/Traffic/gpt2-norm_L96.part6of8.npy* embeddings/Traffic/gpt2-norm_L96.part7of8.npy*' < /dev/null
ssh ${=S} Training "cd $D && tar cf - gpt2-norm_L96.npy gpt2-norm_L96.json" < /dev/null 2>/dev/null > /tmp/traffic_merged.tar
ssh ${=S} Explore "mkdir -p $D.tmp && cd $D.tmp && tar xf - && mkdir -p $D && mv $D.tmp/* $D/ && rmdir $D.tmp && ls -la $D" < /tmp/traffic_merged.tar 2>/dev/null
rm -f /tmp/traffic_merged.tar
echo "$(date +%H:%M) merged store on Training and Explore"
