#!/bin/zsh
# Qwen3 ECL prompt store: shards 0,1 of 3 (dev, Q-52) and shard 2 of 3 rebuilt as shards 4+5 of 6 on Explore.
# Everything is merged on Explore, which then runs the two ECL anchor runs (H 96/336, seed 2024).
S="-o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no -o ConnectTimeout=30 -T"
D=/mnt/proj/pact-code/embeddings/ECL
P=Qwen3-0.6B-Base-norm_L96
done_on() { ssh ${=S} $1 "grep -q '\"finished\": true' $D/$2.progress" < /dev/null 2>/dev/null; }
for spec in dev:$P.part0of3.npy Q-52:$P.part1of3.npy Explore:$P.part4of6.npy Explore:$P.part5of6.npy; do
  until done_on ${spec%%:*} ${spec##*:}; do sleep 300; done
done
for spec in dev:$P.part0of3.npy Q-52:$P.part1of3.npy; do
  h=${spec%%:*}; f=${spec##*:}
  ssh ${=S} $h "cd $D && tar cf - $f $f.progress $f.meta.json" < /dev/null 2>/dev/null > /tmp/qwen3_part.tar
  ssh ${=S} Explore "cd $D && tar xf -" < /tmp/qwen3_part.tar 2>/dev/null
  echo "$(date +%H:%M) $f from $h: $(ssh ${=S} Explore "stat -c %s $D/$f" < /dev/null 2>/dev/null)"
done
rm -f /tmp/qwen3_part.tar
ssh ${=S} Explore 'cd /mnt/proj/pact-code && P=/mnt/proj/qwen38-inference/env-vllm032/bin/python && $P - <<"PY"
import json, numpy as np
from numpy.lib.format import open_memmap
d = "embeddings/ECL/Qwen3-0.6B-Base-norm_L96."
a, b = (np.load(d + f"part{i}of6.npy", mmap_mode="r") for i in (4, 5))
out = open_memmap(d + "part2of3.npy", mode="w+", dtype=np.float16, shape=(a.shape[0] + b.shape[0],) + a.shape[1:])
out[:a.shape[0]] = a
out[a.shape[0]:] = b
out.flush()
json.dump(dict(done=int(out.shape[0]), finished=True), open(d + "part2of3.npy.progress", "w"))
json.dump(json.load(open(d + "part4of6.npy.meta.json")), open(d + "part2of3.npy.meta.json", "w"))
print("part2of3", out.shape)
PY
$P build_embeddings.py --dataset ECL --root_path data --seq_len 96 --model hf_models/Qwen3-0.6B-Base --value_norm window --num_shards 3 --merge 1 2>&1 | tail -1' < /dev/null
ssh ${=S} Explore "cat > /mnt/proj/pact-code/jobs/ecl_qwen3.txt" < /tmp/ecl_qwen3_jobs.txt
ssh ${=S} Explore 'cd /mnt/proj/pact-code && wc -l < jobs/ecl_qwen3.txt && (OMP_NUM_THREADS=4 setsid nohup /mnt/proj/qwen38-inference/env-vllm032/bin/python tools/queue.py jobs/ecl_qwen3.txt --gpus 0,1 > logs/queue_ecl_qwen3.log 2>&1 < /dev/null &)' < /dev/null
echo "$(date +%H:%M) ECL Qwen3 runs started on Explore"
