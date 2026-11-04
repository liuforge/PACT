#!/bin/zsh
# Generic helper queue: SRC runs DST's job file in reverse; SRC's new results.json are copied to DST every
# 10 minutes so DST's forward queue skips them (train.py skips runs whose results.json exists).
#   scripts/relay.sh SRC DST jobs/<name>.txt <per_gpu> <gpus>
SRC=$1 DST=$2 JOBS=$3 PER=${4:-3} GPUS=${5:-0}
S="-o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no -T"
PY=/mnt/proj/qwen38-inference/env-vllm032/bin/python
stem=$(basename $JOBS .txt)_rev
if [ -z "$SYNC_ONLY" ]; then
ssh ${=S} $DST "cat /mnt/proj/pact-code/$JOBS" < /dev/null 2>/dev/null | tail -r > /tmp/$stem.txt
[ -s /tmp/$stem.txt ] || { echo "empty job file"; exit 1; }
ssh ${=S} $SRC "cat > /mnt/proj/pact-code/jobs/$stem.txt" < /tmp/$stem.txt 2>/dev/null
ssh ${=S} $SRC "cd /mnt/proj/pact-code && touch logs/$stem.start && OMP_NUM_THREADS=4 nohup $PY tools/queue.py jobs/$stem.txt \
  --gpus $GPUS --per_gpu $PER > logs/queue_$stem.log 2>&1 < /dev/null &" < /dev/null 2>/dev/null
fi
echo "$(date +%H:%M) $SRC started $stem"
for i in $(seq 1 300); do
  [ $i -gt 1 -o -z "$SYNC_ONLY" ] && sleep 600
  # via a local file: a direct ssh-to-ssh pipe silently transferred nothing in some runs
  ssh ${=S} $SRC "cd /mnt/proj/pact-code && find results -name results.json -newer logs/$stem.start | tar cf - -T -" \
    < /dev/null > /tmp/$stem.sync.tar 2>/dev/null
  ssh ${=S} $DST 'cd /mnt/proj/pact-code && tar xf - --skip-old-files 2>/dev/null; echo "$(date +%H:%M) synced"' \
    < /tmp/$stem.sync.tar 2>/dev/null | tail -1
  echo "  $(tar tf /tmp/$stem.sync.tar 2>/dev/null | grep -c results.json) results in archive"
  ssh ${=S} $SRC "ps aux | grep -q '[j]obs/$stem.txt'" < /dev/null 2>/dev/null || break
done
echo "$(date +%H:%M) $stem finished"
