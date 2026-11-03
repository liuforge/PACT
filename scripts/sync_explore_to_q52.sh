#!/bin/zsh
# Explore runs stage2_rest3 part 2 in reverse; copy its finished results.json to Q-52 so that Q-52's forward
# queue skips them (train.py skips runs whose results.json exists). Runs locally as a relay, every 10 minutes.
S="-o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no -T"
for i in $(seq 1 72); do
  ssh ${=S} Explore 'cd /mnt/proj/pact-code && find results/ETTm2 results/Weather -name results.json -newer logs/p2rev.start | tar cf - -T -' 2>/dev/null \
    | ssh ${=S} Q-52 'cd /mnt/proj/pact-code && tar xf - --skip-old-files 2>/dev/null; echo "$(date +%H:%M) synced"' 2>/dev/null | tail -1
  ssh ${=S} Explore 'ps aux | grep -q "[s]tage2_rest3_p2rev"' 2>/dev/null || break  # Explore queue finished
  sleep 600
done
