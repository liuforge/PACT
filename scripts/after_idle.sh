#!/bin/bash
# Runs a command once no project job (train.py / eval_foundation.py / build_*embeddings.py) is running on this host
# for two consecutive checks; optionally first waits for a file to exist.
#   [WAIT_FILE=logs/x.log] scripts/after_idle.sh <command ...>
cd /mnt/proj/pact-code
busy() { pgrep -f "python[^ ]* (train\.py|eval_foundation\.py|build_(ts_)?embeddings\.py)" > /dev/null; }
echo "$(date '+%F %T') waiting: $*" >> logs/after_idle.log
sleep ${DELAY:-600}
if [ -n "${WAIT_FILE:-}" ]; then until [ -e "$WAIT_FILE" ]; do sleep 300; done; sleep 900; fi
while busy || { sleep 120; busy; }; do sleep 600; done
echo "$(date '+%F %T') starting: $*" >> logs/after_idle.log
"$@"
echo "$(date '+%F %T') finished: $*" >> logs/after_idle.log
