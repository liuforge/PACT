# usage: requeue.py old_jobs.txt new_jobs.txt  -- drop jobs that are finished or currently running
import os, sys, subprocess, re
old, new = sys.argv[1], sys.argv[2]
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
running = {re.sub(r"^\S+ ", "", l) for l in ps if " train.py " in l and not l.startswith("/bin/sh")}
keep = []
for l in open(old):
    l = l.strip()
    if not l:
        continue
    args = l.split()
    a = dict(zip(args[2::2], args[3::2]))
    d = f"results/{a['--dataset']}/{a['--tag']}/L{a['--seq_len']}_H{a['--pred_len']}_s{a['--seed']}/results.json"
    tail = re.sub(r"^python ", "", l)
    if os.path.exists(d) or tail in running:
        continue
    keep.append(l)
open(new, "w").write("\n".join(keep) + "\n")
print(f"{old}: kept {len(keep)} jobs; {len(running)} train.py running")
