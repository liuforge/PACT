"""Run a file of shell commands over local GPUs, a fixed number of jobs per GPU.

  python tools/queue.py jobs/main.txt --gpus 0,1,2,3 --per_gpu 2

Each command gets CUDA_VISIBLE_DEVICES set and its stdout/stderr written to logs/<line-no>.log.
Finished runs are skipped by train.py itself (results.json exists), so the queue can be restarted
or the same job file can be split across machines with --part i/k.
"""
import argparse
import os
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("--gpus", default="0")
    p.add_argument("--per_gpu", type=int, default=1)
    p.add_argument("--part", default="0/1", help="i/k: take every k-th job starting at i (split over machines)")
    p.add_argument("--log_dir", default="logs")
    a = p.parse_args()
    i, k = (int(x) for x in a.part.split("/"))
    lines = [l.strip() for l in open(a.jobs) if l.strip() and not l.startswith("#")]
    todo = [(n, c) for n, c in enumerate(lines) if n % k == i]
    os.makedirs(a.log_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.jobs))[0]
    slots = [g for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    running = {}  # slot index -> (Popen, n, file)
    failed = []
    print(f"{len(todo)} jobs on slots {slots}", flush=True)
    while todo or running:
        for s in range(len(slots)):
            if s not in running and todo:
                n, c = todo.pop(0)
                f = open(os.path.join(a.log_dir, f"{stem}_{n:04d}.log"), "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=slots[s])
                running[s] = (subprocess.Popen(c, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env), n, f)
        for s, (proc, n, f) in list(running.items()):
            if proc.poll() is not None:
                f.close()
                if proc.returncode != 0:
                    failed.append(n)
                print(f"[{time.strftime('%H:%M:%S')}] job {n} rc={proc.returncode} | left {len(todo)} running {len(running) - 1}",
                      flush=True)
                del running[s]
        time.sleep(2)
    print(f"done; failed jobs: {failed}" if failed else "done; all jobs succeeded")


if __name__ == "__main__":
    main()
