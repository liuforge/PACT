"""Stage 2 of the Time-MMD study: for every (domain, horizon, condition) keep the grid configuration with the lowest
seed-2024 validation MSE and emit its command for the remaining seeds.

  python tools/mmd_stage2.py jobs/mmd_grid.txt --seeds 2025,2026,2027,2028 > jobs/mmd_seeds.txt
"""
import argparse
import json
import os
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument("grid")
    p.add_argument("--results", default="results")
    p.add_argument("--seeds", default="2025,2026,2027,2028")
    a = p.parse_args()
    best = {}
    for line in open(a.grid):
        line = line.strip()
        if not line:
            continue
        ds = re.search(r"--dataset (\S+)", line).group(1)
        L, H = re.search(r"--seq_len (\d+)", line).group(1), re.search(r"--pred_len (\d+)", line).group(1)
        tag = re.search(r"--tag (\S+)", line).group(1)
        f = os.path.join(a.results, ds, tag, f"L{L}_H{H}_s2024", "results.json")
        if not os.path.exists(f):
            continue
        v = json.load(open(f))["val"]["mse"]
        key = (ds, H, tag.split("__")[0])
        if key not in best or v < best[key][0]:
            best[key] = (v, line)
    for (ds, H, cond), (_, line) in sorted(best.items()):
        for s in a.seeds.split(","):
            print(line.replace("--seed 2024", f"--seed {s}"))


if __name__ == "__main__":
    main()
