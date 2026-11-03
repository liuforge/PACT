"""Delete downstream results (controls, ablations, sweeps, few-shot) of every (dataset, H) whose chosen
hyper-parameters changed between two chosen.json files, so that they are re-run with the new choice.

  python tools/invalidate.py chosen/old.json chosen/new.json [--root results] [--dry_run]
"""
import argparse
import glob
import json
import os
import re
import shutil

DOWNSTREAM = re.compile(r"^(llm_|abl_|K\d+$|ks\d+$|fs\d+_)")


def norm(c):
    return {(ds, int(h)): (list(v) + [0])[:3] for ds, hs in c.items() for h, v in hs.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--root", default="results")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    old = norm(json.load(open(a.old))) if os.path.exists(a.old) else {}
    new = norm(json.load(open(a.new)))
    changed = [k for k in new if k in old and old[k] != new[k]]
    n = 0
    for ds, H in changed:
        for d in glob.glob(os.path.join(a.root, ds, "*", f"L*_H{H}_s*")):
            tag = d.split(os.sep)[-2]
            if DOWNSTREAM.match(tag):
                n += 1
                if not a.dry_run:
                    shutil.rmtree(d)
    print(f"changed: {[f'{ds}-{H}: {old[(ds, H)]} -> {new[(ds, H)]}' for ds, H in changed]}; removed {n} run dirs")


if __name__ == "__main__":
    main()
