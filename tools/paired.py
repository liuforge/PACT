"""Paired comparisons (Wilcoxon signed-rank over dataset x horizon x seed cells) between a reference tag and
other tags on test MSE, e.g. component ablations or LLM controls.

  python tools/paired.py --root results --ref abl_full --prefix abl_
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--ref", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--metric", default="mse")
    a = p.parse_args()
    R = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(a.root, "*", a.prefix + "*", "*", "results.json")):
        r = json.load(open(f))
        c = r["config"]
        R[(c["dataset"], c["pred_len"], c["seed"])][c["tag"]] = r["test"][a.metric]
    tags = sorted({t for row in R.values() for t in row} - {a.ref})
    for scope in sorted({k[0] for k in R}) + ["ALL"]:
        cells = [k for k in R if (scope == "ALL" or k[0] == scope) and a.ref in R[k]]
        if not cells:
            continue
        ref = np.array([R[k][a.ref] for k in cells])
        print(f"== {scope}: {a.ref} = {ref.mean():.4f} (n={len(cells)})")
        for t in tags:
            ks = [k for k in cells if t in R[k]]
            if len(ks) < 3:
                continue
            x = np.array([R[k][a.ref] for k in ks])
            y = np.array([R[k][t] for k in ks])
            rel = np.mean((y - x) / x) * 100
            pval = wilcoxon(x, y).pvalue if len(ks) >= 6 and np.any(x != y) else float("nan")
            print(f"   {t:24s} {y.mean():.4f}  {rel:+6.2f}% vs ref  ref better in {(x < y).sum():2d}/{len(ks)}  p={pval:.4f}")


if __name__ == "__main__":
    main()
