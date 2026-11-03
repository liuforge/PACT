"""Validation-only report of the structural screening (tags scr_*): test metrics are deliberately not shown.

  python tools/screen_report.py --root results
"""
import argparse
import collections
import glob
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    a = p.parse_args()
    R = collections.defaultdict(dict)
    D = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(a.root, "*", "scr_*", "*", "results.json")):
        r = json.load(open(f))
        c = r["config"]
        R[(c["dataset"], c["pred_len"])][c["tag"][4:]] = r["val"]["mse"]
        D[(c["dataset"], c["pred_len"])][c["tag"][4:]] = r.get("diag", {})
    variants = sorted({v for row in R.values() for v in row})
    print("validation MSE relative to 'base' (lower is better)")
    print("ds-H".ljust(12) + "".join(v[:14].rjust(15) for v in variants))
    ratios = collections.defaultdict(list)
    for k in sorted(R):
        row = R[k]
        if "base" not in row:
            continue
        cells = []
        for v in variants:
            if v in row:
                ratios[v].append(row[v] / row["base"])
                cells.append(f"{row[v] / row['base']:.4f}".rjust(15))
            else:
                cells.append("-".rjust(15))
        print(f"{k[0]}-{k[1]}".ljust(12) + "".join(cells))
    print("mean".ljust(12) + "".join((f"{np.mean(ratios[v]):.4f}" if ratios[v] else "-").rjust(15) for v in variants))
    print("wins".ljust(12) + "".join(f"{sum(x < 1 for x in ratios[v])}/{len(ratios[v])}".rjust(15) for v in variants))
    print("\\ninternal diagnostics (mean over cells)")
    keys = sorted({kk for row in D.values() for d in row.values() for kk in d})
    for v in variants:
        vals = {kk: np.mean([D[k][v][kk] for k in D if v in D[k] and kk in D[k][v]]) for kk in keys}
        print(f"  {v:22s} " + " ".join(f"{kk}={x:.3f}" for kk, x in vals.items() if not np.isnan(x)))


if __name__ == "__main__":
    main()
