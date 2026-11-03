"""Pick, per (dataset, H), the stage-1 hyper-parameters with the lowest seed-2024 validation MSE.

  python tools/choose.py --root results --prefix pact__ > chosen.json
Tags look like '<prefix>lr<lr>_d<d>[_f1]'. Writes {dataset: {H: [lr, d, ffn]}}.
"""
import argparse
import glob
import json
import os
import re
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--prefix", default="pact__")
    p.add_argument("--seed", type=int, default=2024)
    a = p.parse_args()
    best = {}
    pat = re.compile(re.escape(a.prefix) + r"lr([0-9.e-]+)_d(\d+)(_f1)?$")
    for f in glob.glob(os.path.join(a.root, "*", a.prefix + "*", f"*_s{a.seed}", "results.json")):
        r = json.load(open(f))
        c = r["config"]
        m = pat.match(c["tag"])
        if not m:
            continue
        k = (c["dataset"], c["pred_len"])
        v = r["val"]["mse"]
        if k not in best or v < best[k][0]:
            best[k] = (v, m.group(1), int(m.group(2)), 1 if m.group(3) else 0)
    out = {}
    for (ds, H), (v, lr, d, f) in sorted(best.items()):
        out.setdefault(ds, {})[str(H)] = [lr, d, f]
        print(f"{ds}-{H}: lr={lr} d_model={d} ffn={f} (val {v:.4f})", file=sys.stderr)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
