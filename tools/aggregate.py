"""Collect results/*/*/*/results.json into mean ± std tables (markdown, csv or LaTeX).

Every number that goes into the paper should come out of this script.

  python tools/aggregate.py --root results                       # all runs, markdown
  python tools/aggregate.py --root results --tags pact,timecma --format latex
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def collect(root, tags=None, metric_scale="norm"):
    rows = defaultdict(list)
    for f in glob.glob(os.path.join(root, "*", "*", "*", "results.json")):
        r = json.load(open(f))
        c = r["config"]
        if tags and c["tag"] not in tags:
            continue
        sfx = "" if metric_scale == "norm" else "_raw"
        key = (c["dataset"], c["tag"], c["seq_len"], c["pred_len"])
        rows[key].append(dict(seed=c["seed"], mse=r["test"]["mse" + sfx], mae=r["test"]["mae" + sfx],
                              val_mse=r["val"]["mse"], params=r["n_params"], epoch_s=r["train_s_per_epoch"],
                              infer_s=r["test"].get("infer_s_per_batch"), mem=r.get("peak_mem_mib")))
    return rows


def select_by_val(rows, seed=2024):
    """Collapse tags '<name>__<hp>' to '<name>': per (dataset, L, H) keep the hp with the lowest validation MSE
    of the selection seed (all seeds of the chosen hp are then reported)."""
    best = {}
    for (ds, tag, L, H), runs in rows.items():
        if "__" not in tag:
            continue
        base = tag.split("__")[0]
        sel = [r["val_mse"] for r in runs if r["seed"] == seed] or [r["val_mse"] for r in runs]
        v = float(np.mean(sel))
        k = (ds, base, L, H)
        if k not in best or v < best[k][0]:
            best[k] = (v, tag)
    out = {k: v for k, v in rows.items() if "__" not in k[1]}
    for (ds, base, L, H), (_, tag) in best.items():
        out[(ds, base, L, H)] = [dict(r, chosen=tag) for r in rows[(ds, tag, L, H)]]
    return out


def summarise(rows):
    out = []
    for (ds, tag, L, H), runs in sorted(rows.items()):
        mse = np.array([r["mse"] for r in runs])
        mae = np.array([r["mae"] for r in runs])
        out.append(dict(dataset=ds, tag=tag, L=L, H=H, n=len(runs), seeds=sorted(r["seed"] for r in runs),
                        mse=mse.mean(), mse_std=mse.std(ddof=1) if len(runs) > 1 else 0.0,
                        mae=mae.mean(), mae_std=mae.std(ddof=1) if len(runs) > 1 else 0.0,
                        params=runs[0]["params"], epoch_s=np.mean([r["epoch_s"] for r in runs])))
    # horizon average per (dataset, tag, L): mean over horizons of the seed means
    groups = defaultdict(list)
    for s in out:
        groups[(s["dataset"], s["tag"], s["L"])].append(s)
    for (ds, tag, L), ss in groups.items():
        if len(ss) > 1:
            out.append(dict(dataset=ds, tag=tag, L=L, H="avg", n=min(s["n"] for s in ss), seeds=[],
                            mse=np.mean([s["mse"] for s in ss]), mse_std=np.nan,
                            mae=np.mean([s["mae"] for s in ss]), mae_std=np.nan,
                            params=None, epoch_s=None))
    order = {"avg": 10 ** 9}
    return sorted(out, key=lambda s: (s["dataset"], s["tag"], s["L"], order.get(s["H"], s["H"])))


def fmt(m, s, digits=3):
    return f"{m:.{digits}f}" if np.isnan(s) else f"{m:.{digits}f}±{s:.{digits}f}"


def tex(m, s, digits=3):
    return f"${m:.{digits}f}$" if np.isnan(s) else f"${m:.{digits}f}_{{\\pm {s:.{digits}f}}}$"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--tags", default=None)
    p.add_argument("--scale", default="norm", choices=["norm", "raw"])
    p.add_argument("--format", default="md", choices=["md", "csv", "latex"])
    p.add_argument("--digits", type=int, default=3)
    p.add_argument("--no_select", action="store_true", help="keep '<name>__<hp>' variants separate")
    a = p.parse_args()
    tags = a.tags.split(",") if a.tags else None
    rows = collect(a.root, None, a.scale)
    if not a.no_select:
        rows = select_by_val(rows)
    if tags:
        rows = {k: v for k, v in rows.items() if k[1] in tags}
    summ = summarise(rows)
    if a.format == "csv":
        w = csv.DictWriter(sys.stdout, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    elif a.format == "md":
        print("| dataset | tag | L | H | n seeds | MSE | MAE |")
        print("|---|---|---|---|---|---|---|")
        for s in summ:
            print(f"| {s['dataset']} | {s['tag']} | {s['L']} | {s['H']} | {s['n']} | "
                  f"{fmt(s['mse'], s['mse_std'], a.digits)} | {fmt(s['mae'], s['mae_std'], a.digits)} |")
    else:
        for s in summ:
            print(f"{s['dataset']} & {s['tag']} & {s['H']} & {tex(s['mse'], s['mse_std'], a.digits)} & "
                  f"{tex(s['mae'], s['mae_std'], a.digits)} \\\\")


if __name__ == "__main__":
    main()
