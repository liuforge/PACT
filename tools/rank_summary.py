import sys, glob, numpy as np
sys.path.insert(0, "tools")
from make_tables import load, MAIN_MODELS, DATASETS, horizons, mean
R = {}
for root in glob.glob("../remote_results/sync_*/results"):
    for k, v in load(root).items():
        seen = {r["seed"] for r in R.get(k, [])}
        R.setdefault(k, []).extend(r for r in v if r["seed"] not in seen)
trained = [t for t, _ in MAIN_MODELS if not t.startswith("zs_")]
names = dict(MAIN_MODELS)
for key in ("mse", "mae"):
    ranks = {t: [] for t in trained}; gaps = {t: [] for t in trained}; top3 = {t: 0 for t in trained}
    for ds in DATASETS:
        for H in horizons(ds):
            if not all((ds, t, H) in R for t in trained): continue
            v = {t: mean(R[(ds, t, H)], key) for t in trained}
            order = sorted(trained, key=lambda t: v[t]); best = v[order[0]]
            for i, t in enumerate(order):
                ranks[t].append(i + 1); gaps[t].append(100 * (v[t] / best - 1)); top3[t] += i < 3
    print(key.upper(), "cells", len(ranks["pact"]))
    for t in sorted(trained, key=lambda t: np.mean(ranks[t])):
        print(f"  {names[t]:9s} mean rank {np.mean(ranks[t]):.2f}  median gap {np.median(gaps[t]):5.1f}%  mean gap {np.mean(gaps[t]):5.1f}%  top-3 {top3[t]}")
