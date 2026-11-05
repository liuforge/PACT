"""Extract one test window per dataset from the saved predictions of the viz_* runs (for the forecast figure).

The window is fixed by rule, not chosen by error: the middle test window, last variate (OT). The input history is
taken from the z-scored series (same scale as the saved predictions).
  python tools/extract_viz.py --datasets ETTm2,Weather,ECL --models pact,timecma,itransformer,patchtst --out viz_preds
"""
import argparse
import glob
import json
import os

import numpy as np

from pact.data import load_series


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--H", type=int, default=96)
    p.add_argument("--L", type=int, default=96)
    p.add_argument("--out", default="viz_preds")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for ds in a.datasets.split(","):
        series = load_series(ds, "data", a.L)
        scaled, start = series["scaled"], series["border1s"][2]
        for m in a.models.split(","):
            d = f"results/{ds}/viz_{m}/L{a.L}_H{a.H}_s2024"
            f = os.path.join(d, "test_preds.npz")
            if not os.path.exists(f):
                print("missing", f)
                continue
            z = np.load(f)
            pred, true = z["pred"], z["true"]
            i, v = len(pred) // 2, pred.shape[-1] - 1
            x = scaled[start + i:start + i + a.L, v]
            assert np.allclose(scaled[start + i + a.L:start + i + a.L + a.H, v], true[i, :, v], atol=1e-4), "window mismatch"
            r = json.load(open(os.path.join(d, "results.json")))
            np.savez(os.path.join(a.out, f"{ds}_{m}.npz"), x=x, y=true[i, :, v], p=pred[i, :, v],
                     window=i, variate=v, test_mse=r["test"]["mse"], window_mse=float(np.mean((pred[i] - true[i]) ** 2)))
            print(ds, m, "window", i, "test MSE", round(r["test"]["mse"], 4))


if __name__ == "__main__":
    main()
