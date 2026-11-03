"""Why do raw-value prompts hurt? Level shift of each test window vs. the error added by the raw-value prompt.

For every test window w and variate n, the level shift is |mean of the z-scored look-back window|, i.e. how far
the window's level lies from the training mean in training standard deviations. The raw-value prompt spells this
level out as absolute numbers, whereas the standardised prompt does not. We compare the per-(window, variate) test
MSE of the two otherwise identical runs (tags preds_raw and preds_norm, same seed and configuration).

  python analyze_level_shift.py --root results --root_path data --out level_shift.json
(then: python tools/plot_level_shift.py level_shift.json --out ../revision/figures/level_shift.pdf)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pact.data import load_series  # noqa: E402

EDGES = [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--root_path", default="data")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = []
    for f in sorted(glob.glob(os.path.join(a.root, "*", "preds_norm", "*", "test_preds.npz"))):
        g = f.replace(os.sep + "preds_norm" + os.sep, os.sep + "preds_raw" + os.sep)
        if not os.path.exists(g):
            continue
        cn = json.load(open(os.path.join(os.path.dirname(f), "results.json")))["config"]
        cr = json.load(open(os.path.join(os.path.dirname(g), "results.json")))["config"]
        assert cn["emb_tag"].endswith("norm") and not cr["emb_tag"].endswith("norm"), (cn["emb_tag"], cr["emb_tag"])
        ds, L, H = cn["dataset"], cn["seq_len"], cn["pred_len"]
        n, r = np.load(f), np.load(g)
        assert np.allclose(n["true"], r["true"]), "the two runs must share the test targets"
        s = load_series(ds, a.root_path, L)
        x = s["scaled"]
        b1 = s["border1s"][2]
        W = n["pred"].shape[0]
        shift = np.stack([np.abs(x[b1 + w:b1 + w + L].mean(0)) for w in range(W)])  # [W, N]
        en = ((n["pred"] - n["true"]) ** 2).mean(1)  # [W, N]
        er = ((r["pred"] - r["true"]) ** 2).mean(1)
        tr = x[s["border1s"][0]:s["border2s"][0]]
        train_shift = np.abs(np.stack([tr[i:i + L].mean(0) for i in range(0, len(tr) - L + 1, max(1, L // 4))]))
        bins = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (shift >= lo) & (shift < hi)
            if m.sum() == 0:
                continue
            bins.append(dict(lo=lo, hi=None if np.isinf(hi) else hi, count=int(m.sum()), frac=float(m.mean()),
                             mse_norm=float(en[m].mean()), mse_raw=float(er[m].mean()),
                             rel=float(er[m].mean() / en[m].mean() - 1)))
        out.append(dict(dataset=ds, H=H, seed=cn["seed"], windows=W, variates=int(shift.shape[1]),
                        mse_norm=float(en.mean()), mse_raw=float(er.mean()), rel=float(er.mean() / en.mean() - 1),
                        test_frac_shift_gt1=float((shift > 1).mean()), train_frac_shift_gt1=float((train_shift > 1).mean()),
                        rho_shift_vs_delta=spearman(shift.ravel(), (er - en).ravel()), bins=bins))
        print(f"{ds} H={H}: raw/norm {out[-1]['rel'] * 100:+.1f}% | test windows with shift>1: "
              f"{out[-1]['test_frac_shift_gt1'] * 100:.0f}% (train {out[-1]['train_frac_shift_gt1'] * 100:.0f}%) | "
              f"rho {out[-1]['rho_shift_vs_delta']:+.2f} | " +
              " ".join(f"[{b['lo']},{b['hi']}) {b['rel'] * 100:+.0f}%({b['frac'] * 100:.0f}%)" for b in bins), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
