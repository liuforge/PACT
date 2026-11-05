"""Forecast showcase: input, ground truth and prediction of each model on one fixed test window per dataset.

Input: the small npz files written by tools/extract_viz.py (middle test window, last variate; H = 96).
  python tools/plot_forecasts.py --dir viz_preds --datasets ETTm2,Weather,ECL --out ../revision/figures/forecasts.pdf
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from palette import C1, C3, GRID, INK, MUTED, apply_style  # noqa: E402

MODELS = [("pact", "PACT (ours)"), ("timecma", "TimeCMA"), ("itransformer", "iTransformer"), ("patchtst", "PatchTST")]
NVARS = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7, "Weather": 21, "ECL": 321, "Traffic": 862, "ILI": 7}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--datasets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--context", type=int, default=48, help="input steps shown before the forecast")
    a = p.parse_args()
    dss = a.datasets.split(",")
    apply_style(plt, 7)
    fig, axes = plt.subplots(len(dss), len(MODELS), figsize=(6.9, 1.32 * len(dss) + 0.25), sharex=True, sharey="row",
                             squeeze=False)
    for r, ds in enumerate(dss):
        for c, (m, name) in enumerate(MODELS):
            ax = axes[r, c]
            f = os.path.join(a.dir, f"{ds}_{m}.npz")
            if not os.path.exists(f):
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", color=MUTED)
                continue
            z = np.load(f)
            x, y, pr = z["x"][-a.context:], z["y"], z["p"]
            tx, ty = np.arange(-len(x), 0), np.arange(len(y))
            ax.axvspan(0, len(y), color="#F4F3F5", zorder=0, lw=0)
            ax.plot(tx, x, color=MUTED, lw=0.8, alpha=0.7)
            ax.plot(ty, y, color=INK, lw=0.9, label="ground truth")
            ax.plot(ty, pr, color=C3 if m == "pact" else C1, lw=1.2, label="prediction")
            w = float(z["window_mse"])
            row_best = min(float(np.load(os.path.join(a.dir, f"{ds}_{mm}.npz"))["window_mse"]) for mm, _ in MODELS
                           if os.path.exists(os.path.join(a.dir, f"{ds}_{mm}.npz")))
            ax.text(0.97, 0.95, f"MSE {w:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.3,
                    color=INK, fontweight="bold" if w == row_best else "normal",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            if r == 0:
                ax.set_title(name, fontsize=7.5, color=C3 if m == "pact" else INK,
                             fontweight="bold" if m == "pact" else "normal")
            if c == 0:
                ax.set_ylabel(f"{ds} ({NVARS.get(ds, '?')})", fontsize=7, color=INK)
            ax.grid(True, color=GRID, lw=0.4)
            ax.tick_params(colors=MUTED, labelsize=5.8, length=2)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=MUTED, lw=0.8), Line2D([], [], color=INK, lw=0.9), Line2D([], [], color=C3, lw=1.2),
         Line2D([], [], color=C1, lw=1.2)]
    fig.legend(h, ["input", "ground truth", "PACT", "baseline"], loc="lower center", ncol=4, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.005))
    fig.supxlabel("time step (forecast starts at 0; shaded: horizon $H=96$; MSE over all variates of the window)",
                  fontsize=6.5, color=MUTED, y=0.045)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.4, w_pad=0.3)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=200, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
