"""Sensitivity to the number of prototypes K (tags K<k>) and the Gaussian kernel width (tags ks<k>).

Test MSE averaged over the screened horizons and indexed to the default setting (K=16, k=25) of each dataset,
so that datasets share one axis; error bars are the standard deviation over seeds.

  python tools/plot_sensitivity.py --root <results>... --out figures/sensitivity.pdf
"""
import argparse
import collections
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 of the reference palette (validated all-pairs)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def collect(roots, prefix):
    v = collections.defaultdict(lambda: collections.defaultdict(list))  # v[ds][value] -> per-seed mean over H
    per = collections.defaultdict(list)
    for root in roots:
        for f in glob.glob(os.path.join(root, "*", f"{prefix}*", "*", "results.json")):
            r = json.load(open(f))
            c = r["config"]
            tag = c["tag"]
            if not tag[len(prefix):].isdigit():
                continue
            per[(c["dataset"], int(tag[len(prefix):]), c["seed"])].append(r["test"]["mse"])
    for (ds, val, seed), xs in per.items():
        v[ds][val].append(np.mean(xs))
    return v


def panel(ax, data, default, xlabel):
    for i, ds in enumerate(sorted(data)):
        vals = sorted(data[ds])
        if default not in data[ds]:
            continue
        base = np.mean(data[ds][default])
        m = [np.mean(data[ds][x]) / base for x in vals]
        s = [np.std(data[ds][x]) / base if len(data[ds][x]) > 1 else 0 for x in vals]
        ax.errorbar(vals, m, yerr=s, color=SERIES[i % 3], lw=2, marker="o", ms=5, capsize=3, label=ds)
        ax.annotate(ds, (vals[-1], m[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
    ax.axhline(1.0, color=GRID, lw=1, zorder=0)
    ax.set_xscale("log", base=2)
    ticks = sorted({x for ds in data for x in data[ds]})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(xlabel, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    panel(axes[0], collect(a.root, "K"), 16, "number of prototypes $K$")
    panel(axes[1], collect(a.root, "ks"), 25, "kernel width $k$")
    axes[0].set_ylabel("test MSE / default", color=MUTED)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=200, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
