"""One figure for the main finding: what PACT gains, and which parts matter, as the number of variates grows.

(a) PACT against four baselines, (b) component ablations, (c) anchor sources; every point is the mean relative
change of test MSE over the paired (horizon, seed) cells of one dataset (negative: PACT / the component / the
anchor is better). Datasets are ordered by their number of variates.

  python tools/plot_scaling.py --root <results>... --out ../revision/figures/scaling.pdf
"""
import argparse
import collections
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import ALIASES, load  # noqa: E402

ORDER = [("ETTh1", 7), ("ETTh2", 7), ("ETTm1", 7), ("ETTm2", 7), ("ILI", 7), ("Weather", 21), ("Solar", 137),
         ("PEMS08", 170), ("ECL", 321), ("Traffic", 862)]
from palette import BAND, GRID, INK, MUTED, SERIES as SLOTS, edge, apply_style  # noqa: E402
MARKERS = ["o", "s", "D", "^"]


def cells(R):
    c = collections.defaultdict(dict)
    for (ds, tag, H), runs in R.items():
        for r in runs:
            c[(ds, H, r["seed"])][tag] = r["mse"]
    for v in c.values():
        for tag, alias in ALIASES.items():
            if tag not in v and alias in v:
                v[tag] = v[alias]
    return c


def rel(c, ds, num, den):
    ks = [k for k in c if k[0] == ds and num in c[k] and den in c[k]]
    return 100 * np.mean([c[k][num] / c[k][den] - 1 for k in ks]) if ks else None


def panel(ax, c, series, title, xlabel, floor=None, first=False):
    """Horizontal dot plot: one row per dataset (top: fewest variates), one dodged marker per series."""
    y = np.arange(len(ORDER))[::-1]
    step = 0.62 / max(len(series) - 1, 1)
    have = set()
    for i, (label, num, den) in enumerate(series):
        xs = [rel(c, ds, num, den) for ds, _ in ORDER]
        have |= {j for j, v in enumerate(xs) if v is not None}
        pts = [(v, yi + 0.31 - step * i) for yi, v in zip(y, xs) if v is not None]
        ax.scatter([v if floor is None else max(v, floor) for v, _ in pts], [py for _, py in pts], s=22,
                   marker=MARKERS[i], color=SLOTS[i], edgecolor=edge(SLOTS[i]), linewidth=0.6, zorder=3, label=label)
        for v, py in pts:  # clipped points carry their value
            if floor is not None and v < floor:
                ax.annotate(f"{v:.0f}%", (floor, py), xytext=(4, 0), textcoords="offset points", fontsize=6,
                            color=MUTED, va="center")
    for j in set(range(len(ORDER))) - have:
        ax.text(0.5, y[j], "not run", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=6,
                color=MUTED)
    ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
    for yi in y[:-1]:
        ax.axhline(yi - 0.5, color=GRID, lw=0.5, zorder=0)
    ax.axhspan(-0.5, sum(n >= 21 for _, n in ORDER) - 0.5, color=BAND, zorder=0)  # 21 and more variates
    ax.set_yticks(y)
    if first:
        ax.set_yticklabels([f"{d} ({n})" for d, n in ORDER], fontsize=6.5, color=INK)
    ax.tick_params(axis="y", length=0, labelleft=first)
    ax.set_ylim(-0.5, len(ORDER) - 0.5)
    ax.set_title(title, fontsize=7.5, color=INK, loc="left", pad=24)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=6.5)
    ax.tick_params(axis="x", colors=MUTED, labelsize=6.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.1,
              columnspacing=0.6, borderaxespad=0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    R = {}
    for root in a.root:
        for k, v in load(root).items():
            seen = {r["seed"] for r in R.get(k, [])}
            R.setdefault(k, []).extend(r for r in v if r["seed"] not in seen)
    c = cells(R)
    apply_style(plt, 8)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.75), sharey=True)
    panel(axes[0], c, [("RLinear", "pact", "rlinear"), ("TimeCMA", "pact", "timecma"),
                       ("PatchTST", "pact", "patchtst"), ("iTransformer", "pact", "itransformer")],
          "(a) PACT vs. baseline", "rel. MSE (%), < 0: PACT better", floor=-32, first=True)
    panel(axes[1], c, [("w/o BDCA (prompt unused)", "abl_no_bdca", "abl_full"), ("one-way attention", "abl_xattn", "abl_full"),
                       ("w/o PDPL", "abl_no_pdpl", "abl_full")],
          "(b) Component removed or replaced", "rel. MSE (%), > 0: component helps")
    panel(axes[2], c, [("GPT-2", "llm_gpt2", "llm_none"), ("Qwen3-0.6B", "llm_qwen3", "llm_none"),
                       ("Sundial", "llm_sundial", "llm_none"), ("learned vectors", "llm_learned", "llm_none")],
          "(c) Anchor source vs. no anchor", "rel. MSE (%), < 0: anchor helps")
    axes[0].set_xlim(-34, axes[0].get_xlim()[1])
    fig.tight_layout(w_pad=0.6)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=200, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
