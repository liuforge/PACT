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

ORDER = [("ETTh1", 7), ("ETTh2", 7), ("ETTm1", 7), ("ETTm2", 7), ("ILI", 7), ("Weather", 21), ("ECL", 321),
         ("Traffic", 862)]
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 of the reference palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
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


def panel(ax, c, series, title, ylabel=None, floor=None):
    x = np.arange(len(ORDER))
    width = 0.8 / len(series)
    have = set()
    for i, (label, num, den) in enumerate(series):
        ys = [rel(c, ds, num, den) for ds, _ in ORDER]
        have |= {j for j, y in enumerate(ys) if y is not None}
        pts = [(xi - 0.4 + width * (i + 0.5), y) for xi, y in zip(x, ys) if y is not None]
        xs = [px for px, _ in pts]
        ys = [py if floor is None else max(py, floor) for _, py in pts]
        ax.scatter(xs, ys, s=30, marker=MARKERS[i], color=SLOTS[i], edgecolor="white", linewidth=0.8, zorder=3,
                   label=label)
        for (px, py) in pts:  # clipped points carry their value
            if floor is not None and py < floor:
                ax.annotate(f"{py:.0f}%", (px, floor), xytext=(4, 0), textcoords="offset points", fontsize=6.5,
                            color=MUTED, va="center")
    for j in set(range(len(ORDER))) - have:
        ax.text(j, 0, "not run", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color=GRID, lw=0.6, zorder=0)
    ax.axvspan(4.5, 5.5, color="#f4f3f0", zorder=0)
    ax.axvspan(5.5, 7.5, color="#ebe9e4", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n({n})" for d, n in ORDER], fontsize=6.5, color=MUTED)
    ax.set_xlim(-0.5, len(ORDER) - 0.5)
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=7.5)
    ax.tick_params(axis="y", colors=MUTED, labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left", handletextpad=0.2, borderaxespad=0.2)


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
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.4), sharex=True)
    panel(axes[0], c, [("vs. RLinear", "pact", "rlinear"), ("vs. TimeCMA", "pact", "timecma"),
                       ("vs. PatchTST", "pact", "patchtst"), ("vs. iTransformer", "pact", "itransformer")],
          "(a) PACT against the baselines (below 0: PACT better)", "relative MSE (%)", floor=-32)
    panel(axes[1], c, [("w/o BDCA", "abl_no_bdca", "abl_full"), ("one-way attention", "abl_xattn", "abl_full"),
                       ("w/o PDPL", "abl_no_pdpl", "abl_full")],
          "(b) Removing or replacing a component (above 0: the component helps)", "relative MSE (%)")
    panel(axes[2], c, [("GPT-2", "llm_gpt2", "llm_none"), ("Qwen3-0.6B", "llm_qwen3", "llm_none"),
                       ("Sundial", "llm_sundial", "llm_none"), ("learned vectors", "llm_learned", "llm_none")],
          "(c) Anchor source, relative to no anchor (below 0: the anchor helps)", "relative MSE (%)")
    for ax in axes:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -2), max(hi, 2))
    axes[0].set_ylim(-34, axes[0].get_ylim()[1])
    fig.text(0.99, 0.005, "datasets ordered by number of variates (in brackets); shaded: 21 and more variates",
             ha="right", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=200, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
