"""Small-multiple figure of projected prototypes (output of extract_prototypes.py).

One panel per prototype, ordered from low to high spectral centroid of its mean shape: the strongest
retrieved training patches (thin) and their median (thick), with the prototype's usage share.

  python tools/plot_prototypes.py viz/ETTh1.npz --out figures/prototypes_ETTh1.pdf
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = "#2a78d6"  # categorical slot 1 of the reference palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def spectral_centroid(y):
    mag = np.abs(np.fft.rfft(y - y.mean()))
    f = np.arange(len(mag))
    return float((f * mag).sum() / (mag.sum() + 1e-12))


def coherence(patches, centre):
    """Mean Pearson correlation between each patch and a reference shape."""
    a = patches - patches.mean(-1, keepdims=True)
    b = centre - centre.mean()
    num = (a * b).sum(-1)
    den = np.sqrt((a ** 2).sum(-1) * (b ** 2).sum()) + 1e-12
    return float(np.mean(num / den))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--out", required=True)
    p.add_argument("--show", type=int, default=24, help="patches drawn per panel")
    p.add_argument("--cols", type=int, default=4)
    a = p.parse_args()
    d = np.load(a.npz, allow_pickle=False)
    P, W, usage = d["patches"], d["weights"], d["usage"]
    K = P.shape[0]
    means = np.median(P, axis=1)  # robust centre of the retrieved patches
    order = np.argsort([spectral_centroid(m) for m in means])
    coh = np.mean([coherence(P[k], means[k]) for k in range(K)])
    if "random_patches" in d.files:
        R = d["random_patches"]
        base = np.mean([coherence(R, means[k]) for k in range(K)])
        print(f"coherence: retrieved patches {coh:.3f} vs random patches {base:.3f}")
    rows = int(np.ceil(K / a.cols))
    lim = np.percentile(np.abs(P[:, : a.show]), 99)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(rows, a.cols, figsize=(1.7 * a.cols, 1.25 * rows), sharex=True, sharey=True)
    for i, ax in enumerate(np.ravel(axes)):
        if i >= K:
            ax.axis("off")
            continue
        k = order[i]
        t = np.arange(P.shape[2])
        for j in range(min(a.show, P.shape[1])):
            ax.plot(t, P[k, j], color=SERIES, lw=0.5, alpha=0.18)
        ax.plot(t, means[k], color=SERIES, lw=2.0, solid_capstyle="round")
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"P{i + 1}  ({usage[k] * 100:.1f}%)", color=INK, fontsize=8, pad=2)
        ax.tick_params(colors=MUTED, length=2, labelsize=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    branch = str(d["branch"])
    fig.suptitle(f"{d['dataset']} — projected prototypes ({branch} stream), low → high frequency;"
                 f" retrieval entropy {float(d['entropy']):.2f}", color=INK, fontsize=9, y=1.0)
    fig.supxlabel("position within patch", color=MUTED, fontsize=8)
    fig.supylabel("normalised value", color=MUTED, fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(a.out, bbox_inches="tight")
    if a.out.endswith(".pdf"):
        fig.savefig(a.out[:-4] + ".png", dpi=200, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
